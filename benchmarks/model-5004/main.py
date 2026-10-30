import argparse
import math
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import xara
import opensees.openseespy as ops


GRAVITY = 9.81

PARABOLIC_VS_PROFILE = (
    473.7, 469.6, 465.4, 461.2, 456.7, 452.2, 447.5, 442.7, 437.7, 432.5,
    427.1, 421.5, 415.7, 409.6, 403.3, 396.6, 389.6, 382.2, 374.3, 365.9,
    356.8, 347.0, 336.4, 324.5, 311.3, 296.0, 278.0, 255.6, 224.9, 170.9,
)

MODULUS_REDUCTION_CURVE = (
    (1.0e-6, 1.000), (2.0e-6, 1.000), (5.0e-6, 0.996),
    (1.0e-5, 0.984), (2.0e-5, 0.975), (5.0e-5, 0.922),
    (1.0e-4, 0.850), (2.0e-4, 0.734), (5.0e-4, 0.532),
    (1.0e-3, 0.367), (2.0e-3, 0.224), (5.0e-3, 0.139),
    (1.0e-2, 0.085), (2.0e-2, 0.051), (5.0e-2, 0.027),
    (1.0e-1, 0.021),
)

NODE_RESPONSES = (
    ("displacement", "disp"),
    ("velocity", "vel"),
    ("acceleration", "accel"),
)

ELEMENT_RESPONSES = ("stress", "strain")

GAUSS_POINTS = (1, 2, 3, 4)


@dataclass(frozen=True)
class Layer:
    thickness: float
    vs: float


@dataclass(frozen=True)
class PressureIndependSoil:
    cohesion: float
    peak_shear_strain: float
    friction_angle: float = 0.0
    reference_pressure: float = 100.0
    pressure_coefficient: float = 0.0
    num_yield_surfaces: int = 20
    backbone: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class PressureDependSoil:
    friction_angle: float
    peak_shear_strain: float
    reference_pressure: float
    pressure_coefficient: float
    phase_transformation_angle: float
    contraction: float
    dilation: tuple[float, float]
    liquefaction: tuple[float, float, float]
    num_yield_surfaces: int = 20


@dataclass(frozen=True)
class SiteProfile:
    layers: tuple[Layer, ...]
    density: float
    poisson_ratio: float
    soil: PressureIndependSoil | PressureDependSoil
    rock_vs: float
    rock_density: float


@dataclass(frozen=True)
class AnalysisSettings:
    max_frequency: float = 100.0
    elements_per_wavelength: int = 8
    damping_ratio: float = 0.02
    damping_frequencies: tuple[float, float] = (0.2, 20.0)
    newmark_gamma: float = 0.5
    newmark_beta: float = 0.25
    gravity_tolerance: float = 1.0e-5
    gravity_iterations: int = 30
    gravity_time_step: float = 5.0e2
    elastic_gravity_steps: int = 10
    plastic_gravity_steps: int = 40
    dynamic_tolerance: float = 1.0e-3
    dynamic_iterations: int = 15


@dataclass(frozen=True)
class SiteCase:
    profile: SiteProfile
    settings: AnalysisSettings


@dataclass(frozen=True)
class Mesh:
    elements_per_layer: tuple[int, ...]
    element_heights: tuple[float, ...]
    width: float

    @property
    def num_elements(self) -> int:
        return sum(self.elements_per_layer)

    @property
    def num_rows(self) -> int:
        return self.num_elements + 1

    @property
    def num_nodes(self) -> int:
        return 2 * self.num_rows


def single_layer_case() -> SiteCase:
    soil = PressureIndependSoil(
        cohesion=95.0,
        peak_shear_strain=0.05,
        friction_angle=0.0,
        reference_pressure=80.0,
        pressure_coefficient=0.0,
        num_yield_surfaces=25,
    )
    profile = SiteProfile(
        layers=(Layer(thickness=40.0, vs=250.0),),
        density=1.7,
        poisson_ratio=0.0,
        soil=soil,
        rock_vs=760.0,
        rock_density=2.4,
    )
    return SiteCase(profile, AnalysisSettings(elements_per_wavelength=10))


def parabolic_profile(soil) -> SiteProfile:
    return SiteProfile(
        layers=tuple(Layer(thickness=1.0, vs=vs) for vs in PARABOLIC_VS_PROFILE),
        density=2.202,
        poisson_ratio=0.0,
        soil=soil,
        rock_vs=762.0,
        rock_density=2.396,
    )


def layered_pressure_independ_case() -> SiteCase:
    soil = PressureIndependSoil(
        cohesion=95.0,
        peak_shear_strain=0.1,
        friction_angle=0.0,
        reference_pressure=80.0,
        pressure_coefficient=0.0,
        backbone=MODULUS_REDUCTION_CURVE,
    )
    return SiteCase(parabolic_profile(soil), 
                    AnalysisSettings(elements_per_wavelength=8))


def layered_pressure_depend_case() -> SiteCase:
    soil = PressureDependSoil(
        friction_angle=35.0,
        peak_shear_strain=0.1,
        reference_pressure=80.0,
        pressure_coefficient=0.0,
        phase_transformation_angle=27.0,
        contraction=0.06,
        dilation=(0.5, 2.5),
        liquefaction=(0.0, 0.0, 0.0),
        num_yield_surfaces=16,
    )

    return SiteCase(parabolic_profile(soil),
                    AnalysisSettings(elements_per_wavelength=8, dynamic_iterations=35))


CASES = {
    "single":   single_layer_case,
    "independ": layered_pressure_independ_case,
    "depend":   layered_pressure_depend_case,
}




def rayleigh_coefficients(damping_ratio: float, frequencies: tuple[float, float]) -> tuple[float, float]:
    omega_low, omega_high = (2.0 * math.pi * f for f in frequencies)
    mass_coefficient = 2.0 * damping_ratio * omega_low * omega_high / (omega_low + omega_high)
    stiffness_coefficient = 2.0 * damping_ratio / (omega_low + omega_high)
    return mass_coefficient, stiffness_coefficient


def dashpot_coefficient(width: float, rock_density: float, rock_vs: float) -> float:
    return width * rock_density * rock_vs


def target_element_size(profile: SiteProfile, settings: AnalysisSettings) -> float:
    min_vs = min(layer.vs for layer in profile.layers)
    wavelength = min_vs / settings.max_frequency
    return wavelength / settings.elements_per_wavelength


def build_mesh(profile: SiteProfile, settings: AnalysisSettings) -> Mesh:
    size = target_element_size(profile, settings)
    counts = tuple(math.ceil(layer.thickness / size) for layer in profile.layers)
    heights = tuple(layer.thickness / count for layer, count in zip(profile.layers, counts))
    return Mesh(elements_per_layer=counts, element_heights=heights, width=min(heights))


def node_elevations(mesh: Mesh) -> list[float]:
    increments = (
        height
        for height, count in zip(mesh.element_heights, mesh.elements_per_layer)
        for _ in range(count)
    )
    return list(accumulate(increments, initial=0.0))


def element_material_tags(mesh: Mesh) -> list[int]:
    return [
        tag
        for tag, count in enumerate(mesh.elements_per_layer, start=1)
        for _ in range(count)
    ]


def left_node(row: int) -> int:
    return 2 * row + 1


def right_node(row: int) -> int:
    return 2 * row + 2


def dynamic_time_step(motion_dt: float, motion_steps: int, width: float, max_vs: float) -> tuple[int, float]:
    trial_dt = width / math.sqrt(max_vs)
    if motion_dt <= trial_dt:
        return motion_steps, motion_dt
    duration = motion_dt * motion_steps
    num_steps = math.floor(duration / trial_dt) + 1
    return num_steps, duration / num_steps


def add_soil_nodes(model, mesh: Mesh):
    for row, elevation in enumerate(node_elevations(mesh)):
        model.node(left_node(row), 0.0, elevation)
        model.node(right_node(row), mesh.width, elevation)


def add_base_fixity(model):
    model.fix(left_node(0), 0, 1)
    model.fix(right_node(0), 0, 1)


def add_simple_shear_constraints(model, mesh: Mesh):
    for row in range(1, mesh.num_rows):
        model.equalDOF(left_node(row), right_node(row), 1, 2)
    model.equalDOF(left_node(0), right_node(0), 1)


def yield_surface_arguments(soil: PressureIndependSoil) -> tuple[float, ...]:
    if not soil.backbone:
        return (soil.num_yield_surfaces,)
    points = tuple(value for point in soil.backbone for value in point)
    return (-len(soil.backbone), *points)


def add_pressure_independ_material(model, tag: int, density: float, shear_modulus: float,
                                   bulk_modulus: float, soil: PressureIndependSoil):
    model.nDMaterial(
        "PressureIndependMultiYield", tag, 2, density, shear_modulus, bulk_modulus,
        soil.cohesion, soil.peak_shear_strain, soil.friction_angle,
        soil.reference_pressure, soil.pressure_coefficient,
        *yield_surface_arguments(soil),
    )


def add_pressure_depend_material(model, tag: int, density: float, shear_modulus: float,
                                 bulk_modulus: float, soil: PressureDependSoil):
    model.nDMaterial(
        "PressureDependMultiYield", tag, 2, density, shear_modulus, bulk_modulus,
        soil.friction_angle, soil.peak_shear_strain, soil.reference_pressure,
        soil.pressure_coefficient, soil.phase_transformation_angle, soil.contraction,
        *soil.dilation, *soil.liquefaction, soil.num_yield_surfaces,
    )




def add_soil_material(model, 
                      tag: int, 
                      density: float, 
                      layer: Layer, 
                      poisson_ratio: float,
                      soil: PressureIndependSoil | PressureDependSoil):

    shear_modulus  = density*layer.vs**2
    youngs_modulus = 2.0 * shear_modulus * (1.0 + poisson_ratio)
    bulk_modulus   = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))

    if isinstance(soil, PressureIndependSoil):
        add_pressure_independ_material(model, tag, density, shear_modulus, bulk_modulus, soil)
    elif isinstance(soil, PressureDependSoil):
        add_pressure_depend_material(model, tag, density, shear_modulus, bulk_modulus, soil)
    else:
        raise TypeError(f"unsupported soil model {type(soil).__name__}")


def add_soil_materials(model, profile: SiteProfile) -> list[int]:
    tags = []
    for tag, layer in enumerate(profile.layers, start=1):
        add_soil_material(model, tag, profile.density, layer, profile.poisson_ratio, profile.soil)
        tags.append(tag)
    return tags


def add_soil_elements(model, mesh: Mesh, density: float):
    body_force = -GRAVITY * density
    for row, material_tag in enumerate(element_material_tags(mesh)):
        model.element(
            "quad", row + 1,
            left_node(row), right_node(row), right_node(row + 1), left_node(row + 1),
            1.0, "PlaneStrain", material_tag, 0.0, density, 0.0, body_force,
        )


def add_compliant_base(model, 
                       base_node: int, 
                       coefficient: float, 
                       fixed_node: int = 2000,
                       free_node: int = 2001, material_tag: int = 4000, element_tag: int = 5000):
    model.node(fixed_node, 0.0, 0.0)
    model.node(free_node, 0.0, 0.0)
    model.fix(fixed_node, 1, 1)
    model.fix(free_node, 0, 1)
    model.equalDOF(base_node, free_node, 1)
    model.uniaxialMaterial("Viscous", material_tag, coefficient, 1.0)
    model.element("zeroLength", element_tag, fixed_node, free_node, mat=material_tag, dir=1)


def build_soil_column(model, profile: SiteProfile, mesh: Mesh) -> list[int]:
    add_soil_nodes(model, mesh)
    add_base_fixity(model)
    add_simple_shear_constraints(model, mesh)
    material_tags = add_soil_materials(model, profile)
    add_soil_elements(model, mesh, profile.density)
    coefficient = dashpot_coefficient(mesh.width, profile.rock_density, profile.rock_vs)
    add_compliant_base(model, left_node(0), coefficient)
    return material_tags


def recorder_interval_arguments(dt: float | None) -> tuple:
    return () if dt is None else ("-dT", dt)


def add_node_recorders(model, directory: Path, prefix: str, last_node: int, dt: float | None = None):
    for name, response in NODE_RESPONSES:
        path = directory / f"{prefix}{name}.out"
        model.recorder(
            "Node", "-file", str(path), "-time", *recorder_interval_arguments(dt),
            "-nodeRange", 1, last_node, "-dof", 1, 2, response,
        )


def add_element_recorders(model, directory: Path, prefix: str, last_element: int, dt: float | None = None):
    for response in ELEMENT_RESPONSES:
        for point in GAUSS_POINTS:
            path = directory / f"{prefix}{response}{point}.out"
            model.recorder(
                "Element", "-file", str(path), "-time", *recorder_interval_arguments(dt),
                "-eleRange", 1, last_element, "material", str(point), response,
            )


def add_recorders(model, directory: Path, prefix: str, mesh: Mesh, dt: float | None = None):
    add_node_recorders(model, directory, prefix, mesh.num_nodes, dt)
    add_element_recorders(model, directory, prefix, mesh.num_elements, dt)


def configure_transient_analysis(model, tolerance: float, max_iterations: int, gamma: float, beta: float):
    model.constraints("Transformation")
    model.test("NormDispIncr", tolerance, max_iterations, 1)
    model.algorithm("Newton")
    model.numberer("RCM")
    model.system("ProfileSPD")
    model.integrator("Newmark", gamma, beta)
    model.analysis("Transient")


def advance(model, num_steps: int, dt: float):
    status = model.analyze(num_steps, dt)
    if status != 0:
        raise RuntimeError(f"analysis failed with status {status} at time {model.getTime()}")



def run_gravity_analysis(model, material_tags: list[int], settings: AnalysisSettings):
    configure_transient_analysis(
        model, settings.gravity_tolerance, settings.gravity_iterations,
        settings.newmark_gamma, settings.newmark_beta,
    )
    advance(model, settings.elastic_gravity_steps, settings.gravity_time_step)
    for tag in material_tags:
        model.updateMaterialStage("-material", tag, stage=1)
    advance(model, settings.plastic_gravity_steps, settings.gravity_time_step)


def reset_analysis(model):
    model.setTime(0.0)
    model.wipeAnalysis()


def add_base_excitation(model, velocity: np.ndarray, dt: float, factor: float, node: int = 1,
                        series_tag: int = 10, pattern_tag: int = 10):
    model.timeSeries("Path", series_tag, "-dt", dt, "-factor", factor, "-values", *velocity)
    model.pattern("Plain", pattern_tag, series_tag)
    model.load(node, 1.0, 0.0)


def run_dynamic_analysis(model, num_steps: int, dt: float, settings: AnalysisSettings):
    mass_coefficient, stiffness_coefficient = rayleigh_coefficients(
        settings.damping_ratio, settings.damping_frequencies
    )
    model.rayleigh(mass_coefficient, stiffness_coefficient, 0.0, 0.0)
    configure_transient_analysis(
        model, settings.dynamic_tolerance, settings.dynamic_iterations,
        settings.newmark_gamma, settings.newmark_beta,
    )
    advance(model, num_steps, dt)


def run_site_response(model, case: SiteCase, velocity: np.ndarray, motion_dt: float, output_dir: Path) -> Mesh:
    profile, settings = case.profile, case.settings
    mesh = build_mesh(profile, settings)
    print(f"elements: {mesh.num_elements}  nodes: {mesh.num_nodes}  element width: {mesh.width}")

    material_tags = build_soil_column(model, profile, mesh)

    add_recorders(model, output_dir, "G", mesh)
    run_gravity_analysis(model, material_tags, settings)
    print("gravity analysis complete")

    reset_analysis(model)
    add_recorders(model, output_dir, "", mesh, motion_dt)
    coefficient = dashpot_coefficient(mesh.width, profile.rock_density, profile.rock_vs)
    add_base_excitation(model, velocity, motion_dt, coefficient)

    max_vs = max(layer.vs for layer in profile.layers)
    num_steps, dt = dynamic_time_step(motion_dt, len(velocity), mesh.width, max_vs)
    print(f"dynamic analysis: {num_steps} steps at dt = {dt}")
    run_dynamic_analysis(model, num_steps, dt, settings)
    print("dynamic analysis complete")
    return mesh



def acceleration_to_velocity(acceleration_g: np.ndarray, dt: float) -> np.ndarray:
    acceleration = acceleration_g * GRAVITY
    increments = 0.5 * (acceleration[1:] + acceleration[:-1])
    return dt * np.concatenate(([0.0], np.cumsum(increments)))


def write_velocity_history(path: Path, velocity: np.ndarray):
    np.savetxt(path, velocity, fmt="%16.7e")


def load_node_history(path: Path, num_dof: int = 2) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, ndmin=2)
    time = data[:, 0]
    response = data[:, 1:].reshape(len(time), -1, num_dof)
    return time, response


def sdof_transfer_function(omega: np.ndarray, mass: float, damping: float, stiffness: float) -> np.ndarray:
    size = omega.size
    positive = np.arange((size - 1) // 2 + 1)
    transfer = np.zeros(size + 1, dtype=complex)
    transfer[positive] = 1.0 / (-mass * omega[positive] ** 2 + 1j * damping * omega[positive] + stiffness)
    transfer[size - positive] = np.conj(transfer[positive])
    return transfer[:size]


def response_spectra(acceleration_g: np.ndarray, duration: float, periods: np.ndarray | None = None,
                     damping_ratio: float = 0.05, stiffness: float = 1000.0):
    if periods is None:
        periods = np.logspace(-3, 1, 100)
    acceleration = np.concatenate(([0.0], acceleration_g)) * GRAVITY
    omega = (2.0 * math.pi / duration) * np.arange(acceleration.size)
    acceleration_fft = np.fft.fft(acceleration)

    displacement = np.empty(periods.size)
    for j, period in enumerate(periods):
        mass = (period / (2.0 * math.pi)) ** 2 * stiffness
        damping = 2.0 * damping_ratio * math.sqrt(stiffness * mass)
        transfer = sdof_transfer_function(omega, mass, damping, stiffness)
        history = np.real(np.fft.ifft(transfer * (-mass * acceleration_fft)))
        displacement[j] = np.max(np.abs(history))

    velocity = (2.0 * math.pi / periods) * displacement
    acceleration_spectrum = (2.0 * math.pi / periods) * velocity / GRAVITY
    return periods, displacement, velocity, acceleration_spectrum


def plot_acceleration_history(time: np.ndarray, acceleration_g: np.ndarray, path: Path):
    fig, ax = plt.subplots()
    ax.plot(time, acceleration_g, "-b", linewidth=1.5)
    ax.grid(True)
    ax.set_xlabel("time (sec)", fontsize=16)
    ax.set_ylabel("acceleration (g)", fontsize=16)
    ax.tick_params(labelsize=14)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_response_spectra(periods: np.ndarray, displacement: np.ndarray, velocity: np.ndarray,
                          acceleration: np.ndarray, path: Path):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6.4, 8.0))
    series = (
        (acceleration, "$S_a$  (g)"),
        (velocity, "$S_v$  (m/s)"),
        (displacement, "$S_d$  (m)"),
    )
    for ax, (values, label) in zip(axes, series):
        ax.semilogx(periods, values, "b", linewidth=1.5)
        ax.grid(True, which="both")
        ax.set_ylabel(label, fontsize=16)
        ax.tick_params(labelsize=14)
    axes[-1].set_xlabel("Period, T (sec)", fontsize=16)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_surface_response(acceleration_file: Path, output_dir: Path, extension: str = "eps"):
    time, acceleration = load_node_history(acceleration_file)
    surface = acceleration[:, -1, 0] / GRAVITY
    plot_acceleration_history(time, surface, output_dir / f"surfaceAccel.{extension}")
    periods, displacement, velocity, spectral_acceleration = response_spectra(surface, time[-1])
    plot_response_spectra(
        periods, displacement, velocity, spectral_acceleration,
        output_dir / f"logSpectra.{extension}",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Site response analysis of a soil column on an elastic half-space."
    )
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--motion", type=Path, default=Path("GilroyNo1EW.out"))
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--output-dir", type=Path, default=Path("./out/"))
    parser.add_argument("--figure-format", default="png")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def load_ground_motion(path: Path) -> np.ndarray:
    return np.array(Path(path).read_text().split(), dtype=float)


if __name__ == "__main__":
    args = parse_arguments()
    case = CASES[args.case]()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    velocity = acceleration_to_velocity(load_ground_motion(args.motion), args.dt)
    write_velocity_history(args.output_dir / "velocityHistory.out", velocity)

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    try:
        run_site_response(ops, case, velocity, args.dt, args.output_dir)
    finally:
        ops.wipe()

    if not args.no_plot:
        plot_surface_response(args.output_dir / "acceleration.out", 
                              args.output_dir, args.figure_format)

