import xara


materials = [
    {
        "type": "J2BeamThread", 
        "json": ["E"]
    }
]

def check_multiaxial(data):
    materials = {
        type: xara.MultiaxialMaterial(type, **data).asdict()
        for type in ["J2BeamThread", "J2Plasticity"]
    }
    for type, mdata in materials.items():
        print(mdata)
        assert mdata["Fy"] == data["Fy"]
        assert isinstance(mdata["name"], int)

        if "density" in data:
            assert mdata["density"] == data["density"]

    assert materials["J2BeamThread"]["E"] == data["E"]
    assert materials["J2BeamThread"]["nu"] == data["E"] / (2 * data["G"]) - 1


def test_yield():
    data = {
        "E": 200e9,
        "G": 80e9,
        "Fy": 250e6,
    }
    check_multiaxial(data)

def test_density():
    data = {
        "E": 200e9,
        "G": 80e9,
        "Fy": 250e6,
        "density": 7850,
    }
    check_multiaxial(data)

if __name__ == "__main__":
    test_yield()
    test_density()
