"""
In opensees, doubly-constraining a node prints an error message
but does not throw one. Here we test that this behavior is 
preserved by xara's "compatibility modes", and throws a real error in 
standard mode.
"""

import pytest
import xara 
import os
import subprocess



def test_xcp_0005_python():
    # dont throw error on double SP constraint
    def attempt():
        m = xara.Model(ndm=1,ndf=1)
        m.node(1,0.0)
        m.fix(1,1)
        m.fix(1,1)

    try:
        attempt()
        assert False, "Failed to throw error"
    except:
        assert True


    os.environ["XARA_COMPATIBILITY_VERSION"] = "O3"
    try:
        attempt()
        assert True 
    except:
        assert False



def test_xcp_0005_stdin():
    import sys

    script = """
        model basic -ndm 1 -ndf 1
        node 1 0.0
        fix 1 1
        fix 1 1
    """

    # The Python test may have enabled compatibility mode in this process.
    env = os.environ.copy()
    env.pop("XARA_COMPATIBILITY_VERSION", None)

    standard = subprocess.run(
        [sys.executable, "-m", "xara"], input=script, text=True, env=env, capture_output=True
    )
    assert standard.returncode != 0, "Standard mode should reject a duplicate constraint"

    # The opensees entry point selects compatibility mode.
    compatible = subprocess.run(
        [sys.executable, "-m", "opensees"], input=script, text=True, env=env, capture_output=True
    )
    assert compatible.returncode == 0, compatible.stderr


def test_xcp_0005_file(tmp_path):
    import sys

    script = tmp_path / "double_constraint.tcl"
    script.write_text("""
        model basic -ndm 1 -ndf 1
        node 1 0.0
        fix 1 1
        fix 1 1
    """)

    # The Python test may have enabled compatibility mode in this process.
    env = os.environ.copy()
    env.pop("XARA_COMPATIBILITY_VERSION", None)

    standard = subprocess.run(
        [sys.executable, "-m", "xara", str(script)], env=env, capture_output=True
    )
    assert standard.returncode != 0, "Standard mode should reject a duplicate constraint"

    # The opensees entry point selects compatibility mode.
    compatible = subprocess.run(
        [sys.executable, "-m", "opensees", str(script)], env=env, capture_output=True
    )
    assert compatible.returncode == 0, compatible.stderr.decode(errors="replace")