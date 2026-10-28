"""Parameter-file parsing: literals, expressions, templating, comments, overrides, and errors."""

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import BASE_PARAMS
from vof.constants import arcsec
from vof.load_param_file import load_param_file, load_params


def write(tmp_path, text, name="run.param"):
    """Write a parameter file and return its path."""
    path = tmp_path / name
    path.write_text(text)
    return path


def test_value_types(tmp_path):
    """Numbers, lists, booleans, None, and bare words/paths parse to the right types."""
    params = load_param_file(
        write(
            tmp_path,
            "n = 512\nres = 5.2\nangles = [0, 45]\nflag = True\ncores = None\n"
            "name = m12m\npath = /data/sims/\n",
        )
    )
    assert params == {
        "n": 512,
        "res": 5.2,
        "angles": [0, 45],
        "flag": True,
        "cores": None,
        "name": "m12m",
        "path": "/data/sims/",
    }


def test_expressions_and_templates_use_earlier_keys(tmp_path):
    """Values can do arithmetic on earlier keys and constants, or template them into strings."""
    params = load_param_file(
        write(
            tmp_path,
            "res_km_s = 5.0\nbandwidth = 128*res_km_s\nbeam = 6*arcsec\n"
            "galName = m12m\nfileDir = /sims/{galName}/\n",
        )
    )
    assert params["bandwidth"] == 640.0
    assert params["beam"] == pytest.approx(6 * arcsec)
    assert params["fileDir"] == "/sims/m12m/"


def test_comments_and_blank_lines(tmp_path):
    """Trailing comments and blank lines are ignored, but '#' inside quotes is kept."""
    params = load_param_file(
        write(
            tmp_path,
            "# header comment\n\nn = 3   # trailing comment\nlabel = 'run #2'\n",
        )
    )
    assert params == {"n": 3, "label": "run #2"}


def test_override_reaches_later_templates(tmp_path):
    """An override replaces the file's value, including where later lines template it in."""
    params = load_param_file(
        write(tmp_path, "galName = m12m\noutput = /out/{galName}/\n"),
        overrides={"galName": "m12i", "minSnap": 500, "maxSnap": None},
    )
    assert params == {"galName": "m12i", "output": "/out/m12i/", "minSnap": 500}


def test_malformed_line_reports_line_number(tmp_path):
    """A line without '=' raises ValueError naming the file and line."""
    with pytest.raises(ValueError, match=r"run\.param:2"):
        load_param_file(write(tmp_path, "n = 3\nthis line is wrong\n"))


def test_failed_expression_raises(tmp_path):
    """An expression that fails to evaluate (not just a bare string) raises ValueError."""
    with pytest.raises(ValueError, match="could not evaluate"):
        load_param_file(write(tmp_path, "n = 1/0\n"))


def test_load_params(tmp_path):
    """load_params adds the .param suffix, wraps array fields, and applies the CLI overrides."""
    write(
        tmp_path,
        "galName = m12m\ninclinations = [45, 55]\nobserverVelocity = [0, 0, 0]\n",
    )
    params = load_params(str(tmp_path / "run"), galName="m12f", inclination=30)
    assert params["galName"] == "m12f"
    assert isinstance(params["observerVelocity"], np.ndarray)
    assert params["inclinations"] == [30]


def test_template_parses():
    """The shipped template parameter file parses and sets every parameter a run needs."""
    template = Path(__file__).parents[1] / "configs" / "param_template.param"
    params = load_params(str(template))
    missing = {*BASE_PARAMS, "fileDir", "statsDir", "output"} - set(params)
    assert not missing, f"param_template.param is missing {sorted(missing)}"
