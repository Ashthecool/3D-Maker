import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "colab_mcp_launcher.py"
_spec = importlib.util.spec_from_file_location("colab_mcp_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launcher)

ID = "1hiBnEs7Qk2GrJIXfILu-Ly-1DqQOo2Wf"


@pytest.mark.parametrize(
    "value",
    [
        f"https://colab.research.google.com/drive/{ID}",
        f"https://colab.research.google.com/drive/{ID}#mcpProxyToken=x&mcpProxyPort=1",
        f"https://colab.google.com/drive/{ID}?usp=sharing",
        f"drive/{ID}",
        f"/drive/{ID}\n",
        ID,
    ],
)
def test_drive_notebook_forms(value):
    assert launcher.notebook_path(value) == f"/drive/{ID}"


def test_other_paths_and_errors():
    assert launcher.notebook_path("/notebooks/empty.ipynb") == "/notebooks/empty.ipynb"
    assert launcher.notebook_path("  ") is None
    with pytest.raises(ValueError):
        launcher.notebook_path("https://evil.example/drive/x")


def test_env_wins_over_file(tmp_path):
    f = tmp_path / ".colab-notebook"
    f.write_text("drive/fromfile_aaaaaaaaaaaaaaaaaaaa\n", encoding="utf-8")
    assert launcher.configured_path({}, f) == "/drive/fromfile_aaaaaaaaaaaaaaaaaaaa"
    assert launcher.configured_path({"COLAB_NOTEBOOK": ID}, f) == f"/drive/{ID}"
    assert launcher.configured_path({}, tmp_path / "missing") is None
