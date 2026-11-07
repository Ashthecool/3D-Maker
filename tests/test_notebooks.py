from pathlib import Path

import nbformat

NOTEBOOKS = sorted((Path(__file__).parent.parent / "notebooks").glob("*.ipynb"))


def test_notebooks_exist():
    assert NOTEBOOKS


def test_notebooks_are_valid():
    for path in NOTEBOOKS:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
