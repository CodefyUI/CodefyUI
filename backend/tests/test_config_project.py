"""Project-mode Settings derivation (spec 7.1): PROJECT_DIR repoints the
graph/asset roots, with an explicitly-provided root always winning."""

from pathlib import Path

from app.config import Settings


def test_no_project_dir_keeps_install_defaults():
    s = Settings()
    assert s.PROJECT_DIR is None
    assert s.LAYOUT_DIR is None
    # Install-relative defaults are untouched.
    assert s.GRAPHS_DIR.name == "graphs"
    assert s.MODELS_DIR.name == "models"


def test_project_dir_derives_all_roots(tmp_path):
    s = Settings(PROJECT_DIR=tmp_path)
    proj = tmp_path.resolve()
    assert s.PROJECT_DIR == proj
    assert s.GRAPHS_DIR == proj / "graphs"
    assert s.IMAGES_DIR == proj / "assets" / "images"
    assert s.MODELS_DIR == proj / "assets" / "models"
    assert s.LAYOUT_DIR == proj / "layout"


def test_explicit_graphs_dir_beats_project_derivation(tmp_path):
    other = tmp_path / "custom-graphs"
    s = Settings(PROJECT_DIR=tmp_path, GRAPHS_DIR=other)
    # Explicit CODEFYUI_GRAPHS_DIR (here via init kwarg) wins...
    assert s.GRAPHS_DIR == other
    # ...but the roots NOT explicitly set still derive from the project.
    assert s.MODELS_DIR == tmp_path.resolve() / "assets" / "models"
    assert s.IMAGES_DIR == tmp_path.resolve() / "assets" / "images"


def test_models_dir_parent_is_assets(tmp_path):
    # The io nodes' confinement invariant: assets/models is a direct child of
    # assets/ (spec 7.2). Guard it here so a future refactor can't break it.
    s = Settings(PROJECT_DIR=tmp_path)
    assert s.MODELS_DIR.parent == tmp_path.resolve() / "assets"
