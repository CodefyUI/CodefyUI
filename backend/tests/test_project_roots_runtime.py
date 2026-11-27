"""Phase 4A: the project-derived roots are derived by one function, used at
construction and by a run-time switch, with the same precedence."""

from pathlib import Path

from app.config import PROJECT_DERIVED_ROOTS, Settings


def _clean(monkeypatch):
    for name in ("PROJECT_DIR", *PROJECT_DERIVED_ROOTS):
        monkeypatch.delenv(f"CODEFYUI_{name}", raising=False)


def test_flat_mode_keeps_the_class_defaults(monkeypatch):
    _clean(monkeypatch)
    s = Settings()
    for name in PROJECT_DERIVED_ROOTS:
        assert getattr(s, name) == Settings.model_fields[name].default
    assert s.PROJECT_DIR is None and s.LAYOUT_DIR is None


def test_construction_and_run_time_derivation_agree(monkeypatch, tmp_path):
    _clean(monkeypatch)
    monkeypatch.setenv("CODEFYUI_PROJECT_DIR", str(tmp_path / "a"))
    built = Settings()
    monkeypatch.delenv("CODEFYUI_PROJECT_DIR")
    switched = Settings()
    switched.derive_project_roots(tmp_path / "a")
    for name in ("PROJECT_DIR", *PROJECT_DERIVED_ROOTS):
        assert getattr(built, name) == getattr(switched, name), name
    assert built.GRAPHS_DIR == (tmp_path / "a").resolve() / "graphs"
    assert built.MEDIA_DIR == (tmp_path / "a").resolve() / "assets" / "media"


def test_an_explicit_root_survives_every_switch(monkeypatch, tmp_path):
    _clean(monkeypatch)
    monkeypatch.setenv("CODEFYUI_PROJECT_DIR", str(tmp_path / "a"))
    monkeypatch.setenv("CODEFYUI_MODELS_DIR", str(tmp_path / "pinned"))
    s = Settings()
    for target in (tmp_path / "b", None, tmp_path / "c"):
        s.derive_project_roots(target)
        assert s.MODELS_DIR == tmp_path / "pinned"
    assert s.GRAPHS_DIR == (tmp_path / "c").resolve() / "graphs"


def test_switching_back_to_flat_restores_the_defaults(monkeypatch, tmp_path):
    _clean(monkeypatch)
    s = Settings()
    s.derive_project_roots(tmp_path / "p")
    assert s.LAYOUT_DIR == (tmp_path / "p").resolve() / "layout"
    s.derive_project_roots(None)
    for name in PROJECT_DERIVED_ROOTS:
        assert getattr(s, name) == Settings.model_fields[name].default
    assert s.PROJECT_DIR is None


def test_derived_roots_are_not_mistaken_for_pinned_ones(monkeypatch, tmp_path):
    """Derivation assigns the roots, which adds them to model_fields_set;
    a second derivation must still move them."""
    _clean(monkeypatch)
    monkeypatch.setenv("CODEFYUI_PROJECT_DIR", str(tmp_path / "a"))
    s = Settings()
    s.derive_project_roots(tmp_path / "b")
    assert s.GRAPHS_DIR == Path(tmp_path / "b").resolve() / "graphs"
