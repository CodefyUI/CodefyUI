"""The constraints file that makes a live pack install additive-only.

Installing a pack while the server runs is only safe if pip cannot REPLACE a
package the process has already imported: on Windows the loader holds the
``.pyd`` files open, and even where it does not, swapping numpy under a
running interpreter is a crash waiting for its first array. So every install
runs under a constraints file pinning every distribution in this interpreter
to the exact version it already has -- uv is then free to ADD what the pack
needs and free to do nothing else.

Two things therefore have to be true, and both are tested here: the pins must
cover everything (including local build tags like ``+cu128``, or the pin would
quietly authorise a swap from the CUDA wheel to the CPU one), and the file
must never pin the project itself -- CodefyUI is installed editable, is not on
any index, and a pin on it turns every install into an unsatisfiable resolve.
"""

from __future__ import annotations

import importlib.metadata

import pytest

from app.core.packs.constraints import (
    _canonical,
    constraints_text,
    installed_distributions,
    write_constraints_file,
)


class _FakeDist:
    """The parts of ``importlib.metadata.Distribution`` this module reads."""

    def __init__(self, name: str, version: str, direct_url: str | None = None):
        self.metadata = {"Name": name}
        self.version = version
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        return self._direct_url if filename == "direct_url.json" else None


def _yielding(*dists: _FakeDist):
    """A stand-in for ``importlib.metadata.distributions``."""
    return lambda **kwargs: iter(dists)


# ── reading this interpreter ─────────────────────────────────────────────


def test_pins_torch_at_the_version_uv_resolves_against():
    """The pin has to match what uv's resolver sees, not what torch reports.

    ``installed_distributions()`` reads ``importlib.metadata`` -- the same
    place uv resolves constraints against -- so its value has to be the
    metadata version, never ``torch.__version__`` itself: the two only
    coincide for an index wheel (e.g. a developer box's ``2.11.0+cu128``,
    where the local build tag is part of the metadata version too). PyPI's
    own torch wheels carry the local tag ONLY in ``torch.__version__``
    (``2.13.0+cu130`` on Linux, ``2.13.0+cpu`` on Windows) while their
    metadata version is the bare ``2.13.0`` -- so a pin built from
    ``torch.__version__`` would not match what uv already has installed and
    every subsequent resolve would fail. Either family is fine; the pin just
    has to be the version uv itself would report.
    """
    import torch

    dists = installed_distributions()

    assert dists["torch"] == importlib.metadata.version("torch")
    if "+" in torch.__version__:
        public_part = torch.__version__.split("+", 1)[0]
        assert dists["torch"] in (torch.__version__, public_part), (
            "torch's metadata version is neither the full local version "
            "(index wheels) nor its public part before the local tag "
            "(PyPI wheels)"
        )


def test_names_are_pep503_canonical():
    """``huggingface_hub`` and ``huggingface-hub`` are one distribution; the
    keys are the canonical spelling so a caller can look one up.

    The underscore half is proved through ``_canonical`` on a synthesised
    name rather than by naming a real package that happens to have one
    today. It used to assert ``"huggingface-hub" in dists`` -- a TRANSITIVE
    dependency, so the day a resolver stops pulling it this test would go
    red about PEP 503 canonicalisation for a reason that has nothing to do
    with it. What the real list is still asked is only that it exists.
    """
    dists = installed_distributions()

    assert dists, "this interpreter reported no distributions at all"
    for name in dists:
        assert name == name.lower()
        assert "_" not in name
        assert " " not in name

    assert _canonical("Some_Name") == "some-name"
    assert _canonical("zope.interface") == "zope-interface"


def test_the_editable_project_is_never_pinned():
    """CodefyUI installs itself editable and publishes to no index. A pin on
    it makes every subsequent resolve unsatisfiable.

    This venv is the reason the exclusion is by NAME rather than per record:
    the project shows up twice, once as a ``.egg-info`` next to
    ``pyproject.toml`` (which carries no ``direct_url.json`` and so does not
    look editable at all) and once as the real editable ``.dist-info``.
    Skipping only the record that admits to being editable leaves the other
    one behind.
    """
    assert "codefyui-backend" not in installed_distributions()


# ── the editable / duplicate rules, without depending on this venv ───────


@pytest.mark.parametrize("direct_url", [
    pytest.param('{"dir_info": {"editable": true}}', id="pep-610-nested"),
    pytest.param('{"editable": true}', id="flat"),
])
def test_skips_editable_project(monkeypatch, direct_url):
    monkeypatch.setattr(
        importlib.metadata, "distributions",
        _yielding(_FakeDist("codefyui-backend", "2.4.1", direct_url),
                  _FakeDist("requests", "2.32.3")))

    assert installed_distributions() == {"requests": "2.32.3"}


def test_keeps_a_non_editable_dist_with_a_direct_url(monkeypatch):
    """A wheel installed from a URL or a local path is still a real, pinnable
    distribution -- only ``editable`` marks the one that must not be pinned."""
    monkeypatch.setattr(
        importlib.metadata, "distributions",
        _yielding(_FakeDist("wheel-from-url", "1.0",
                            '{"url": "https://example/w.whl"}')))

    assert installed_distributions() == {"wheel-from-url": "1.0"}


def test_an_editable_record_excludes_every_record_of_that_name(monkeypatch):
    """The egg-info / dist-info pair from this venv, in miniature: one record
    hides the editable install, the other declares it. The name is out."""
    monkeypatch.setattr(
        importlib.metadata, "distributions",
        _yielding(_FakeDist("Codefyui_Backend", "2.4.1"),
                  _FakeDist("codefyui-backend", "2.4.1",
                            '{"dir_info": {"editable": true}}')))

    assert installed_distributions() == {}


def test_duplicate_names_keep_the_first_seen(monkeypatch):
    """Two records for one canonical name means one shadows the other on
    ``sys.path``; the first is the one this interpreter actually imported."""
    monkeypatch.setattr(
        importlib.metadata, "distributions",
        _yielding(_FakeDist("Typing_Extensions", "4.12.2"),
                  _FakeDist("typing-extensions", "3.0.0")))

    assert installed_distributions() == {"typing-extensions": "4.12.2"}


def test_skips_records_without_a_name_or_version(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata, "distributions",
        _yielding(_FakeDist("", "1.0"), _FakeDist("nameless", ""),
                  _FakeDist("real", "1.0")))

    assert installed_distributions() == {"real": "1.0"}


def test_a_broken_record_does_not_sink_the_whole_file(monkeypatch):
    """One unreadable ``.dist-info`` on disk must cost its own pin, not every
    other pin in the file."""
    class _Exploding:
        @property
        def metadata(self):
            raise OSError("unreadable METADATA")

    monkeypatch.setattr(
        importlib.metadata, "distributions",
        lambda **kwargs: iter([_Exploding(), _FakeDist("real", "1.0")]))

    assert installed_distributions() == {"real": "1.0"}


# ── rendering the file ───────────────────────────────────────────────────


def test_constraints_text_format():
    text = constraints_text({"requests": "2.32.3", "torch": "2.11.0+cu128"})

    assert text == "requests==2.32.3\ntorch==2.11.0+cu128\n"
    assert "\r" not in text


def test_constraints_text_of_nothing_is_empty():
    assert constraints_text({}) == ""


def test_constraints_text_defaults_to_this_interpreter():
    assert "\ntorch==" in "\n" + constraints_text()


@pytest.mark.parametrize("name", [
    pytest.param("-leading-dash", id="leading-dash"),
    pytest.param("space in name", id="space"),
    pytest.param("evil\n--index-url http://attacker", id="newline-injection"),
    # The one a `$`-anchored pattern used with `.match()` waves through: `$`
    # matches before a newline at the end of the string, so this name -- the
    # exact shape the guard exists to stop -- would emit "evil\n==1.0".
    pytest.param("evil\n", id="trailing-newline"),
    pytest.param("", id="empty"),
])
def test_constraints_text_drops_unusable_names(name):
    """The file is handed to uv as options-and-requirements text. Nothing
    here should ever produce a name with a newline in it -- and if something
    ever does, it must not become a second line uv reads as a flag."""
    text = constraints_text({name: "1.0", "requests": "2.32.3"})

    assert text == "requests==2.32.3\n"


@pytest.mark.parametrize("version", [
    pytest.param("1.0 --index-url http://attacker", id="space"),
    pytest.param("1.0\n--index-url http://attacker", id="newline"),
    pytest.param("", id="empty"),
])
def test_constraints_text_drops_unusable_versions(version):
    text = constraints_text({"sneaky": version, "requests": "2.32.3"})

    assert text == "requests==2.32.3\n"


# ── writing the file ─────────────────────────────────────────────────────


def test_write_constraints_file_returns_path_in_directory(tmp_path):
    path = write_constraints_file(tmp_path, {"requests": "2.32.3"})

    assert path == tmp_path / "constraints.txt"
    assert path.parent == tmp_path


def test_written_file_uses_lf_and_utf8(tmp_path):
    """Read as BYTES on purpose: this is a Windows-first project, and the
    default newline translation would turn every pin into a CRLF line."""
    path = write_constraints_file(tmp_path, {"requests": "2.32.3", "a": "1.0"})

    assert path.read_bytes() == b"a==1.0\nrequests==2.32.3\n"


def test_write_constraints_file_defaults_to_this_interpreter(tmp_path):
    path = write_constraints_file(tmp_path)

    assert "\ntorch==" in "\n" + path.read_text(encoding="utf-8")
    assert "codefyui-backend==" not in path.read_text(encoding="utf-8")


def test_write_constraints_file_does_not_create_its_directory(tmp_path):
    """The CALLER owns the directory -- a per-job temporary one it makes and
    cleans up -- so a missing one is the caller's bug and has to be raised
    rather than papered over. A ``mkdir(parents=True)`` here would leave a
    stray directory behind on every such bug, in a place nothing cleans."""
    gone = tmp_path / "gone"

    with pytest.raises(OSError):
        write_constraints_file(gone)

    assert not gone.exists()
