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


def test_pins_every_installed_distribution_with_local_tag():
    """The pin has to carry the LOCAL version tag.

    ``2.11.0+cu128`` and ``2.11.0+cpu`` are the same version to a resolver
    that only sees ``2.11.0``: pinning the short form lets an install "keep"
    torch while silently replacing a CUDA build with a CPU one, which is the
    single worst thing this file exists to prevent.
    """
    import torch

    dists = installed_distributions()

    assert dists["torch"] == torch.__version__
    if "+" in torch.__version__:
        assert "+" in dists["torch"], "the local build tag was stripped off"


def test_names_are_pep503_canonical():
    """``huggingface_hub`` and ``huggingface-hub`` are one distribution; the
    keys are the canonical spelling so a caller can look one up."""
    dists = installed_distributions()

    for name in dists:
        assert name == name.lower()
        assert "_" not in name
        assert " " not in name

    assert "huggingface-hub" in dists, "underscores were not canonicalised"


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
