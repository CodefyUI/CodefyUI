"""Tests for PrintNode."""

from __future__ import annotations

from app.nodes.utility.print_node import PrintNode


def test_node_metadata():
    assert PrintNode.NODE_NAME == "Print"
    assert PrintNode.CATEGORY == "Utility"


def test_passes_value_through():
    res = PrintNode().execute({"value": 42}, {})
    assert res["value"] == 42


def test_log_includes_value():
    res = PrintNode().execute({"value": "hello"}, {})
    assert "__log__" in res
    assert "hello" in res["__log__"]


def test_label_prefix_in_log(capsys):
    res = PrintNode().execute({"value": "world"}, {"label": "greeting"})
    assert res["__log__"] == "[greeting] world"
    captured = capsys.readouterr()
    assert "[greeting] world" in captured.out


def test_empty_label_omits_prefix():
    res = PrintNode().execute({"value": 1}, {"label": ""})
    assert res["__log__"] == "1"


def test_passes_through_complex_objects():
    obj = {"key": [1, 2, 3]}
    res = PrintNode().execute({"value": obj}, {})
    assert res["value"] is obj


def test_handles_missing_input():
    res = PrintNode().execute({}, {})
    # value is None when not provided
    assert res["value"] is None


# ── Consoles that cannot spell what the graph carries (I4 #cp950) ──────────
#
# `print` encodes with the console's codepage. On a Traditional Chinese
# Windows install that is cp950, which covers Han characters but not, say,
# a superscript T -- and a run died at the Print node because of one
# character in a label. These pin the trade: the console degrades, the run
# and the value do not.


class _NarrowStdout:
    """A stdout whose encoding rejects anything outside cp950."""

    encoding = "cp950"

    def __init__(self) -> None:
        self.written: list[str] = []

    def write(self, s: str) -> int:
        s.encode(self.encoding)  # raises exactly as the real stream does
        self.written.append(s)
        return len(s)

    def flush(self) -> None:
        pass


def test_label_outside_console_codepage_does_not_fail_the_node(monkeypatch):
    """U+1D40 in a label used to raise UnicodeEncodeError out of execute."""
    out = _NarrowStdout()
    monkeypatch.setattr("sys.stdout", out)
    res = PrintNode().execute({"value": 1}, {"label": "Q\u00b7K\u1d40 scores"})
    assert res["value"] == 1
    assert "".join(out.written).strip()  # something reached the console


def test_value_outside_console_codepage_does_not_fail_the_node(monkeypatch):
    """The likelier case: generated text carrying an un-encodable character."""
    monkeypatch.setattr("sys.stdout", _NarrowStdout())
    res = PrintNode().execute({"value": "once upon a time \U0001f600"}, {})
    assert res["value"] == "once upon a time \U0001f600"


def test_log_keeps_the_exact_text_the_console_could_not(monkeypatch):
    """Only the console rendering is lossy; the UI log is verbatim."""
    monkeypatch.setattr("sys.stdout", _NarrowStdout())
    res = PrintNode().execute({"value": "x"}, {"label": "K\u1d40"})
    assert res["__log__"] == "[K\u1d40] x"


def test_a_broken_stdout_does_not_fail_the_node(monkeypatch):
    class _Exploding:
        encoding = "utf-8"

        def write(self, s: str) -> int:
            raise ValueError("I/O operation on closed file")

        def flush(self) -> None:
            pass

    monkeypatch.setattr("sys.stdout", _Exploding())
    res = PrintNode().execute({"value": 7}, {"label": "ok"})
    assert res["value"] == 7
    assert res["__log__"] == "[ok] 7"


def test_plain_ascii_still_reaches_a_narrow_console_unchanged(monkeypatch):
    out = _NarrowStdout()
    monkeypatch.setattr("sys.stdout", out)
    PrintNode().execute({"value": "hello"}, {"label": "greeting"})
    assert "[greeting] hello" in "".join(out.written)


def test_han_characters_are_not_mangled_on_cp950(monkeypatch):
    """cp950 covers Chinese -- the fix must not degrade what already worked."""
    out = _NarrowStdout()
    monkeypatch.setattr("sys.stdout", out)
    PrintNode().execute({"value": 1}, {"label": "原始分數矩陣"})
    assert "原始分數矩陣" in "".join(out.written)
