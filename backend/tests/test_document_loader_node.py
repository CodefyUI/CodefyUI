"""Tests for DocumentLoaderNode -- the first node of the RAG chain.

Filesystem-only, no network and no optional pack: this node reads .txt and
.md files and nothing else, so every test builds its own tiny corpus under
``tmp_path``. The one exception is
``test_bundled_samples_resolve_from_any_cwd``, which reads the corpus that
actually ships in ``backend/data/samples/rag`` -- that file IS the contract
the RAG examples depend on ("open the example, press Run, no setup"), so it
is asserted against the real thing rather than a fixture.

Two resolution rules get their own tests because both are invisible until
they break on someone else's machine: a source string is POSIX even on
Windows (a chunk citation must read the same in every screenshot), and a
relative ``directory`` falls back to the CodefyUI backend folder so the
bundled samples are found whatever directory the server was started from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import settings
from app.nodes.llm.document_loader_node import DocumentLoaderNode


def _run(**params) -> dict:
    return DocumentLoaderNode().execute({}, params)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# -- metadata ------------------------------------------------------------


def test_node_metadata():
    assert DocumentLoaderNode.NODE_NAME == "DocumentLoader"
    assert DocumentLoaderNode.CATEGORY == "LLM"
    # No pack: the whole point of this node is that the RAG chain starts
    # working before anything is installed.
    assert DocumentLoaderNode.REQUIRES_PACK is None
    assert DocumentLoaderNode.cacheable is True
    assert DocumentLoaderNode.define_inputs() == []
    assert [p.name for p in DocumentLoaderNode.define_outputs()] == [
        "documents", "texts", "count"]

    params = {p.name: p for p in DocumentLoaderNode.define_params()}
    assert params["source"].options == ["directory", "uploaded_file"]
    assert params["source"].default == "directory"
    assert params["directory"].default == "data/samples/rag"
    for name in ("directory", "recursive"):
        assert params[name].visible_when == {"source": "directory"}, name
    assert params["file"].visible_when == {"source": "uploaded_file"}
    # max_docs applies to both sources, so it must NOT be conditional.
    assert params["max_docs"].visible_when is None


# -- directory mode ------------------------------------------------------


def test_reads_txt_and_md_in_name_order_and_skips_other_suffixes(tmp_path):
    _write(tmp_path / "b.txt", "second")
    _write(tmp_path / "a.md", "first")
    _write(tmp_path / "c.png", "not text")
    _write(tmp_path / "d.json", '{"not": "text"}')

    result = _run(directory=str(tmp_path))

    assert [d["source"] for d in result["documents"]] == ["a.md", "b.txt"]
    assert [d["text"] for d in result["documents"]] == ["first", "second"]
    assert result["texts"] == ["first", "second"]
    assert result["count"] == 2


def test_source_is_posix_relative_path(tmp_path):
    """A citation must read ``sub/note.md`` on Windows too.

    ``relative_to(root)`` alone gives a WindowsPath whose str() is
    ``sub\\note.md``; the source string ends up in printed answers and in
    saved run output, where a backslash would make the same graph look
    different depending on the machine that ran it.
    """
    _write(tmp_path / "sub" / "note.md", "nested")

    result = _run(directory=str(tmp_path), recursive=True)

    assert [d["source"] for d in result["documents"]] == ["sub/note.md"]


def test_non_recursive_by_default(tmp_path):
    _write(tmp_path / "top.md", "top level")
    _write(tmp_path / "sub" / "note.md", "nested")

    result = _run(directory=str(tmp_path))

    assert [d["source"] for d in result["documents"]] == ["top.md"]


def test_empty_and_whitespace_files_are_skipped(tmp_path):
    """An empty document is worse than no document.

    It survives chunking as a zero-length chunk, embeds to a vector that is
    close to nothing in particular, and then turns up in retrieval results
    as a citation with no text under it.
    """
    _write(tmp_path / "a-empty.md", "")
    _write(tmp_path / "b-blank.txt", "   \n\n\t\n")
    _write(tmp_path / "c-real.md", "real content")

    result = _run(directory=str(tmp_path))

    assert [d["source"] for d in result["documents"]] == ["c-real.md"]
    assert result["count"] == 1


def test_invalid_utf8_names_the_file(tmp_path):
    _write(tmp_path / "fine.md", "readable")
    (tmp_path / "broken.md").write_bytes(b"valid start \xff\xfe then not")

    with pytest.raises(ValueError, match="broken.md"):
        _run(directory=str(tmp_path))


def test_no_matching_files_raises_friendly_error(tmp_path):
    _write(tmp_path / "notes.pdf", "not plain text")

    with pytest.raises(FileNotFoundError, match="found no .txt or .md files"):
        _run(directory=str(tmp_path))


def test_max_docs_caps_in_name_order(tmp_path):
    for name in ("a.md", "b.md", "c.md", "d.md"):
        _write(tmp_path / name, f"content of {name}")

    result = _run(directory=str(tmp_path), max_docs=2)

    assert [d["source"] for d in result["documents"]] == ["a.md", "b.md"]
    assert result["count"] == 2


# -- uploaded_file mode --------------------------------------------------


def test_uploaded_file_resolves_against_data_files_dir(tmp_path, monkeypatch):
    """A bare filename is what the upload dropdown produces.

    It must resolve against DATA_FILES_DIR, never against the server's
    working directory -- picking a file from the dropdown cannot depend on
    where the server happened to be started.
    """
    monkeypatch.setattr(settings, "DATA_FILES_DIR", tmp_path)
    _write(tmp_path / "uploaded.txt", "one uploaded note")

    result = _run(source="uploaded_file", file="uploaded.txt")

    assert result["documents"] == [
        {"text": "one uploaded note", "source": "uploaded.txt"}]
    assert result["count"] == 1


# -- path safety ---------------------------------------------------------


def test_project_mode_refuses_escape(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    _write(tmp_path / "outside" / "secret.md", "not yours")
    monkeypatch.setattr(settings, "PROJECT_DIR", project)

    with pytest.raises(ValueError) as excinfo:
        _run(directory="../outside")

    message = str(excinfo.value)
    assert "escapes the project directory" in message
    # Names the PARAM, so the reader knows which field to fix.
    assert "DocumentLoader: directory" in message
    assert "../outside" in message


def test_project_mode_still_allows_the_bundled_samples(tmp_path, monkeypatch):
    """The one exemption from the rule above, mirroring CSVReader.

    ``data/samples/...`` names the install, not the project, so a project
    opened anywhere still loads the shipped corpus instead of being told it
    escaped.
    """
    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    result = _run()

    assert result["count"] == 5


# -- the bundled corpus --------------------------------------------------


def test_bundled_samples_resolve_from_any_cwd(tmp_path, monkeypatch):
    """Default params, from a directory that has no ``data/`` in it at all.

    This is the "press Run and it works" guarantee: the RAG examples ship
    with these defaults, and the server's working directory is whatever the
    person who started it happened to be in.
    """
    monkeypatch.chdir(tmp_path)

    result = _run()

    assert result["count"] == 5
    assert len(result["documents"]) == 5
    assert [d["source"] for d in result["documents"]] == sorted(
        d["source"] for d in result["documents"])
    for doc in result["documents"]:
        assert doc["source"].endswith(".md"), doc["source"]
        text = doc["text"]
        assert any(ch.isascii() and ch.isalpha() for ch in text), doc["source"]
        # CJK Unified Ideographs -- the Chinese half of every sample.
        assert any("一" <= ch <= "鿿" for ch in text), doc["source"]


# -- cache fingerprint ---------------------------------------------------


def test_cache_fingerprint_changes_when_a_file_is_edited(tmp_path):
    """Without this the node would serve the first run's documents forever.

    ``params`` records WHERE to read; only the fingerprint records what was
    there, so an edited note has to move the key.
    """
    note = _write(tmp_path / "note.md", "first version")
    params = {"source": "directory", "directory": str(tmp_path)}

    before = DocumentLoaderNode.cache_fingerprint(params)
    note.write_text("second version", encoding="utf-8")
    after = DocumentLoaderNode.cache_fingerprint(params)

    assert before is not None
    assert before != after
