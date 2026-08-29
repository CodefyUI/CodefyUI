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
from app.nodes.llm.document_loader_node import (
    DocumentLoaderNode,
    _resolve_document_path,
)


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


def test_a_bom_never_reaches_a_chunk(tmp_path):
    """Notepad's byte-order mark is invisible and travels a long way.

    A UTF-8 BOM is a legal file whose first character is U+FEFF. Read as
    plain utf-8 it survives into the first chunk, into its embedding and
    into the first citation the model is shown -- a corruption with no
    symptom anybody can see. ``utf-8-sig`` eats it.
    """
    (tmp_path / "notepad.md").write_bytes(
        b"\xef\xbb\xbfWhat is CodefyUI?")
    (tmp_path / "plain.md").write_text("No mark here.", encoding="utf-8")

    result = _run(directory=str(tmp_path))

    assert [d["source"] for d in result["documents"]] == [
        "notepad.md", "plain.md"]
    assert [d["text"] for d in result["documents"]] == [
        "What is CodefyUI?", "No mark here."]
    # Spelled with the escape on purpose: the literal character is invisible
    # in every editor, so a raw one here would be a test nobody can read and
    # that a careless reformat could silently delete.
    assert not any("\ufeff" in d["text"] for d in result["documents"])


def test_no_matching_files_raises_friendly_error(tmp_path):
    _write(tmp_path / "notes.pdf", "not plain text")

    with pytest.raises(FileNotFoundError, match="found no .txt or .md files"):
        _run(directory=str(tmp_path))


def test_a_folder_of_only_blank_files_is_an_error(tmp_path):
    """Zero documents from a folder that HAS files is not a result.

    Every blank file is skipped one at a time, so a folder of them loads
    nothing and says so cheerfully; the first real complaint then comes from
    VectorStore, two nodes downstream, about an empty embedding matrix --
    which is true and names the wrong node.
    """
    _write(tmp_path / "a.md", "")
    _write(tmp_path / "b.txt", "   \n\t\n")

    with pytest.raises(FileNotFoundError) as excinfo:
        _run(directory=str(tmp_path))

    message = str(excinfo.value)
    assert "found only blank files" in message
    assert str(tmp_path) in message

    # One real file among them is enough: the folder is not empty, and the
    # blanks are skipped as before.
    _write(tmp_path / "c.md", "something to say")
    assert _run(directory=str(tmp_path))["count"] == 1


def test_a_missing_folder_names_the_param(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        _run(directory=str(tmp_path / "nowhere"))

    message = str(excinfo.value)
    assert "DocumentLoader: no folder at" in message
    assert "nowhere" in message
    assert "`directory`" in message


def test_an_empty_directory_param_points_at_the_bundled_corpus(tmp_path):
    """A cleared text box is not a folder, and the message says what to do.

    Resolving "" would give the working directory, which is whatever the
    server was started in -- a folder of someone's source code loaded as a
    corpus is a worse outcome than an error.
    """
    with pytest.raises(ValueError) as excinfo:
        _run(directory="   ")

    message = str(excinfo.value)
    assert "DocumentLoader has no folder set" in message
    assert "data/samples/rag" in message
    assert "uploaded_file" in message


def test_a_missing_uploaded_file_names_the_param(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_FILES_DIR", tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        _run(source="uploaded_file", file="not-uploaded.txt")

    message = str(excinfo.value)
    assert "DocumentLoader: no file at" in message
    assert "`file`" in message

    # Nothing picked at all is a different sentence: there is no path to
    # name, so it names the dropdown and the upload button instead.
    with pytest.raises(ValueError) as unset:
        _run(source="uploaded_file", file="")
    assert "no file selected" in str(unset.value)


def test_unknown_source_lists_the_two():
    """A hand-edited graph, or one saved by a newer version. Defaulting to
    ``directory`` would read a folder the caller never asked for."""
    with pytest.raises(ValueError) as excinfo:
        _run(source="s3")

    assert str(excinfo.value) == (
        "DocumentLoader: unknown source 's3'; set the `source` param to one "
        "of ['directory', 'uploaded_file'].")


def test_documents_sort_case_sensitively_by_their_posix_source(tmp_path):
    """Sorted by the citation string, not by ``Path`` order.

    ``Path`` comparison folds case on Windows and does not on Linux, so a
    folder with mixed-case names loads in one order there and the opposite
    one here -- and ``max_docs`` then keeps DIFFERENT documents depending on
    the machine, which is the same graph giving two answers.
    """
    for name in ("apple.md", "Banana.md", "Cherry.md", "date.md"):
        _write(tmp_path / name, f"content of {name}")

    result = _run(directory=str(tmp_path))

    # Plain ASCII order: every capital sorts before every lower-case letter.
    assert [d["source"] for d in result["documents"]] == [
        "Banana.md", "Cherry.md", "apple.md", "date.md"]
    capped = _run(directory=str(tmp_path), max_docs=2)
    assert [d["source"] for d in capped["documents"]] == [
        "Banana.md", "Cherry.md"]


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


def test_a_blank_uploaded_file_says_the_file_is_blank(tmp_path, monkeypatch):
    """One picked file that is empty is not "zero documents loaded".

    The folder case already refuses a corpus of blanks; a single blank file
    is the same dead end with an even clearer culprit, so it gets the same
    answer instead of quietly handing an empty list to the chunker.
    """
    monkeypatch.setattr(settings, "DATA_FILES_DIR", tmp_path)
    _write(tmp_path / "empty.txt", "   \n\n\t")

    with pytest.raises(FileNotFoundError) as excinfo:
        _run(source="uploaded_file", file="empty.txt")

    message = str(excinfo.value)
    assert "empty.txt" in message
    assert "the file is blank" in message


def test_an_uploaded_file_of_the_wrong_kind_names_the_suffixes(
        tmp_path, monkeypatch):
    """The upload button takes anything; this node reads two suffixes.

    Read as text a PDF is a page of binary noise, and it would chunk, embed
    and be retrieved exactly like a real passage -- so the suffix is refused
    by name, with the two that do work in the same sentence.
    """
    monkeypatch.setattr(settings, "DATA_FILES_DIR", tmp_path)
    _write(tmp_path / "report.pdf", "%PDF-1.4 pretend binary")

    with pytest.raises(ValueError) as excinfo:
        _run(source="uploaded_file", file="report.pdf")

    message = str(excinfo.value)
    assert "report.pdf" in message
    assert ".pdf" in message
    assert ".md and .txt" in message


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


def test_project_mode_refuses_a_dotted_path_wearing_the_samples_prefix(
        tmp_path, monkeypatch):
    """The ``data/samples/`` exemption is a PREFIX test, so it needs the
    second half: no component may be ``..``.

    ``data/samples/../../../elsewhere`` starts with the exempt prefix and
    ends outside the project entirely, and without the ``..`` half it would
    skip the project check the exemption exists to sidestep for the shipped
    corpus alone.
    """
    project = tmp_path / "project"
    project.mkdir()
    _write(tmp_path / "outside" / "secret.md", "not yours")
    monkeypatch.setattr(settings, "PROJECT_DIR", project)

    with pytest.raises(ValueError) as excinfo:
        _run(directory="data/samples/../../../outside")

    assert "escapes the project directory" in str(excinfo.value)

    # The Windows spelling of the same trick, since the prefix test
    # normalises backslashes before it looks.
    with pytest.raises(ValueError):
        _run(directory="data\\samples\\..\\..\\..\\outside")


def test_dot_paths_never_reach_the_uploads_directory(tmp_path, monkeypatch):
    """``"."`` and ``".."`` are not bare filenames, whatever pathlib says.

    ``Path(".").name`` is ``""`` and ``Path("..").name`` is ``".."``, and
    both have ``"."`` for a parent -- so the upload-dropdown rule used to
    accept them and hand back ``DATA_FILES_DIR`` itself (every file anyone
    ever uploaded) or the folder above it, BEFORE the project guard could
    object.
    """
    uploads = tmp_path / "uploads"
    _write(uploads / "someone-elses.md", "private upload")
    monkeypatch.setattr(settings, "DATA_FILES_DIR", uploads)

    project = tmp_path / "project"
    _write(project / "own-note.md", "inside the project")
    monkeypatch.setattr(settings, "PROJECT_DIR", project)

    # ".." leaves the project, so it is refused by name.
    with pytest.raises(ValueError) as excinfo:
        _run(directory="..")
    message = str(excinfo.value)
    assert "escapes the project directory" in message
    assert "DocumentLoader: directory" in message
    assert "'..'" in message

    # "." IS the project directory -- legitimately inside it -- so it
    # resolves there rather than to the uploads folder.
    result = _run(directory=".")
    assert [d["source"] for d in result["documents"]] == ["own-note.md"]


def test_a_dot_directory_outside_project_mode_is_the_working_directory(
        tmp_path, monkeypatch):
    """The same rule with no project: ``"."`` means the cwd, as typed."""
    uploads = tmp_path / "uploads"
    _write(uploads / "someone-elses.md", "private upload")
    monkeypatch.setattr(settings, "DATA_FILES_DIR", uploads)
    monkeypatch.setattr(settings, "PROJECT_DIR", None)

    here = tmp_path / "here"
    _write(here / "local.md", "beside the server")
    monkeypatch.chdir(here)

    result = _run(directory=".")

    assert [d["source"] for d in result["documents"]] == ["local.md"]


def test_a_windows_spelling_of_a_relative_folder_resolves(
        tmp_path, monkeypatch):
    """``data\\samples\\rag`` is the same folder as ``data/samples/rag``.

    A graph saved on Windows carries the backslash spelling, and on POSIX
    ``Path("data\\samples\\rag")`` is ONE component whose name happens to
    contain backslashes -- so without normalising before resolving, the
    shipped example opens on a Linux machine and finds nothing.
    """
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_document_path("data\\samples\\rag", kind="directory")

    assert resolved.is_dir()
    assert resolved.parts[-3:] == ("data", "samples", "rag")
    assert _run(directory="data\\samples\\rag")["count"] == 5


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
