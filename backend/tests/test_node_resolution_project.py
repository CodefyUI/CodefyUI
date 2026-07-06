"""Project-mode node file resolution (spec 7.2): FileReader/CSVReader resolve
under the project, DatasetNode downloads into the project, all confined."""

from pathlib import Path

import pytest

from app.config import settings
from app.nodes.data.csv_reader_node import CSVReaderNode
from app.nodes.data.dataset_node import DatasetNode
from app.nodes.io.file_reader_node import FileReaderNode


@pytest.fixture
def project(monkeypatch, tmp_path):
    (tmp_path / "assets" / "data").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "PROJECT_DIR", tmp_path.resolve())
    return tmp_path.resolve()


def test_filereader_relative_resolves_under_assets_data(project):
    (project / "assets" / "data" / "note.txt").write_text("hello", encoding="utf-8")
    out = FileReaderNode().execute({}, {"path": "note.txt", "mode": "text"})
    assert out["text"] == "hello"


def test_filereader_escape_is_rejected(project):
    with pytest.raises(ValueError):
        FileReaderNode().execute({}, {"path": "../../secret.txt", "mode": "text"})


def test_csvreader_relative_resolves_under_project(project):
    (project / "mydata.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    out = CSVReaderNode().execute({}, {"path": "mydata.csv"})
    assert out["tensor"].shape[0] == 2


def test_csvreader_escape_is_rejected(project):
    with pytest.raises(ValueError):
        CSVReaderNode().execute({}, {"path": "../escape.csv"})


def test_csvreader_iris_default_special_cased_to_install(project):
    # The bundled sample resolves against the install CWD (backend/), not the
    # project, so demos keep working with no copy. cwd is backend/ under pytest.
    if not Path("data/samples/iris.csv").exists():
        pytest.skip("iris sample not present in this checkout")
    out = CSVReaderNode().execute(
        {}, {"path": "data/samples/iris.csv", "target_column": "species"})
    assert out["tensor"].shape[0] > 0


def test_datasetnode_relative_data_dir_remaps_into_project(project, monkeypatch):
    captured = {}

    class _FakeDS:
        def __init__(self, root, train, download, transform):
            captured["root"] = root

    monkeypatch.setattr("torchvision.datasets.MNIST", _FakeDS)
    DatasetNode().execute({}, {"name": "MNIST", "split": "train", "data_dir": "./data"})
    assert Path(captured["root"]) == project / "assets" / "data"


def test_datasetnode_absolute_data_dir_is_honored(project, monkeypatch, tmp_path):
    captured = {}

    class _FakeDS:
        def __init__(self, root, train, download, transform):
            captured["root"] = root

    monkeypatch.setattr("torchvision.datasets.MNIST", _FakeDS)
    abs_dir = str(tmp_path / "elsewhere")
    DatasetNode().execute({}, {"name": "MNIST", "data_dir": abs_dir})
    assert captured["root"] == abs_dir
