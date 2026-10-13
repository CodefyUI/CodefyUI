"""A sweep over KFoldSplit's ``fold`` is k-fold cross-validation (#712).

#712's last acceptance bullet asks that a regression metric be recorded for
sweeps, and its k-fold item asks for a split a sweep can turn. This runs
both through the real ``POST /api/sweeps`` route and the real queue:

    CSVReader(target_type=number) → KFoldSplit(fold) → LinearRegression
        → RegressionMetrics

swept over ``fold`` = 0..3, ranked by ``r2``. Each variant must record
``mse``, ``mae`` and ``r2`` as run metrics, each variant's objective must
equal sklearn's R² on that fold, and the mean over the variants must equal
``sklearn.model_selection.cross_val_score`` with the same ``KFold``.

Fixtures are local, following the house rule that test modules stay
independent (test_sweep_ranks_classical_and_rl.py builds the same service).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core.auth import TOKEN_HEADER, session_token
from app.core.db import Database
from app.core.run_service import QueueLimits, RunService
from app.core.run_store import RunStore
from app.core.sweep_store import SweepStore
from app.main import app

BASE_URL = f"http://127.0.0.1:{settings.PORT}"
_FOLDS = 4
_ROWS = 60
_SEED = 0


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "codefyui.db")
    database.connect()
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
async def client(db):
    store = RunStore(db)
    service = RunService(
        store, shutdown_grace_s=5.0,
        limits=QueueLimits(cpu=2, gpu=1, interactive=2, overrides=()))
    app.state.db = db
    app.state.run_service = service
    app.state.sweep_store = SweepStore(db)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url=BASE_URL,
            headers={TOKEN_HEADER: session_token()},
        ) as http:
            yield http, store
    finally:
        await service.shutdown()
        for attribute in ("db", "run_service", "sweep_store"):
            if hasattr(app.state, attribute):
                delattr(app.state, attribute)


def _write_csv(path: Path) -> Path:
    """y = 1.5 a - 2 b + 4 plus enough noise that each fold's R² differs."""
    rng = np.random.default_rng(7)
    x = rng.uniform(-2.0, 2.0, size=(_ROWS, 2))
    y = 1.5 * x[:, 0] - 2.0 * x[:, 1] + 4.0 + 0.8 * rng.standard_normal(_ROWS)
    lines = ["a,b,y"] + [f"{a:.6f},{b:.6f},{t:.6f}" for (a, b), t in zip(x, y)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _edge(source: str, source_handle: str, target: str, target_handle: str) -> dict:
    return {"id": f"{source}.{source_handle}->{target}.{target_handle}",
            "source": source, "sourceHandle": source_handle,
            "target": target, "targetHandle": target_handle}


def _graph(csv_path: Path) -> dict[str, Any]:
    def node(node_id: str, node_type: str, **params) -> dict:
        return {"id": node_id, "type": node_type, "data": {"params": params}}

    return {
        "nodes": [
            node("start", "Start"),
            node("csv", "CSVReader", path=str(csv_path), target_column="y",
                 target_type="number"),
            node("kfold", "KFoldSplit", n_folds=_FOLDS, fold=0, shuffle=True,
                 seed=_SEED),
            node("ols", "LinearRegression"),
            node("metrics", "RegressionMetrics"),
        ],
        "edges": [
            {"id": "t-csv", "source": "start", "target": "csv",
             "sourceHandle": "trigger", "type": "trigger"},
            _edge("csv", "tensor", "kfold", "features"),
            _edge("csv", "labels", "kfold", "labels"),
            _edge("kfold", "x_train", "ols", "x_train"),
            _edge("kfold", "y_train", "ols", "y_train"),
            _edge("kfold", "x_val", "ols", "x_query"),
            _edge("ols", "predictions", "metrics", "predictions"),
            _edge("kfold", "y_val", "metrics", "targets"),
        ],
    }


def _reference(csv_path: Path) -> tuple[list[float], float]:
    """Per-fold R² and their mean, computed by sklearn alone."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold, cross_val_score

    table = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    x, y = table[:, :2], table[:, 2]
    cv = KFold(n_splits=_FOLDS, shuffle=True, random_state=_SEED)
    per_fold = [
        float(r2_score(y[val], LinearRegression().fit(x[train], y[train]).predict(x[val])))
        for train, val in cv.split(x)
    ]
    mean = float(cross_val_score(LinearRegression(), x, y, cv=cv, scoring="r2").mean())
    return per_fold, mean


async def _run_sweep(http: AsyncClient, store: RunStore,
                     body: dict[str, Any]) -> dict[str, Any]:
    response = await http.post("/api/sweeps", json=body)
    assert response.status_code == 201, response.text
    sweep_id = response.json()["sweep_id"]
    deadline = time.monotonic() + 60.0
    for child in await store.list_runs_by_sweep(sweep_id):
        while (await store.get_run(child.id)).finished_at is None:
            assert time.monotonic() < deadline, "a sweep variant never finished"
            await asyncio.sleep(0.05)
    result = await http.get(f"/api/sweeps/{sweep_id}")
    assert result.status_code == 200, result.text
    return result.json()


async def test_a_sweep_over_fold_is_kfold_cross_validation(client, tmp_path):
    http, store = client
    csv_path = _write_csv(tmp_path / "noisy_linear.csv")
    per_fold, cv_mean = _reference(csv_path)
    # The folds must differ, or matching them proves little.
    assert len({round(v, 6) for v in per_fold}) == _FOLDS

    body = await _run_sweep(http, store, {
        "base_graph": _graph(csv_path),
        "sweep_spec": {"method": "grid", "seed": 5, "params": [
            {"node_id": "kfold", "param": "fold",
             "values": list(range(_FOLDS))}]},
        # Different run seeds per variant must not move the folds.
        "seed_variants": True,
        "objective": {"metric": "r2", "direction": "maximize"},
        "options": {"device": "cpu"},
        "name": "4-fold cv",
    })

    assert body["counts"]["succeeded"] == _FOLDS
    assert "objective_warning" not in body
    variants = body["variants"]
    by_fold: dict[int, float] = {}
    for variant in variants:
        # RegressionMetrics recorded all three as run metrics.
        assert {"mse", "mae", "r2"} <= set(variant["final_metrics"]), variant
        assert variant["objective_node_id"] == "metrics"
        fold = variant["params"][0]["value"]
        by_fold[fold] = variant["objective"]
        assert variant["objective"] == pytest.approx(per_fold[fold], abs=1e-4)
        assert variant["final_metrics"]["r2"] == pytest.approx(per_fold[fold], abs=1e-4)

    assert sorted(by_fold) == list(range(_FOLDS))
    assert np.mean(list(by_fold.values())) == pytest.approx(cv_mean, abs=1e-4)
    # Ranked best first, as for any sweep.
    assert [v["rank"] for v in variants] == [1, 2, 3, 4]
    assert [v["objective"] for v in variants] == sorted(by_fold.values(), reverse=True)
