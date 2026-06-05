#!/usr/bin/env python3
"""跨裝置煙霧測試：把範例 graph 強制跑在指定裝置上（cpu / cuda / mps）。

用途：multi-backend 工作流程的「真機驗證」工具。把 device-aware 範例的
device 參數覆寫成目標裝置後實際執行，藉此確認該裝置（尤其 Apple MPS、
NVIDIA CUDA、AMD ROCm）能端到端跑完訓練/推論路徑。

用法：
    python scripts/device_smoke.py                 # 自動挑最佳可用裝置
    python scripts/device_smoke.py mps             # 指定裝置
    python scripts/device_smoke.py cuda --full     # 不限制 epochs（完整跑）
    python scripts/device_smoke.py mps a.json b.json   # 指定自訂 graph

預設會挑選 repo 內「會實際使用裝置」的範例（含 TrainingLoop / Inference /
Checkpoint / ModelLoader 這類 sink 節點）。device 不相容（例如 MPS 不支援
float64）會在第一個 batch 就拋錯，因此預設把 epochs 上限設為 1 以加速。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = _REPO_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# Datasets default to a relative ``./data`` dir; the server runs from backend/,
# so chdir there too and the already-downloaded MNIST/CIFAR are reused instead
# of re-fetched.
os.chdir(_BACKEND)

from app.config import settings  # noqa: E402
from app.core.device_utils import get_available_devices  # noqa: E402
from app.core.graph_engine import execute_graph  # noqa: E402
from app.core.logging_config import setup_logging  # noqa: E402
from app.core.node_registry import registry  # noqa: E402
from app.core.preset_registry import preset_registry  # noqa: E402

# device 會實際生效的範例（透過 sink 節點 .to(device)）。其餘純前向範例沒有
# device 參數，永遠跑 CPU，故不在預設清單內。
_DEFAULT_EXAMPLES = [
    "examples/Usage_Example/GPT-Mini/TrainGPT-Mini/graph.json",
    "examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json",
    "examples/Usage_Example/ResNet-CIFAR10/TrainResNet-CIFAR10/graph.json",
]


def _best_device() -> str:
    avail = get_available_devices()
    for preferred in ("cuda", "mps"):
        if preferred in avail:
            return preferred
    return "cpu"


def patch_device(graph: dict, device: str, cap_epochs: int | None) -> int:
    """把每個 device 參數（含 preset internalParams）覆寫成 device，回傳數量。

    同時把 epochs 上限設為 cap_epochs（None 表示不限制）以加速煙霧測試。
    """
    n = 0

    def _patch(params: dict) -> None:
        nonlocal n
        if "device" in params:
            params["device"] = device
            n += 1
        if cap_epochs is not None and "epochs" in params:
            try:
                params["epochs"] = min(int(params["epochs"]), cap_epochs)
            except (TypeError, ValueError):
                pass

    for node in graph.get("nodes", []):
        data = node.get("data", {})
        _patch(data.get("params", {}))
        for sub in data.get("internalParams", {}).values():
            if isinstance(sub, dict):
                _patch(sub)
    return n


async def _run_one(path: Path, device: str, cap_epochs: int | None) -> tuple[bool, str]:
    graph = json.loads(path.read_text(encoding="utf-8"))
    n = patch_device(graph, device, cap_epochs)
    try:
        await execute_graph(graph["nodes"], graph["edges"])
        return True, f"patched {n} device param(s)"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Run example graphs on a chosen device.")
    parser.add_argument("device", nargs="?", default=None, help="cpu / cuda / mps (預設自動)")
    parser.add_argument("graphs", nargs="*", help="自訂 graph.json 路徑（預設用內建範例清單）")
    parser.add_argument("--full", action="store_true", help="不限制 epochs，完整執行")
    args = parser.parse_args()

    setup_logging(level="WARNING")
    registry.discover(settings.NODES_DIR, "app.nodes")
    registry.discover(settings.CUSTOM_NODES_DIR, "app.custom_nodes")
    preset_registry.discover(settings.PRESETS_DIR, registry)

    device = args.device or _best_device()
    cap_epochs = None if args.full else 1
    graph_paths = [Path(g) for g in args.graphs] or [_REPO_ROOT / p for p in _DEFAULT_EXAMPLES]

    print(f"available devices: {get_available_devices()}")
    print(f"target device    : {device}")
    print(f"epoch cap        : {'none (full)' if cap_epochs is None else cap_epochs}\n")

    failures = 0
    for path in graph_paths:
        if not path.exists():
            print(f"⚠️  [SKIP] {path} (not found)")
            continue
        ok, detail = await _run_one(path, device, cap_epochs)
        marker = "✅" if ok else "❌"
        status = "OK" if ok else "FAIL"
        label = path.parent.name if path.name == "graph.json" else path.name
        print(f"{marker} [{status}] {label}")
        first = detail.splitlines()[0] if detail else ""
        print(f"      {first}")
        if not ok:
            failures += 1
            for line in detail.splitlines()[1:]:
                print(f"      {line}")

    print(f"\n{len(graph_paths) - failures}/{len(graph_paths)} passed on {device}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
