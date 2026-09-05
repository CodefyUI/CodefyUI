---
sidebar_position: 7
title: CLI 圖形執行器
description: 使用 run_graph.py 直接從命令列執行已儲存的 graph.json — 不需要伺服器。
---

# CLI 圖形執行器

你可以直接從命令列執行任何圖，而不需要啟動伺服器。這對於批次執行、CI，或在無介面（headless）下重現一條管線都很方便。

如果你要透過 HTTP 呼叫*執行中*伺服器上*已儲存*的圖，傳入已宣告的輸入並取得已宣告的輸出，請見 **[把 graph 當成函式呼叫](./graph-as-a-function)**。

```bash
cd backend
python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
```

執行器會透過 registry 探索所有節點、驗證 DAG、依拓撲順序執行它，並印出每個節點的輸出摘要。

## 選項

| 旗標 | 效果 |
|------|--------|
| `--validate-only` | 驗證圖（DAG、型別、連接埠、Start 節點）但不執行它。 |
| `--verbose`、`-v` | `DEBUG` 等級的 log，外加節點在執行期失敗時的完整 traceback。沒有任何 CLI 開關可以輸出檢視器的步驟追蹤。 |
| `--device` | 全域運算裝置：`cpu` / `cuda` / `mps`。 |
| `--seed N` | 用 `N` 為每個節點設定種子，讓執行可以重現。設了種子的執行會一次只跑一個節點 — 見 **[可重現的執行](./running-graphs#可重現的執行亂數種子)**。 |
| `--deterministic` | 要求 PyTorch 使用決定性運算核心（`warn_only`，沒有決定性實作的運算會發出警告，而不會讓執行失敗）。 |

```bash
# 驗證一個架構但不執行它
python run_graph.py ../examples/Model_Architecture/ResNet-SkipConnection-CNN/graph.json --validate-only
```

## 圖從哪裡來

任何從 UI 匯出的圖（**[分頁與持久化 → 匯入／匯出](./tabs-persistence)**）都是格式相同的純 JSON 檔案，所以你可以視覺化地建構一條管線，然後從 CLI 執行它。`examples/` 底下隨附的範例已可直接執行 — 見 **[範例集](./examples-gallery)**。
