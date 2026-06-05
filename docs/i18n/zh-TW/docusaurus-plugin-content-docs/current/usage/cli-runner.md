---
sidebar_position: 7
title: CLI 圖形執行器
description: 使用 run_graph.py 直接從命令列執行已儲存的 graph.json — 不需要伺服器。
---

# CLI 圖形執行器

你可以直接從命令列執行任何圖，而不需要啟動伺服器。這對於批次執行、CI，或在無介面（headless）下重現一條管線都很方便。

```bash
cd backend
python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
```

執行器會透過 registry 探索所有節點、驗證 DAG、依拓撲順序執行它，並印出每個節點的輸出摘要。

## 選項

| 旗標 | 效果 |
|------|--------|
| `--validate-only` | 驗證圖（DAG、型別、連接埠、Start 節點）但不執行它。 |
| `--verbose` | 發出中間步驟追蹤，與 Inspector 的 **Steps** 分頁所顯示的資料相同。 |

```bash
# 驗證一個架構但不執行它
python run_graph.py ../examples/Model_Architecture/ResNet-SkipConnection-CNN/graph.json --validate-only
```

## 圖從哪裡來

任何從 UI 匯出的圖（**[分頁與持久化 → 匯入／匯出](./tabs-persistence)**）都是格式相同的純 JSON 檔案，所以你可以視覺化地建構一條管線，然後從 CLI 執行它。`examples/` 底下隨附的範例已可直接執行 — 見 **[範例集](./examples-gallery)**。
