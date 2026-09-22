---
sidebar_position: 7
title: CLI 圖形執行器
description: 使用 run_graph.py 直接從命令列執行已儲存的 graph.json — 不需要伺服器。
---

# CLI 圖形執行器

你可以直接從命令列執行任何圖，而不需要啟動伺服器。這對於批次執行、CI，或在無介面（headless）下重現一條管線都很方便。

如果你要透過 HTTP 呼叫*執行中*伺服器上*已儲存*的圖，傳入已宣告的輸入並取得已宣告的輸出，請見 **[把 graph 當成函式呼叫](./graph-as-a-function)**。若要把圖檔排入執行中伺服器的佇列，讓執行在關閉終端機後繼續，並出現在**執行任務**面板中，請使用 `cdui run`；請見 **[執行佇列](./run-queue#cdui-run)**。

`run_graph.py` 會匯入後端程式，所以要在後端的虛擬環境中執行（`backend/.venv`，由 `cdui install` 建立）：

```bash
cd backend
source .venv/bin/activate    # Windows：.venv\Scripts\activate
python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
```

執行器會透過 registry 探索所有節點、驗證 DAG、依拓撲順序執行它，並印出每個節點的輸出摘要。

## 選項 {/* #options */}

| 旗標 | 效果 |
|------|--------|
| `--validate-only` | 驗證圖（DAG、型別、連接埠、Start 節點）但不執行它。 |
| `--verbose`、`-v` | `DEBUG` 等級的 log，外加節點在執行期失敗時的完整 traceback。沒有任何 CLI 開關可以輸出檢視器的步驟追蹤。 |
| `--device` | `cpu` / `cuda` / `cuda:N` / `mps` / `auto`。省略時用圖檔的 [`settings.device`](/advanced/device-backends#the-graph-settings-object)，再沒有就是 `cpu`；`auto` 會用目前最好的加速器。 |
| `--seed N` | 用 `N` 為每個節點設定種子，讓執行可以重現。設了種子的執行會一次只跑一個節點 — 見 **[可重現的執行](./running-graphs#reproducible-runs-seed)**。 |
| `--deterministic` | 要求 PyTorch 使用決定性運算核心（`warn_only`，沒有決定性實作的運算會發出警告，而不會讓執行失敗）。 |

```bash
# 驗證一個架構但不執行它
python run_graph.py ../examples/Model_Architecture/ResNet-SkipConnection-CNN/graph.json --validate-only
```

## 離開碼 {/* #exit-codes */}

| 代碼 | 意義 |
| --- | --- |
| 0 | 圖已執行完畢，或在 `--validate-only` 下驗證通過 |
| 1 | 找不到檔案、圖驗證失敗，或節點在執行期失敗（`-v` 會加上完整 traceback） |
| 2 | 命令列有誤（參數解析失敗） |

## 外掛節點 {/* #plugin-nodes */}

執行器會載入 `CODEFYUI_USER_DATA_DIR` 下 plugin lockfile 所列的外掛包；未設定該變數時，則讀取平台資料目錄（`%LOCALAPPDATA%\codefyui`、`~/.local/share/codefyui` 或 `~/Library/Application Support/codefyui`）。`cdui` 指令（包括 `cdui plugin install` 與以 `cdui start` 啟動的伺服器）則把 lockfile 放在 `<install dir>/.codefyui_dev/`。若要執行使用這些外掛包節點的圖，請先設定 `CODEFYUI_USER_DATA_DIR=<install dir>/.codefyui_dev`；否則執行器不認得這些節點類型，驗證會失敗。

## 相對檔案路徑 {/* #relative-file-paths */}

讀取節點中的相對 `path` 由執行這張圖的行程解析。對執行器而言，工作目錄就是你啟動它時所在的目錄（上面指令中的 `backend/`）。

- **ImageReader** 會直接使用絕對路徑。相對路徑會先在圖片上傳區（預設為 `backend/data/images`）中尋找，找不到再相對於工作目錄解析。
- **CSVReader** 會先在資料檔上傳區（預設為 `backend/data/files`）中尋找單純的檔名。其他相對路徑則相對於工作目錄解析；設定了 `CODEFYUI_PROJECT_DIR` 時，改在該專案目錄內解析，而且不能超出該目錄（隨附的 `data/samples/iris.csv` 例外）。

## 圖從哪裡來 {/* #where-graphs-come-from */}

任何從 UI 匯出的圖（**[分頁與持久化 → 匯入／匯出](./tabs-persistence)**）都是格式相同的純 JSON 檔案，所以你可以視覺化地建構一條管線，然後從 CLI 執行它。`examples/` 底下隨附的範例已可直接執行 — 見 **[範例集](./examples-gallery)**。
