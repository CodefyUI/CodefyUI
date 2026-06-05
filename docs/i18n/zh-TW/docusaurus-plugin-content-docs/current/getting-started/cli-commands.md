---
sidebar_position: 4
title: CLI 指令
description: cdui 啟動器指令 —— install、start、status、dev、build、外掛管理等等。
---

# CLI 指令

`cdui` 是一支由安裝程式放到 `~/.local/bin/cdui` 的輕量啟動器（Windows 上為 `cdui.cmd`）。若你還沒重新開啟 terminal，可改用絕對路徑 `~/CodefyUI/cdui start`，或使用 `python scripts/dev.py <cmd>` —— `dev.py` 會自動切換到 venv 的 Python 重新執行。

## 核心指令

| 指令 | 說明 |
|------|------|
| `cdui install` | 安裝後端依賴；下載預編好的前端（若有 `pnpm` 則改在本地 build）。 |
| `cdui update` | 更新到最新 release（prebuilt 路徑），或拉取 `main`（原始碼建置）並重新同步前端。 |
| `cdui start` | 正式模式 —— 單一 uvicorn 跑 `:8000`，在背景執行（不需要 Node）。`--foreground`／`-f` 改為前景執行。 |
| `cdui status` | btop／k9s 風格的儀表板：CPU、記憶體、磁碟、GPU、前幾名程序，外加伺服器的 PID 與健康狀態。即時刷新（每 2 秒；`Ctrl+C` 離開）。傳入一個數字可設定間隔（`cdui status 1`），或用 `--once` 只顯示單一畫面。 |
| `cdui dev` | 開發者模式 —— 後端 `:8000` + Vite HMR `:5173`（需要 Node + pnpm）。 |
| `cdui build` | 在本地建置前端 bundle（需要 Node + pnpm）。 |
| `cdui stop` | 停止所有服務（包含背景伺服器）。 |
| `cdui test` | 執行後端測試。 |
| `cdui clean` | 移除虛擬環境、`node_modules` 與 `frontend/dist`。 |
| `cdui uninstall` | clean + 移除 PATH 上的啟動器。 |

## 外掛指令

| 指令 | 說明 |
|------|------|
| `cdui plugin install <name\|url>` | 安裝一個外掛包（型錄名稱如 `foundations`、`owner/repo[@ref]`，或完整的 GitHub URL）。 |
| `cdui plugin list` | 列出已安裝的外掛包。 |
| `cdui plugin info <id>` | 顯示某個外掛包的 manifest、涵蓋的課程與節點名稱。 |
| `cdui plugin search <query>` | 查詢外掛型錄。 |
| `cdui plugin uninstall <id>` | 移除一個已安裝的外掛包。 |

完整的外掛工作流程請見 **[外掛](/advanced/plugins)**。

## 背景與前景

`cdui start` 預設在**背景**執行 —— 關閉 terminal 而伺服器會繼續運作。用以下指令管理它：

```bash
cdui status     # 即時儀表板 + 健康狀態
cdui stop       # 停止背景伺服器
cdui start -f   # 改為前景執行（Ctrl+C 停止）
```

## 不啟動伺服器執行圖

你不需要網頁 UI 就能執行一張圖 —— 請見 **[CLI 圖形執行器](/usage/cli-runner)**：

```bash
cd backend
python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
```
