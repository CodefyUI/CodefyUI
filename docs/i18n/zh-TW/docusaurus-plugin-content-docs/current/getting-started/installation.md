---
sidebar_position: 1
title: 安裝
description: 透過一行指令安裝 CodefyUI —— 一般使用者只需要 git、uv 與 Python，不需要 Node.js。
---

# 安裝

快速安裝程式會自動設定好 `git`、`uv` 與 Python（透過 uv）。前端 bundle 會從 GitHub 最新 release 直接下載預編好的版本，後端則會 checkout 到同一個 release tag，讓兩者保持同步 —— **一般使用者不需要 Node.js 或 pnpm**。

:::tip 我該用哪種安裝方式？
- **快速安裝**（本頁）—— 你只想*執行* CodefyUI。
- **[開發者安裝](./dev-install)** —— 你想編輯程式碼或貢獻（手動設定 `uv` + pnpm，並支援熱重載）。
:::

## 快速安裝

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/CodefyUI/CodefyUI/main/install.sh | bash
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/CodefyUI/CodefyUI/main/install.ps1 | iex"
```

預設安裝到 `~/CodefyUI`（macOS/Linux）或 `%USERPROFILE%\CodefyUI`（Windows）。可用環境變數 `CODEFYUI_DIR` 覆寫。

在 Windows 上，`install.ps1` 會透過 [winget](https://learn.microsoft.com/windows/package-manager/) 安裝缺少的 `git`。`winget` 內建於 Windows 11 與較新的 Windows 10（透過「App Installer」套件）。若 `winget` 不可用，或其套件來源無法連線（企業網路 TLS 攔截會使 `msstore` 來源回報 `0x8a15005e`），安裝程式會改以 [PortableGit](https://git-scm.com/download/win) 解壓至 `%LOCALAPPDATA%\CodefyUI\PortableGit` —— 不需系統管理員權限。

安裝程式會把 `cdui` 啟動器放到 `~/.local/bin/cdui`（Windows 為 `%USERPROFILE%\.local\bin\cdui.cmd`）。**請重新開啟你的 terminal**，然後在任何目錄執行：

```bash
cdui start
```

開啟 [http://localhost:8000](http://localhost:8000)。單一 uvicorn 行程會同時提供 API 與預編好的 React 前端。`cdui start` 預設在**背景**執行。關閉終端機後，伺服器仍會繼續執行；使用 `cdui status` 與 `cdui stop` 管理。加上 `--foreground`（`-f`）可改為前景執行，並以 `Ctrl+C` 停止。

:::note
本快速開始假設使用預設的 PyTorch 版本，它適用於所有平台（CPU / Apple Silicon MPS）。若需特定的 NVIDIA CUDA 版本、AMD ROCm，或想驗證 GPU 偵測，請參考 **[GPU 與裝置設定](./gpu-device)**。

安裝後切換版本也不必使用終端機。伺服器若由 `cdui start` 啟動，套件中心（工具列 > 設定 > 選用套件）裡的 **GPU 版 PyTorch** 卡片可以安裝對應的 wheel 並重新啟動伺服器。卡片下方也會顯示等效的 `cdui install --gpu <choice>` 指令，供手動執行。詳見[讓伺服器重新啟動的安裝](/usage/optional-packs#讓伺服器重新啟動的安裝)。
:::

## 安裝旗標與環境變數

`install.sh` / `install.ps1` 只會讀取下列環境變數，並一律執行 `cdui install --yes`；它們不接受旗標，也不會顯示提示。安裝完成後，若要使用互動式選單或傳入旗標，請直接執行 `cdui install`。只有在終端機中執行，且旗標與環境變數都未決定選項時，才會顯示互動式選單；透過 pipe 或 CI 執行則採用安全預設值。

| 旗標 | 環境變數 | 值 | 用途 |
|------|----------|----|------|
| `--gpu <choice>` | `CODEFYUI_GPU` | `auto` / `cu118` / `cu121` / `cu124` / `cu126` / `cu128` / `rocm6.1` / `rocm6.2` / `cpu` / `mps` / `skip` | 選擇 PyTorch wheel index。`auto` 透過 `nvidia-smi`／`rocm-smi`／Apple Silicon 自動偵測。`skip` 完全不裝 torch（進階）。 |
| `--dev` / `--no-dev` | `CODEFYUI_DEV` | `1` / `0` | 是否安裝 `[dev]` extra（pytest、httpx、httpx-ws）。`cdui test` 需要。一般使用者預設關閉，貢獻者開啟。 |
| `--yes` | — | — | 全部用預設值，不互動（CI／headless）。 |
| `--lang <code>` | `CODEFYUI_LANG` | `en` / `zh`（環境變數也接受 `zh-TW`、`zh-HK`、`zh-CN`、`english`、`chinese`） | 旗標只對 `cdui install` 與 `cdui update` 有效；環境變數會設定每個 `cdui` 指令的輸出語言。 |
| — | `CODEFYUI_DIR` | path | 安裝目錄（預設 `~/CodefyUI`）。 |
| — | `CODEFYUI_RELEASE_TAG` | tag | 把前端 bundle **與**後端 checkout 鎖定到同一個 release（預設 `latest`）。 |
| — | `CODEFYUI_FORCE_BUILD` | `1` | 跳過下載 prebuilt dist，改在本地用 pnpm build（追蹤 `main`）。 |
| — | `CODEFYUI_UV_INSTALL_TIMEOUT` | seconds | `PATH` 中找不到 `uv` 時，允許自動下載 `uv` 的時間（預設 `180` 秒；`0` = 不設上限）。 |

## 正式模式與開發者模式

- `cdui start` —— 單一 uvicorn 跑 `:8000` 提供預編前端。**不需要 Node。** 這是一般使用者的預設模式。
- `cdui dev` —— Vite dev server 跑 `:5173`（HMR）+ uvicorn 跑 `:8000`。**需要 Node 24+ 與 pnpm。** 編輯前端程式碼時使用 —— 請參考[開發者安裝](./dev-install)。
- `cdui build` —— 在本地重建 `frontend/dist`（也需要 Node + pnpm）。

完整的啟動器指令清單請見 **[CLI 指令](./cli-commands)**。

## 裝在伺服器上給一個團隊用

上面的步驟裝出來的是一台只在 `127.0.0.1` 上、給自己用的環境。如果有好幾個人要共用一台機器，請先讀 **[放在反向代理後面](/usage/deployment)**：CodefyUI 沒有使用者帳號，所以身分驗證和 TLS 都來自前面那台代理，而代理的主機名稱必須加進 `CODEFYUI_EXTRA_ALLOWED_HOSTS`，否則每一個請求 —— 包含網頁本身 —— 都會被以 `421` 拒絕，瀏覽器上就是一片空白。

## 驗證是否正常運作

```bash
curl http://127.0.0.1:8000/api/health
```

這應該會回傳類似 `{"status":"ok","nodes_loaded":152,"presets_loaded":3}` 的內容（`nodes_loaded` 數量會隨每個版本增加，確認其為非零即可）。

接著開啟前端，載入 **Train CNN on MNIST** 範例並點擊 **執行**。你應該會在下方面板看到訓練進度出現。

## 選用套件包

上面的安裝刻意保持精簡，因此不包含某些課程需要的大型附加內容，例如 `sentence-transformers`、各個嵌入模型（每個 90 MB 到 470 MB）與 69 MB 的 GloVe 詞向量表。你可以在**套件中心**（工具列 > 設定 > 選用套件）或用 `cdui packs install <id>` 安裝需要的項目。其他行為不變：執行圖形時不會自行下載套件包內容；缺少套件包的節點會停止並指出所需套件包，不會在執行途中下載數百 MB 的內容。

型錄內容、檔案會放在哪裡，以及該挑哪一個嵌入模型，請見 **[選用套件包](/usage/optional-packs)**。

## 更新

```bash
cdui update
```

更新到最新 release（prebuilt 路徑），或拉取 `main`（從原始碼建置時）並重新同步前端。

和 `cdui install` 不同，這個指令不會詢問任何問題。它會直接從已安裝的 wheel 讀出 PyTorch 變體，沿用 venv 中既有的變體與 dev 工具設定，因此你刻意選的 torch 版本不會被動到，沒有變動時也不會重新下載。真的要換的時候，`--gpu` / `--dev` 旗標與 `CODEFYUI_GPU` / `CODEFYUI_DEV` 環境變數依然可以覆蓋。
