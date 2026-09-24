---
sidebar_position: 7.7
title: 版本控管你的 graphs
description: 把圖表 JSON 放進自己的 git 服務儲存庫，用 run_graph.py 在 CI 驗證每一張圖，再從有版本紀錄的來源發佈。
---

# 版本控管你的 graphs

:::tip 現在已經有正式的專案目錄了
這一頁講的是比較舊的「單層目錄」做法，它的形狀**沒辦法**直接對應到邏輯與版面分開的新格式。新的服務請改用[專案目錄](./project-directories)；已經存在的 graphs 目錄可以一行指令搬過去：`cdui project init my-service --adopt /path/to/my-graphs`。
:::

:::tip git 的步驟可以交給編輯器做
圖表放進專案目錄之後，編輯器的**版本控制**分頁就能代你完成[設定服務儲存庫](#set-up-a-service-repo)裡的那些 git 動作：暫存、提交、開分支、推送、處理衝突，以及查看歷史與差異（一次性的登入憑證設定仍然要在終端機裡做）。見[版本控制](./source-control)。
:::

CodefyUI 把每一張圖存成一個純 JSON 檔，所以圖表天生就適合放進 git：你建的流程從此有了歷史、審查與回溯。這一頁是一份做法：把圖表放進它們自己的 git 儲存庫、在 CI 裡驗證，再從有版本紀錄的來源發佈。

先講一個前提：預設的儲存位置**不在**版本控制裡。存下來的圖會落在 `backend/data/graphs/`，而這個路徑被這個儲存庫自己的 `.gitignore` 排除掉了（`backend/data/graphs/*.json`，只留下 `.gitkeep`）。要把你的成果納入版本控制，就要把 CodefyUI 的圖表目錄指到你自己的儲存庫——這一頁接下來講的就是怎麼做。

## 什麼該進版本控制，什麼絕對不要 {/* #what-to-version-and-what-never-to */}

**這些請納入版本控制：**

- 你的圖表 JSON 檔——每張存下來的圖一個 `<name>.json`。
- 你自己擁有、而且希望能重現的小型資料或模型檔，或是一支去抓大檔案的腳本。

**這些絕對不要提交。** 它們是機器本機的狀態、機密，或是衍生資料，而且沒有一項存在圖表 JSON 裡——多數預設就放在你的 graphs 目錄之外，請讓它們留在那裡：

- SQLite 資料庫 `codefyui.db`（預設 `backend/data/codefyui.db`，可用 `CODEFYUI_DB_PATH` 覆寫）。它裡面有已發佈的應用、版本快照、API 金鑰與執行紀錄。
- 執行紀錄——只存在那個資料庫裡，不會寫成你會提交的檔案。
- 已發佈應用的 API 金鑰（`cdui_...` 開頭的 bearer token）——同樣只在資料庫裡，而且是以 sha256 雜湊保存。
- `.env` 檔以及任何本機機密。
- 編輯器的工作階段 token 檔：以 `cdui start` 或 `cdui dev` 啟動的伺服器放在 `<install dir>/.codefyui_dev/session.token`（預設安裝目錄為 `~/CodefyUI`）；設定了 `CODEFYUI_USER_DATA_DIR` 時放在 `<dir>/session.token`。只有手動啟動的 `uvicorn app.main:app` 會使用平台的使用者資料目錄（Windows 是 `%LOCALAPPDATA%\codefyui\session.token`）。它寫在你的 graphs 目錄之外，而且每次伺服器重新啟動都會換一份，所以絕對不該被複製進儲存庫。
- LLM 供應商的 API 金鑰——請放在環境變數裡；見下面的[機密](#secrets-keep-keys-out-of-your-graphs)。

## 設定服務儲存庫 {/* #set-up-a-service-repo */}

建一個放圖表的目錄，初始化 git，再用 `CODEFYUI_GRAPHS_DIR` 把 CodefyUI 指過去。接著啟動伺服器，從介面存圖，然後像對待任何原始碼一樣提交它們。

```bash
mkdir my-graphs && cd my-graphs
git init
```

把 CodefyUI 的圖表目錄指到它。這個變數只在伺服器啟動時讀一次，所以要設在你執行 `cdui start` 的同一個 shell（或同一個工作階段）裡。

PowerShell：

```powershell
$env:CODEFYUI_GRAPHS_DIR = "C:\path\to\my-graphs"
cdui start
```

cmd.exe：

```bat
set CODEFYUI_GRAPHS_DIR=C:\path\to\my-graphs
cdui start
```

bash：

```bash
export CODEFYUI_GRAPHS_DIR=/path/to/my-graphs
cdui start
```

現在你從介面存下的每一張圖，都會以 `<name>.json` 寫進 `my-graphs/`。存一張圖，然後提交它：

```bash
git add .
git commit -m "Add my first classifier graph"
```

**每次啟動伺服器都要設 `CODEFYUI_GRAPHS_DIR`**——它不會被保存在任何地方。開一個沒設它的新終端機執行 `cdui start`，就會退回預設的 `backend/data/graphs/`，你的服務儲存庫看起來會是空的。把它寫進 shell 設定檔設定一次（PowerShell 的 `$PROFILE`，或 `~/.bashrc`／`~/.zshrc`），或是把這兩行包成一支放在儲存庫旁邊的小啟動腳本：

```bash
# start.sh
export CODEFYUI_GRAPHS_DIR="$(cd "$(dirname "$0")" && pwd)"
cdui start
```

## 給服務儲存庫的 .gitignore {/* #a-gitignore-for-your-service-repo */}

把這份放進 graphs 儲存庫的根目錄，權重、資料庫與機密就不會偷偷混進來：

```
*.pt
*.pth
*.safetensors
*.onnx
*.ckpt
*.db
.env
__pycache__/
```

大型資料集請提交一支小小的下載腳本（或一個網址加上檢查碼），而不是資料本身——讓儲存庫只留下圖表，以及去把其他東西抓回來的程式碼。

## 機密：把金鑰留在圖表之外 {/* #secrets-keep-keys-out-of-your-graphs */}

`LLMChat` 有兩個 API 金鑰欄位：`openai_api_key` 與 `anthropic_api_key`。它們是機密參數：打進去的值只在這次編輯器工作階段中使用，而且會從存下來的圖、匯出檔、已發佈的版本與執行紀錄中清空（完整清單見[共用的伺服器](./shared-instances#what-is-per-graph-instead)）。所以你提交的圖不會帶著金鑰，但重新載入之後或換一台機器，這個欄位也會是空的。

請改用環境變數提供金鑰。節點會依下列順序取用第一個非空值：

1. 節點的 `openai_api_key` 欄位（只在這次工作階段有效）
2. `CODEFYUI_OPENAI_API_KEY`（環境變數）
3. `OPENAI_API_KEY`（環境變數）

Anthropic 的做法一樣，順序是 `anthropic_api_key`、`CODEFYUI_ANTHROPIC_API_KEY`，然後 `ANTHROPIC_API_KEY`。請在 `cdui start` 之前設好環境變數。

金鑰仍然可能進到提交裡：貼進不是機密參數的欄位（例如提示詞），或是手動寫進 JSON。發佈前置檢查會拒絕仍帶有機密值的圖表檔。沒有載入的外掛所提供的節點不會被辨識出來，所以存檔時它的機密欄位不會被清空。

## 在 CI 裡驗證每一張圖 {/* #validate-every-graph-in-ci */}

`run_graph.py` 可以不執行圖就檢查它：它會探索所有節點，驗證 DAG、型別、連接埠與 Start 的接線，只要有問題就以非零狀態結束。這正是 CI 要的——壞掉的圖直接讓建置失敗。執行器本身見 [CLI 圖表執行器](./cli-runner)。

```bash
# 在本機，從 CodefyUI 的 checkout 執行
cd backend
python run_graph.py /path/to/my-graphs/classifier.json --validate-only
```

執行器住在 CodefyUI 的後端裡，目前 PyPI 上沒有獨立套件，所以在*另一個* graphs 儲存庫裡取得它最可靠的方式，是把 CodefyUI 以固定的發佈標籤 checkout 到你的儲存庫旁邊，再用 uv 安裝它的後端（就像 CodefyUI 自己的 CI 安裝自己的做法）。安裝後端會把整個執行環境拉進來，PyTorch 也包含在內，所以這個工作並不輕量——請把 venv 快取起來，否則冷啟動要等上幾分鐘。這是今天誠實的現況；一個輕量的驗證指令是很自然的未來方向。

下面這份工作假設這個儲存庫用的是專案目錄的版面（一份 `codefyui.project.toml` 說明檔加上 `graphs/`／`layout/`），而不是一個只有 `*.json` 的單層目錄，因為 `cdui project validate` 需要那份說明檔——搬移方式見[專案目錄](./project-directories)，用 `cdui project init <dir> --adopt <this-repo>`。還想留在單層版面？那就照上面的做法，繼續用 `run_graph.py` 一個檔一個檔驗證。

```yaml
name: validate-graphs
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Check out this graphs repo
        uses: actions/checkout@v4

      - name: Check out CodefyUI (pinned)
        uses: actions/checkout@v4
        with:
          repository: CodefyUI/CodefyUI
          ref: "2.8.4" # 固定在 1.4.0 以後的某個發佈標籤（cdui project 從 1.4.0 開始提供）
          path: CodefyUI

      - name: Install uv
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          echo "$HOME/.local/bin" >> "$GITHUB_PATH"

      - name: Install the CodefyUI backend
        working-directory: CodefyUI/backend
        run: |
          uv venv
          uv pip install -e .

      - name: Restore plugin pins, then validate the project
        run: |
          ./CodefyUI/cdui project restore .    # 或：CodefyUI/backend/.venv/bin/python CodefyUI/scripts/dev.py project restore .
          ./CodefyUI/cdui project validate .   # 檢查專案裡的每一張圖
```

這個工作是用 checkout 裡的路徑呼叫 `cdui` 啟動器：`uv pip install -e .` 不會安裝 `cdui` 指令，所以那裡的 `PATH` 上沒有 `cdui`。`cdui project validate .` 會檢查專案裡的每一張圖（檢查項目列在[專案目錄](./project-directories#4-validate-the-ci-gate)）。它不會把 `layout/*.layout.json` 交給驗證器，所以沒有 `*.json` 萬用字元可以寫錯。請先執行 `cdui project restore`，讓外掛提供的節點在驗證前就已安裝（CI 的順序：先 restore，再 validate）。

## 從有版本紀錄的圖發佈 {/* #publishing-from-a-versioned-graph */}

版本控制不會改變你怎麼發佈：存好圖，然後[發佈](./publish)它。因為發佈時會把已存檔圖表的實際位元組快照進資料庫，所以你送出去的那一版，不會受到之後任何編輯或提交影響。

想把某個已發佈版本追回它的來源，可以把該圖的 git commit 雜湊放進發佈的 `note` 欄位——它是自由文字，會和版本一起存下來，也會在版本清單裡顯示：

```json
{"graph": "classifier", "create": true, "note": "git 1a2b3c4"}
```

這樣你就有一條從執行中的應用版本回到來源 commit 的線索——或者乾脆跳過這個手動的 `note` 慣例：[專案目錄](./project-directories)本身就帶有正式的發佈來源紀錄，`cdui project publish` 會在每一版自動記下當下的 `git_commit`／`git_dirty`。

## 目前已知的粗糙處 {/* #known-rough-edges */}

這一頁的單層 `<name>.json` 做法有兩個 diff 雜訊的來源：

- **節點位置會製造 diff 雜訊。** 拖動節點會改變存下來的座標，所以只是整理版面也會產生 JSON 差異，即使流程完全沒變。
- **複製貼上會重新產生節點 id。** 複製節點會配到新的 id，所以一個很小的邏輯改動可能看起來像一大片差異。

[專案目錄](./project-directories)能減少這兩種雜訊。節點位置和邏輯檔分開，存在 `layout/<name>.layout.json`，而 **隱藏版面檔案** 可以把這些檔案從版本控制分頁的「變更」清單中拿掉。分頁的圖表摘要會計算移動過的位置，並忽略重新產生的連線 id；不過 id 改變了的節點，仍然會算成移除一個、新增一個。見[圖表到底改了什麼](./source-control#what-changed-in-the-graph)。
