---
sidebar_position: 4
title: CLI 指令
description: cdui 啟動器的 install、start、status、dev、build、外掛管理等指令。
---

# CLI 指令

`cdui` 是安裝程式放在 `~/.local/bin/cdui` 的輕量啟動器，在 Windows 上為 `cdui.cmd`。如果尚未重新開啟 terminal，請使用絕對路徑 `~/CodefyUI/cdui start`，或執行 `python scripts/dev.py <cmd>`。`dev.py` 會自動改用 venv 的 Python 重新執行。

## 核心指令

| 指令 | 說明 |
|------|------|
| `cdui install` | 安裝後端依賴；下載預編好的前端（若有 `pnpm` 則改在本地 build）。 |
| `cdui update` | 預先建置的安裝會更新至最新 release；原始碼安裝會拉取 `main` 並重新建置前端。此指令不會顯示確認提示。除非以 `--gpu` 或 `--dev` 覆寫，否則會沿用 venv 中的 PyTorch 變體與開發工具。伺服器執行中時，更新會移除正在提供的 `frontend/dist`，所以此指令會拒絕執行；請先執行 `cdui stop`。重新啟動安裝仍在收尾時，此指令也會拒絕並以離開碼 `1` 結束。請參閱[套件包指令](#套件包指令)。 |
| `cdui start` | 在正式模式下以背景行程啟動單一 uvicorn，預設使用 `:8000`。不需要 Node。`--foreground`／`-f` 改為前景執行。 |
| `cdui run <graph.json>` | 將已儲存的 graph 提交至執行中伺服器依裝置區分的 FIFO 佇列。執行由伺服器管理，因此關閉此終端機後仍會繼續。旗標：`--name`、`--device`（`cpu` / `auto` / `cuda` / `cuda:N` / `mps`）、`--seed`、`--deterministic`、`--record-outputs`、`--wait`（預設）或 `--detach`、`--timeout <s>`、`--host` / `--port`。離開碼：`0` 成功（或 `--detach` 已提交）、`1` 失敗、取消或無法提交、`2` 命令列有誤、`130` 表示 `Ctrl+C`；這只會停止等待結果，不會停止該次執行。詳見 **[執行佇列](/usage/run-queue#cdui-run)**。 |
| `cdui status` | btop／k9s 風格的儀表板：CPU、記憶體、磁碟、GPU、前幾名行程，以及伺服器的 PID 與健康狀態。即時重新整理（每 2 秒；`Ctrl+C` 離開）。傳入數字可設定間隔（`cdui status 1`），或用 `--once` 只顯示單一畫面；`-w` / `--watch [secs]` 即使 stdout 不是 terminal，也會強制進入即時迴圈（間隔下限 0.5 秒）。單一畫面模式（`--once` 或將輸出導向其他位置）會在伺服器未執行時以 `1` 結束，讓腳本可依離開碼判斷狀態。它也會回報會重新啟動伺服器的套件包安裝：認領紀錄存在時顯示「重啟安裝」一行；輔助程式仍在執行時顯示「收尾中」，已不在執行時顯示「已中斷」。工作完成後的一小時內則會顯示「上次重啟安裝」一行。 |
| `cdui dev` | 以開發模式啟動 `:8000` 的後端與 `:5173` 的 Vite HMR。需要 Node 與 pnpm。和 `cdui start` 一樣接受 `--project <dir>`。重新啟動安裝仍在收尾時，此指令會拒絕並以離開碼 `1` 結束。 |
| `cdui build` | 在本地建置前端 bundle（需要 Node + pnpm）。 |
| `cdui stop` | 停止**此安裝**的服務，包括 pidfile 記錄的背景伺服器，以及從此目錄啟動的殘留行程，例如前景 `cdui start`、`cdui dev` 的 Vite 與遺留 worker。`--all` 會停止整台機器上的所有 CodefyUI 與 Vite 行程，包括其他使用者的伺服器與無關的 Vite dev server。請勿在共用主機使用 `--all`。 |
| `cdui test` | 執行整個專案的測試：後端（`pytest`）與前端（`vitest`）。沒有 pnpm 時，前端測試會標示為 `SKIPPED`，不會使指令失敗，因為 release 安裝不含 Node。兩組測試都會執行完畢；任一組失敗時，指令以離開碼 `1` 結束。`--backend` / `--frontend` 可限定其中一組；其他參數會被拒絕（離開碼 `2`），不會直接忽略。若要篩選個別測試，請直接使用 `pytest` 或 `pnpm test`。 |
| `cdui clean` | 移除虛擬環境、`node_modules` 與 `frontend/dist`。 |
| `cdui uninstall` | clean + 移除 PATH 上的啟動器。 |
| `cdui --version` | 印出 `CodefyUI <version>`（也可用 `-V` 與 `cdui version`）。此指令會在其他作業前回傳版本，不需要 `uv` 或 venv，因此安裝未完成時仍可使用。 |

## 外掛指令

應用程式內的**外掛中心**可從側邊欄的**自訂與外掛**分頁或**設定 → 外掛**開啟，並與這些指令使用相同的安裝函式。請參閱 **[外掛中心](/advanced/plugins#plugin-center)**。

| 指令 | 說明 |
|------|------|
| `cdui plugin install <name\|url>` | 安裝一個外掛包——來源可以是型錄名稱、`owner/repo[@ref]` 或完整的 GitHub URL。型錄名稱同時涵蓋內建外掛包與各自存放在儲存庫中的官方外掛，因此 `cdui plugin install graph-copilot` 會抓取型錄指定的儲存庫。下載前會先讀取並顯示 manifest，包括外掛包用途、要新增的 Python 套件與宣告的能力，再要求確認；拒絕時不會下載任何內容。`--force` 會覆蓋已安裝的版本，`-y` 只跳過 `Proceed?` 確認，`--accept-capabilities` 會直接授予 manifest 宣告的能力，`--trust-author` 則接受要求匯入白名單以外模組的外掛包。 |
| `cdui plugin sync` | 安裝目前環境中尚未決定是否使用的所有**內建**外掛包，適合在更新新增內建外掛包後執行。指令會要求確認一次；`--yes` 可略過確認，且沒有終端機時為必要選項。`--dry-run` 只列出清單；`--prune` 也會移除已不再隨版本提供的內建外掛 lockfile 項目。先前手動移除的外掛包不會重新安裝。 |
| `cdui plugin update [<id>]` | 依記錄的 ref 重新讀取外掛儲存庫，並在 commit 變更時重新安裝；未指定 id 時會更新所有已安裝的第三方外掛包。外掛包會保留安裝來源的型錄項目，因此官方外掛更新後仍標示為官方，先前停用的外掛也維持停用。若儲存庫的 manifest 改為宣告**另一個**外掛 id，更新會被拒絕（離開碼 `1`），不會以新名稱安裝或覆蓋現有外掛；請將重新命名的儲存庫視為新外掛安裝。內建與連結外掛包會略過：內建外掛包使用 `cdui update` 更新，連結目錄則直接反映作者磁碟上的內容。 |
| `cdui plugin list` | 列出已安裝的外掛包，以及尚待決定是否使用的內建外掛包。 |
| `cdui plugin info <id \| catalog-name \| owner/repo[@ref]>` | 顯示外掛包的 manifest、涵蓋課程與節點名稱。若尚未安裝，只會讀取解析後 commit 中的 manifest，不會下載任何內容。 |
| `cdui plugin search [query]` | 查詢外掛型錄。未提供 query 時，會列出完整型錄、標示已安裝項目，並為 GitHub 上的項目加上標籤（官方項目為 `[github, official]`）。 |
| `cdui plugin uninstall <id>` | 移除已安裝的外掛包。若為內建外掛包，系統會記錄此決定，因此 `cdui plugin sync` 不會重新安裝；可用 `cdui plugin install <id>` 恢復。所有類型的外掛包都會保留其 Python 套件。執行中的伺服器已載入這些套件，直接移除可能使伺服器處於不完整狀態。若要釋放空間，請先停止伺服器，再手動移除套件。 |
| `cdui plugin enable <id>` / `cdui plugin disable <id>` | 在不修改檔案的情況下啟用或停用已安裝外掛：切換 lockfile 中的 `enabled`，並熱重新載入執行中的伺服器。外掛未安裝時以 `1` 結束；外掛已處於要求狀態時不執行變更（離開碼 `0`）。 |
| `cdui plugin link <path>` | 登記含有 `cdui.plugin.toml` 的本機外掛目錄，直接從原始位置載入，不複製檔案。`--force` 會覆蓋 id 相同的現有項目。 |
| `cdui plugin unlink <id>` | 移除連結外掛的 lockfile 項目，不會修改原始檔案。 |
| `cdui plugin reload` | 要求執行中的伺服器熱重新載入外掛與節點。這是手動執行 `link`、`enable` 與 `disable` 所觸發的相同行為。 |
| `cdui plugin dev <path>` | 先執行 `link` 再監看。目錄中的 manifest、nodes、presets 或 frontend 每次變更時都會觸發 reload。`--interval <s>` 設定檢查間隔（預設 `1`，下限 `0.2`）；`--once` 會連結、重新載入一次後結束。只要連結的外掛處於啟用狀態，編輯器就會自動重新載入它的 frontend bundle，不需重新整理瀏覽器。 |
| `cdui plugin new <id>` | 使用內建範本建立外掛目錄，內容包括 manifest、範例節點與測試。`--ui` 會加入使用 SDK 的 React frontend。`--name` 設定顯示名稱；預設會從 id 推導。`--dir` 設定上層目錄；預設為目前目錄。`--force` 允許寫入非空目錄。 |

腳本可使用下列離開碼：`0` 表示完成，包括在 `Proceed?` 提示選擇不要；`1` 表示安裝失敗，或能力／模組要求遭拒；`2` 表示執行前即遭拒，例如來源無法解析或未提供來源；`3` 表示伺服器執行中，無法安裝此外掛的 Python 套件，並會印出替代指令；`130` 表示以 `Ctrl+C` 中斷。

完整的外掛工作流程請見 **[外掛](/advanced/plugins)**。

## 套件包指令

選用套件包是預設安裝未包含的大型附加內容，例如 `sentence-transformers`、嵌入模型、GloVe 詞向量表與加速版 PyTorch。應用程式內的**套件中心**可顯示進度並安裝套件包；以下指令可從終端機執行相同作業。如果套件包必須替換執行中伺服器已載入的內容，則只能使用終端機指令安裝。

| 指令 | 說明 |
|------|------|
| `cdui packs list` | 列出型錄中的所有套件包，包括內容、下載大小及安裝狀態。 |
| `cdui packs status` | 跟 `list` 一樣，另外加上這個 venv 裡的 PyTorch 版本，以及接下來該執行什麼。 |
| `cdui packs install <id>` | 安裝一個套件包。`--items a,b` 只下載指定的模型（預設會補齊這個套件包缺的全部）；`--yes`／`-y` 可跳過下載大小的確認，沒有終端機可確認時必須加。只接受型錄裡的 id —— 沒有任何方式可以傳入套件安裝字串、index 網址或 repo id。 |
| `cdui packs remove <id> <item-id>` | 刪除一個已下載的模型並清掉紀錄。套件包的 Python 套件不會一併移除；能移除它們的 `uv pip uninstall` 指令會印出來，請在伺服器停掉後自行執行。 |

腳本可使用下列離開碼：`0` 表示完成；`1` 表示安裝失敗或在提示中拒絕安裝；`2` 表示執行前即遭拒，包括套件包或 `--items` id 不存在、相依未滿足、套件包只能透過重新啟動安裝（`gpu-torch` 會印出改用的 `cdui install --gpu` 指令），或沒有終端機可進行確認；`3` 表示伺服器執行中，無法執行作業，並會印出替代指令；`130` 表示以 `Ctrl+C` 中斷。

**讓伺服器重新啟動的安裝。** 由 `cdui start` 啟動的伺服器，可透過自動重新啟動安裝 GPU PyTorch 套件包，或處理線上安裝遇到相依衝突的套件包。伺服器會先記錄安裝內容，以分離模式啟動 `cdui packs-run-pending`，再自行關閉。輔助程式會等待原行程結束、執行安裝、記錄結果，最後用原本的 `cdui start` 參數重新啟動伺服器。`packs-run-pending` 是**內部指令**，刻意不列在說明文字中。它收到的檔案會指定要等待的行程；若手動對執行中的伺服器執行，會等待兩分鐘後停止該伺服器。若自動重新啟動在某台機器上無法正常完成，可設定 `CODEFYUI_ENABLE_RESTART_INSTALL=0` 關閉此功能。重新啟動安裝仍在「收尾中」時——記錄的輔助程式仍在執行，或認領單建立未滿六十秒且輔助程式尚未寫入 pid——`cdui start` 不會在正在修改的 venv 中啟動第二個伺服器，而會提示查看 `cdui status` 後返回。輔助程式結束後，或它未啟動且已超過六十秒，認領單會標示為「已中斷」；此時 `cdui start` 會刪除認領單並正常啟動。詳見 **[讓伺服器重新啟動的安裝](/usage/optional-packs#讓伺服器重新啟動的安裝)**。

## 快取指令

有些節點會將可重新計算的結果寫入磁碟，除非手動刪除，否則不會清除。最大的快取來自 `LMTokenizedDataset`：它將整份語料封裝成 token 流，存放在 `<data>/cache/lm_blocks/`。每種語料、tokenizer、`seq_len`、`append_eos` 與 `max_tokens` 組合各產生一個檔案，每個 token 佔 8 bytes。100M-token 語料的每個檔案約 800 MB，因此以三個值掃描 `seq_len` 會留下三份完整副本。這些指令只處理可重新產生的快取，不會變更下載的模型與素材（由 `cdui packs remove` 刪除），也不會變更執行輸出、已儲存模型或 graph。

| 指令 | 說明 |
|------|------|
| `cdui cache list` | 列出每一個衍生快取：有幾個項目、佔多少磁碟空間，以及放在哪個目錄。 |
| `cdui cache prune` | 經過 `[y/N]` 確認後刪除這些項目。`--older-than DAYS` 會保留寫入時間比指定天數更近的項目；以最後**寫入**時間為準，讀取不算，因為快取命中不會修改檔案。`--yes`／`-y` 可略過確認，且沒有終端機時為必要選項。若有**背景**伺服器（`cdui start`）執行中，指令會拒絕作業，因為其中的 graph 可能正在讀取區塊檔案。此檢查只能偵測背景伺服器；前景 `cdui dev` 或 `cdui start -f` 不會寫入 pidfile，因此必須先手動停止。 |

兩個指令都接受 `--project <dir>`，與啟動伺服器時使用的旗標相同。專案模式的快取位於 `<dir>/assets/cache/`；未加此旗標時，指令會處理目前安裝的 `<data>/cache`。

腳本可使用下列離開碼：`0` 表示完成，包括沒有項目需要刪除；`1` 表示在提示中拒絕刪除，或無法刪除某個項目；`2` 表示執行前即遭拒，例如 `--older-than` 為負數或沒有終端機可進行確認；`3` 表示背景伺服器執行中，必須先執行 `cdui stop`；`130` 表示以 `Ctrl+C` 中斷。

## 專案指令 {/* #project-commands */}

專案目錄是作為服務儲存空間的 git 儲存庫。每個 graph 都有一份 logic 檔與一份 layout 檔，另有專案專用的 assets 與機密。完整流程請參閱 **[專案目錄](/usage/project-directories)**。

| 指令 | 說明 |
|------|------|
| `cdui project init <dir>` | 建立 `graphs/`、`layout/`、`assets/`、manifest、`.gitignore`、`.env.example` 與 `README.md`，再執行 `git init`。`--adopt <old-graphs-dir>` 會複製扁平 graphs 目錄中的每個 `*.json`，並把它拆成 logic 與 layout；`--force` 允許寫入非空目錄，但絕不覆寫既有的 manifest 或 `README.md`。 |
| `cdui project validate <dir>` | 載入完整的節點 registry，並對 `graphs/` 下的每個 graph 執行發佈驗證。如果 git 已追蹤 `.env`，此指令也會失敗。重複使用 `--graph <name>` 可指定要檢查的 graph。`--strict` 會將缺少 plugin pin 的警告視為錯誤。 |
| `cdui project freeze <dir>` | 將每個已安裝 GitHub plugin 的確切 commit SHA 寫入 manifest 的 `[plugins]` 表。連結的本機 plugin 會被略過。 |
| `cdui project restore <dir>` | 依 manifest 記錄的確切 SHA 安裝 plugin pin。在 CI 中，請先執行此指令再執行 `validate`。 |
| `cdui project publish <dir>` | 將 graph 發佈到本機伺服器，並在版本上記錄 git commit。`--graph` / `--slug` 可覆蓋 manifest 的 `[publish]` 目標；`--note` 加上一則不可變更的版本註記；`--create` 允許首次發佈到伺服器尚未認得的 `--slug`。 |

## 背景與前景

`cdui start` 預設在**背景**執行。關閉終端機後，伺服器仍會繼續執行。使用以下指令管理：

```bash
cdui status     # 即時儀表板 + 健康狀態
cdui stop       # 停止背景伺服器
cdui start -f   # 改為前景執行（Ctrl+C 停止）
```

背景執行時，伺服器的所有輸出都會寫入 `<install dir>/.codefyui_dev/server.log`；`cdui start` 與 `cdui status` 都會印出此路徑。

## `cdui start` 的參數

| 參數 | 說明 |
|------|------|
| `--foreground`、`-f` | 改為前景執行，不進入背景模式。由 systemd 等監督程式管理行程時必須使用。 |
| `--host <addr>` | 綁定位址（預設 `127.0.0.1`）。使用 `0.0.0.0` 或區網 IP 可讓其他機器連線；任何能存取此埠的人都可控制該實例，因此只應在信任的網路中使用。詳見[發佈](/usage/publish)。 |
| `--port <n>` | 埠號（預設 `8000`）。 |
| `--project <dir>` | 指定含有 `codefyui.project.toml` 的專案目錄 —— 請見[專案目錄](/usage/project-directories)。 |
| `--` | 單獨的 `--` 之後，所有參數都會原樣轉交 uvicorn，例如 `cdui start -- --proxy-headers --root-path /x`。`cdui start` 只解析分隔符號之前的參數，因此兩組參數不會衝突。在 uvicorn 參數區段使用 `--host`、`--port` 或 `--ws-max-size` 會遭拒（離開碼 `2`）：`cdui` 會自行記錄綁定位址，並從 `CODEFYUI_WS_MAX_MESSAGE_BYTES` 推導 WebSocket 上限。請改用 `cdui start --host` / `--port` 與該環境變數。 |

```bash
# 放在反向代理後面：只綁定回送位址，並信任代理轉發的標頭。
cdui start --host 127.0.0.1 --port 8000 -- --proxy-headers --forwarded-allow-ips 127.0.0.1
```

:::warning 加了代理還必須把它的主機名稱加進白名單
伺服器對任何不在允許清單中的 `Host` 都會回傳 `421`，包括網頁請求。因此，在將 `CODEFYUI_EXTRA_ALLOWED_HOSTS` 設為對外主機名稱前，透過代理開啟的頁面會是空白。完整說明請見 **[放在反向代理後面](/usage/deployment)**。
:::

## 環境變數 {/* #environment-variables */}

| 變數 | 讀取者 | 意義 |
|------|--------|------|
| `CODEFYUI_DIR` | 一行指令安裝程式 | 安裝目錄。預設：`~/CodefyUI`。 |
| `CODEFYUI_RELEASE_TAG` | 安裝程式、`cdui install`、`cdui update` | 要安裝的 release。前端 bundle 與後端 checkout 都會固定在該 tag。預設：`latest`。 |
| `CODEFYUI_FORCE_BUILD` | 安裝程式、`cdui install`、`cdui update` | 設為 `1` 時，會使用 pnpm 在本機建置前端而不下載 release bundle，並追蹤 `main`。 |
| `CODEFYUI_GPU` | `cdui install`、`cdui update` | `--gpu` 的預設值。命令列旗標優先。有效值請參閱[安裝](/getting-started/installation)。 |
| `CODEFYUI_DEV` | `cdui install`、`cdui update` | `--dev` 的預設值。使用 `1`、`true` 或 `yes` 啟用；使用 `0`、`false` 或 `no` 停用。 |
| `CODEFYUI_LANG` | 每個指令 | `cdui` 指令的輸出語言。英文值為 `en` 或 `english`；中文值為 `zh`、`zh-TW`、`zh-HK`、`zh-CN` 或 `chinese`。未設定時，由 `LANG` 與系統 locale 決定。 |
| `CODEFYUI_UV_INSTALL_TIMEOUT` | 每個可能需要 `uv` 的指令 | `PATH` 中找不到 `uv` 時，允許自動下載 `uv` 的秒數。預設：`180`。設為 `0` 表示不設上限。 |
| `CODEFYUI_USER_DATA_DIR` | `cdui start`、`cdui dev`、`cdui run`、`plugin` / `project` / `cache` / `packs` 指令群組，以及伺服器 | session token、plugin lockfile、asset cache、ChatGPT 登入與重新啟動安裝檔案的目錄。除非已自行匯出，否則這些指令會將它設為 `<install dir>/.codefyui_dev/`。請參閱[專案目錄](/usage/project-directories#6-建立-api-keyinvoke-需要)。 |
| `CODEFYUI_HOST`、`CODEFYUI_PORT` | 伺服器 | 綁定位址與埠號。`cdui start --host` 與 `--port` 會自動匯出這些值。只有手動啟動 uvicorn 時才需直接設定。伺服器會使用這些值推導 Host 允許清單與僅限 loopback 的安裝限制。請參閱[開發者安裝](/getting-started/dev-install)。 |
| `CODEFYUI_ENABLE_RESTART_INSTALL` | 伺服器 | 設為 `0` 會停用重新啟動伺服器的安裝。 |
| `CODEFYUI_GITHUB_TOKEN` | plugin 安裝 | `cdui plugin install`、`info`、`update`、`cdui project restore` 與外掛中心使用的 GitHub token。它會提高 GitHub 每個 IP 每小時 60 次未驗證 API 請求的上限。每次呼叫時讀取，只會傳送至 GitHub，且不會寫入 log。 |

`CODEFYUI_EXTRA_ALLOWED_HOSTS`、`CODEFYUI_WS_MAX_MESSAGE_BYTES`、`CODEFYUI_LOG_LEVEL`、`CODEFYUI_LOG_DIR` 與 `CODEFYUI_LOG_JSON` 請參閱[放在反向代理後面](/usage/deployment)。佇列設定請參閱[執行佇列](/usage/run-queue#設定)。

## 不啟動伺服器執行圖

你不需要網頁 UI 就能執行一張圖 —— 請見 **[CLI 圖形執行器](/usage/cli-runner)**：

```bash
cd backend
python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
```
