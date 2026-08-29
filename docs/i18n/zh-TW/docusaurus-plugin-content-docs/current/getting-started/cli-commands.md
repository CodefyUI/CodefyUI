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
| `cdui update` | 更新到最新 release（prebuilt 路徑），或拉取 `main`（原始碼建置）並重新同步前端。不會詢問任何問題 —— 除非用 `--gpu` / `--dev` 覆蓋，否則沿用 venv 中既有的 PyTorch 變體與 dev 工具。伺服器還在執行時會拒絕（它會刪掉那台伺服器正在服務的 `frontend/dist`）—— 請先 `cdui stop`。 |
| `cdui start` | 正式模式 —— 單一 uvicorn 跑 `:8000`，在背景執行（不需要 Node）。`--foreground`／`-f` 改為前景執行。 |
| `cdui status` | btop／k9s 風格的儀表板：CPU、記憶體、磁碟、GPU、前幾名程序，外加伺服器的 PID 與健康狀態。即時刷新（每 2 秒；`Ctrl+C` 離開）。傳入一個數字可設定間隔（`cdui status 1`），或用 `--once` 只顯示單一畫面。它也會回報會重新啟動伺服器的套件包安裝：只要認領單還在就會多出一行「重啟安裝」—— 輔助程式還在做事時顯示「收尾中」，不在了則顯示「已中斷」—— 結束後的一小時內則會多出一行「上次重啟安裝」。 |
| `cdui dev` | 開發者模式 —— 後端 `:8000` + Vite HMR `:5173`（需要 Node + pnpm）。 |
| `cdui build` | 在本地建置前端 bundle（需要 Node + pnpm）。 |
| `cdui stop` | 停止**這個安裝**的服務：pidfile 記錄的背景伺服器，加上從這個目錄啟動的殘留行程（前景 `cdui start`、`cdui dev` 的 Vite、遺留 worker）。加上 `--all` 則改為停止整台機器上所有 CodefyUI 與 Vite 行程 —— 那會波及其他使用者的伺服器與無關的 Vite dev server，共用主機請勿使用。 |
| `cdui test` | 執行整個專案的測試：後端（`pytest`）與前端（`vitest`）。沒有 pnpm 時前端那半會標成 `SKIPPED` 而不是失敗 —— release 安裝本來就沒有 Node。兩邊一定都會跑完，所以跑一次就能同時知道兩邊的結果；任一邊失敗就以離開碼 1 結束。用 `--backend` / `--frontend` 可以只跑其中一邊。 |
| `cdui clean` | 移除虛擬環境、`node_modules` 與 `frontend/dist`。 |
| `cdui uninstall` | clean + 移除 PATH 上的啟動器。 |

## 外掛指令

| 指令 | 說明 |
|------|------|
| `cdui plugin install <name\|url>` | 安裝一個外掛包（型錄名稱如 `foundations`、`owner/repo[@ref]`，或完整的 GitHub URL）。 |
| `cdui plugin sync` | 安裝所有你還沒做過決定的**內建**外掛包——升級後多了新外掛包時，只要跑這一個指令。會先確認一次（`--yes` 可略過確認；沒有終端時必須加）；`--dry-run` 只列出清單；`--prune` 會順手清掉已不再隨版本發行的內建外掛 lockfile 項目。你自己移除過的外掛包不會被裝回來。 |
| `cdui plugin list` | 列出已安裝的外掛包，以及還在等你決定的內建外掛包。 |
| `cdui plugin info <id>` | 顯示某個外掛包的 manifest、涵蓋的課程與節點名稱。 |
| `cdui plugin search <query>` | 查詢外掛型錄。 |
| `cdui plugin uninstall <id>` | 移除一個已安裝的外掛包。若是內建外掛包，這個移除會被記住，`cdui plugin sync` 不會再把它裝回來；要拿回它就執行 `cdui plugin install <id>`。 |

完整的外掛工作流程請見 **[外掛](/advanced/plugins)**。

## 套件包指令

選用套件包是預設安裝刻意不放進來的大型附加內容 —— `sentence-transformers`、各個嵌入模型、GloVe 詞向量表、加速版的 PyTorch。應用程式裡的**套件中心**會帶著進度條幫你安裝；下面這些指令從終端做同樣的事，而且若某個套件包必須換掉執行中伺服器已經載入的東西，只能靠這裡裝。

| 指令 | 說明 |
|------|------|
| `cdui packs list` | 列出型錄中的每個套件包：裡面有什麼、下載要花多少空間，以及哪些已經裝好了。 |
| `cdui packs status` | 跟 `list` 一樣，另外加上這個 venv 裡的 PyTorch 版本，以及接下來該執行什麼。 |
| `cdui packs install <id>` | 安裝一個套件包。`--items a,b` 只下載指定的模型（預設會補齊這個套件包缺的全部）；`--yes`／`-y` 可跳過下載大小的確認，沒有終端可確認時必須加。只接受型錄裡的 id —— 沒有任何方式可以傳入套件安裝字串、index 網址或 repo id。 |
| `cdui packs remove <id> <item-id>` | 刪除一個已下載的模型並清掉紀錄。套件包的 Python 套件不會一併移除；能移除它們的 `uv pip uninstall` 指令會印出來，請在伺服器停掉後自行執行。 |

離開碼（給腳本用）：`0` 完成、`1` 安裝失敗或在確認時選了不要、`2` 還沒開始就拒絕（id 不存在、相依未滿足、沒有終端可確認）、`3` 伺服器執行中無法進行 —— 會印出改用的指令、`130` 以 `Ctrl+C` 中斷。

**會重新啟動伺服器的安裝。** 用 `cdui start` 啟動的伺服器，可以靠「先離開、再回來」把 GPU PyTorch 套件包裝好 —— 線上安裝撞到相依衝突的套件包也一樣：它會先把要裝什麼寫下來，以分離的方式啟動 `cdui packs-run-pending`，然後把自己關掉；那個輔助程式會等這個行程真的消失，執行安裝，記下結果，再用這次 `cdui start` 收到的參數把伺服器啟動回來。`packs-run-pending` 是**內部指令**，而且刻意不出現在說明文字裡 —— 交給它的檔案指名了一個「要等它結束」的行程，所以拿去對著執行中的伺服器手動跑，只會等兩分鐘然後把它停掉。若某台機器上重新啟動回不來，用 `CODEFYUI_ENABLE_RESTART_INSTALL=0` 可以把整條路徑關掉。而在一次重新啟動安裝還在「收尾中」時 —— 它記下的輔助程式還活著，或那張認領單還不到六十秒、輔助程式也還沒把行程編號寫進去 —— `cdui start` 不會在那個輔助程式正在改寫的 venv 上再起第二台伺服器：它會說有一次重新啟動安裝正在收尾，要你去看 `cdui status`，然後返回。等輔助程式不在了，或它根本沒出現而那六十秒已經過去，這張認領單就算「已中斷」，這時 `cdui start` 會刪掉它並照常啟動。詳見 **[讓伺服器重新啟動的安裝](/usage/optional-packs#讓伺服器重新啟動的安裝)**。

## 背景與前景

`cdui start` 預設在**背景**執行 —— 關閉 terminal 而伺服器會繼續運作。用以下指令管理它：

```bash
cdui status     # 即時儀表板 + 健康狀態
cdui stop       # 停止背景伺服器
cdui start -f   # 改為前景執行（Ctrl+C 停止）
```

## `cdui start` 的參數

| 參數 | 說明 |
|------|------|
| `--foreground`、`-f` | 改為前景執行，不進背景。當行程是由 systemd 之類的管理者看著時必須用這個。 |
| `--host <位址>` | 綁定位址（預設 `127.0.0.1`）。`0.0.0.0` 或區網 IP 可讓其他機器連進來 —— 任何連得上這個埠的人都能控制此實例，所以只在信任的網路使用。請見[發佈](/usage/publish)。 |
| `--port <n>` | 埠號（預設 `8000`）。 |
| `--project <目錄>` | 指定含有 `codefyui.project.toml` 的專案目錄 —— 請見[專案目錄](/usage/project-directories)。 |
| `--` | 單獨一個 `--` 之後的所有參數會原樣轉給 uvicorn，例如 `cdui start -- --proxy-headers --root-path /x`。`cdui start` 只從分隔符號之前的部分讀自己的參數，所以兩邊不會互相干擾。`--host` 與 `--port` 在那裡會被拒絕（離開碼 2），因為綁定位址是由 `cdui` 自己記錄的 —— 請改用 `cdui start --host`。 |

```bash
# 放在反向代理後面：只綁定回送位址，並信任代理轉發的標頭。
cdui start --host 127.0.0.1 --port 8000 -- --proxy-headers --forwarded-allow-ips 127.0.0.1
```

:::warning 加了代理還必須把它的主機名稱加進白名單
伺服器對任何不認得的 `Host` 一律回 `421`，連網頁本身也不例外，所以在你把 `CODEFYUI_EXTRA_ALLOWED_HOSTS` 設成對外名稱之前，代理後面看到的會是一片空白的網頁。完整說明請見 **[放在反向代理後面](/usage/deployment)**。
:::

## 不啟動伺服器執行圖

你不需要網頁 UI 就能執行一張圖 —— 請見 **[CLI 圖形執行器](/usage/cli-runner)**：

```bash
cd backend
python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
```
