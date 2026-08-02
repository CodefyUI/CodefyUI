---
sidebar_position: 3
title: 外掛
description: 安裝教育節點的外掛包，並學習如何撰寫與發布你自己的外掛。
---

# 外掛包

教育（「Edu」）節點以可安裝的**外掛包**形式提供，**依方向**組織，因此每一個都對應到一個動手實作的教科書模組，並在你逐步學習時累進安裝。

```bash
cdui plugin install foundations deep rl   # full textbook companion
cdui plugin list
cdui plugin info deep                      # manifest, lessons covered, node names
cdui plugin search attention               # query the catalog
cdui plugin install foo/bar                # third-party pack from GitHub
cdui plugin uninstall deep
```

## 有哪些可用的外掛包

| 外掛包 | 動手實作模組 | 節點 |
|------|------------------|-----------|
| `foundations` | I1 Data Representation · I2 Classical ML | Edu-ColumnStats、Edu-KNN、Edu-LinearRegression、Edu-LogisticRegression、Edu-TokenEmbedding、Edu-FFN |
| `deep` | I3 Vision · I4 Sequences | Edu-CrossAttention、Edu-ResBlock、Edu-SelfAttention、Edu-MultiHeadAttention、Edu-Patchify |
| `rl` | I5 Reinforcement Learning | Edu-PolicyGradient |
| `stats` | —（任何資料集） | Stats-Describe、Stats-GroupByAggregate、Stats-Histogram、Stats-Percentile、Stats-Correlation、Stats-ConfusionMatrix、Stats-TableView、Stats-ChartView |

`stats` 是這裡的例外：它不是教科書的配套，而是給第三方外掛作者的實作範例。它只用 numpy 與 torch，以[第 0 級](#安全性三個層級)安裝且**整份 manifest 沒有 `[security]` 區段**；它的 [README](https://github.com/CodefyUI/CodefyUI/blob/main/plugins/stats/README.md) 正式記載了資料類外掛需要的兩份契約——表格如何在連接埠之間傳遞，以及 `chart` 輸出如何宣告與繪製。

每個 Edu 節點都把單一課程概念分解成一連串具名步驟，由 [Teaching Inspector](/usage/teaching-inspector) 一次渲染一列——`Edu-ColumnStats` 將母體標準差公式呈現為 `sum → divide → deviations² → variance → sqrt`；`Edu-PolicyGradient` 暴露 `softmax → gather → log → baseline → loss`；`Edu-Patchify` 讓 `unfold → permute → flatten` 變得可見。在 Settings popover 中開啟 **Verbose mode** 即可擷取它們。

## 外掛包如何儲存

- **內建方向外掛包**位於 repo 內的 `plugins/<id>/`，並就地啟用（不複製）。
- **第三方外掛包**會以固定 SHA 的 tarball 下載到 `<USER_DATA>/plugins/<id>/`，並在安裝前經過 **AST 驗證**（見[安全性](#安全性三個層級)）。
- 位於 `<USER_DATA>/plugins/installed.json` 的 lockfile 會記錄每一次安裝——包含你授權了哪些能力——因此 `cdui start` 會在下次啟動時重新探索它們。

外掛節點會加上命名空間，以避免衝突並讓圖能自我說明——內建節點使用像 `Conv2d` 這樣的裸名稱，而外掛節點則會像 `foundations:Edu-KNN` 這樣加上限定。

## 安全性——三個層級

外掛包是在 CodefyUI 行程內執行的 Python。第三方外掛包安裝前，其中每一個 `.py` 檔都會被 AST 閘門走訪，決定它可以 import 什麼。閘門有三種答案，而中間那一種才是重點。

| 層級 | 外掛如何取得 | 涵蓋範圍 |
|------|----------------------|----------------|
| **0——預設** | 不需宣告 | 純運算：`math`、`statistics`、`collections`、`itertools`、`functools`、`json`、`re`、`dataclasses`、`typing`、`enum`、`decimal`、`random`、`numpy`、`torch`、`pandas`——外加路徑輔助函式（見下）。所有第一方外掛包都在這一級。 |
| **1——宣告能力** | manifest 寫 `[security] capabilities = [...]`，並在安裝時由使用者確認 | 每個能力對應一組具名模組。 |
| **2——信任作者** | `[security] allowed_modules = [...]` **加上** `cdui plugin install --trust-author` | 任何東西，包括 `subprocess`、`ctypes` 與 `importlib`。 |

### 能力清單

| 能力 | 解鎖 | 你正在同意的事 |
|------------|---------|--------------------------|
| `network` | `requests`、`urllib`、`http`、`socket` | 這個外掛可以與任何主機收發資料——**並把下載到的內容寫入磁碟**，因為 `urllib.request.urlretrieve(url, dest)` 只要一行。 |
| `filesystem` | `pathlib`、`tempfile`、`shutil`、`zipfile`、`tarfile`、`gzip`、`bz2`、`lzma`、`codecs`、`sqlite3`、`glob`、`fileinput` | 這個外掛可以使用檔案**函式庫**。這不是寫入的邊界：單純的 `open(p, "w")` 是內建函式，完全不需要任何宣告（見[這不是什麼](#這不是什麼)）。 |
| `process-env` | `os`、`ntpath`、`posixpath`、`genericpath` | 這個外掛拿到**整個 `os` 模組**：讀取*並修改*此行程的環境變數（**包含其中的 API 金鑰**）、啟動其他程式（`os.execv`、`os.spawnve`、`os.startfile`），以及刪除或重新命名檔案。這個名字是大家索取它的理由，但授予的範圍比名字大。 |

除此之外都不是能力。`subprocess`、`sys`、`importlib`、`ctypes`、`pickle`、`marshal`、`dill`、`shelve`、`runpy`、`code`、`signal`、`atexit`、`webbrowser`、`threading`、`asyncio`、`multiprocessing` 一律只能走第 2 級：**沒有任何能力會交出一個「本身就是用來執行程式碼、或伸手進入直譯器」的模組。** 請注意這句話的精確之處——`process-env` 授予 `os`，而 `os` 會啟動行程。任何能力都不會給你的，是一個為執行程式碼而生的模組。

### 路徑輔助函式屬於第 0 級

`os.path.join` 是字串處理，因此不需要任何能力——但僅限於**唯一一種**綁定輔助函式本身的寫法，而且它綁定的每一個名稱都必須是純函式：

```python
from os.path import join, basename   # 可以，第 0 級
from os import path                  # 需要 "process-env"
from os.path import genericpath      # 需要 "process-env"
import os                            # 需要 "process-env"
import os.path                       # 一樣綁定 `os`——需要 "process-env"
import ntpath / posixpath            # 需要 "process-env"
```

被拒絕的那幾行不是吹毛求疵。`os.path` **就是** `ntpath` / `posixpath`，而這兩個模組在模組層級執行 `import os` 與 `import sys`，並把兩者都留成一般屬性——所以 `path.os.remove(p)` 會刪掉檔案，`path.sys.modules['subprocess'].run([...])` 會執行指令。`from os.path import <名稱>` 只在該名稱不是模組時才被允許，而這份「哪些名稱是模組」的清單是從執行中的 `os.path` 讀出來的，不是寫死的。

### 宣告，以及被詢問

```toml
[security]
capabilities = ["network"]
```

```console
$ cdui plugin install alice/metric-logger
  Source: https://github.com/alice/metric-logger
  Ref: default branch (a1b2c3d)

> 此外掛要求下列能力
    network → 連線網路——可與任何主機收發資料，並把下載到的內容寫入磁碟（requests、urllib、http、socket）
  能力是宣告，不是沙箱：授權後外掛就能使用該類模組，CodefyUI 不會再逐一攔截。
  要授權嗎？ [y/N]:
```

- **沒有終端機時**（腳本、CI、以管線輸入的安裝）答案一律是**否**，訊息會指出 `--accept-capabilities`——它會直接授予 manifest 宣告的那一組而不詢問。`-y` / `--no-confirm` **不會**連帶授權：那個旗標跳過的是「要從這個 URL 安裝嗎？」，而同意一段會連線網路的程式碼是另一個問題。
- **授權內容會被記錄**在 `<USER_DATA>/plugins/installed.json`，並由 `cdui plugin list` 與 `cdui plugin info` 顯示。
- **`cdui plugin update` 不會重複詢問**——只要新版要求的是你已授權範圍的子集；一旦它多要了一項能力就會**停下來**，而這正是更新流程真正攔得到的供應鏈風險形狀。

### 每一級都成立的規則

`torch.load(...)` 仍然必須明確寫出 `weights_only=True`；dunder 存取（`__class__`、`__globals__`、`__subclasses__`……）、frame 走訪（`f_globals`、`gi_frame`……）、`eval` / `exec` / `compile` / `__import__`，以及 `os.system` / `os.popen` / `os.spawn*`，不論宣告了什麼都一律拒絕。能力永遠買不到反射能力。

### 這不是什麼

**這是防護欄，不是沙箱**——與[畫布內腳本政策](/advanced/python-script-node)同一套說法，而且在這裡更值得重講一次，因為這是**別人的**程式碼真正執行的地方。

- **閘門讀的是你外掛自己的 `import` 敘述。** 它不會去讀你 import 的函式庫，也無法判斷一個被允許的函式實際上做了什麼。
- **能力攔的是 *import*，不是 *行為*。** 有兩個後果值得直接講明，而不是留給你自己踩到：
  - **`filesystem` 並不會攔截寫檔。** `open(p, "w")` 是內建函式、不需要 import，在第 0 級什麼都不宣告就能過。我們考慮過攔它然後放棄了：模式字串經常是算出來的（`open(p, "w" if overwrite else "r")`），所以這個檢查只要一個變數就能繞過，卻會誤傷誠實的外掛——是一個沒有相應安全價值的誤判。
  - **`network` 隱含了寫檔能力**，透過 `urllib.request.urlretrieve(url, dest)`。
- **能力涵蓋的是黑名單上的模組根名稱，不是整個類別。** `requests` 有被攔；`httpx` 從來就不在黑名單上，所以一個 import 它的外掛什麼都不用宣告就能連網。把 PyPI 上每一個 HTTP 用戶端都列進清單，不是清單做得到的事。
- **有兩條路徑刻意完全跳過閘門。** 內建外掛包隨這個 repo 一起發行、由 PR 審查；`cdui plugin link` 載入的是**你自己的**工作目錄，並且會印出警告說明。`cdui project restore` 也會以非互動方式授予專案 manifest 宣告的能力——它本來就帶著 `--trust-author`，所以不會增加額外曝險，但這代表一份專案檔本身也是一個信任決定。
- **任何能寫入 `installed.json` 的東西，都能預先批准下一次更新。** lockfile 正是 `cdui plugin update` 用來「這個能力已授權過、不必再問」的依據，所以能編輯它的程式碼（包含已取得 `filesystem` 的外掛，或任何用 `open` 的外掛）都可以在自己的條目裡加上一項能力，讓下一次更新靜默接受。這屬於入侵後的持久化，而非第一步的提權——但 lockfile 是一份信任存放區，它的保護程度就等於你帳號的保護程度。
- **宣告是作者的意圖聲明。** 它提高了順手攻擊的成本，也讓你在同意前有東西可讀。真正該問的問題仍然是：「我信任寫這個東西的人嗎？」

### 從舊版升級

不需要做任何事。在能力機制出現之前寫入的 lockfile 條目沒有 `capabilities` 欄位，讀起來就是「未授權任何能力」——與它原本的行為完全一致。既有的外掛包重新驗證後行為不變。

## 撰寫你自己的外掛

最快的起手式是 **`cdui plugin new`**，一個指令就能產生可直接編輯的外掛骨架：

```bash
cdui plugin new my-plugin          # 純後端骨架
cdui plugin new my-plugin --ui     # 另含一個接好 SDK 的 React 前端
```

它會產生 manifest、一個範例節點、一個測試（內含 `cdui_plugins.<id>` 命名空間 shim，讓本地 `pytest` 可直接執行），並在加上 `--ui` 時產生一個 Vite + React 的 `ui/`，其 `src/sdk/` 即為型別化的外掛 SDK。外掛會建立在 `./my-plugin/`；用下方的 `cdui plugin dev` 連結後即可開始編輯。

若需更完整的參考，可 Fork **[官方外掛模板](https://github.com/treeleaves30760/CodefyUI-Plugin-Official)**——一個可運作、採 MIT 授權的外掛，包含兩個範例節點、一張範例圖、一套測試，以及一份完整註解的資訊清單 (manifest)。它的 README 逐欄解說每個欄位與 AST 安全閘門。

```bash
# Install the template itself to see the pattern live
cdui plugin install treeleaves30760/CodefyUI-Plugin-Official

# After forking
cdui plugin install your-username/your-fork
```

一個外掛包可隨附下列任意內容：一個 `nodes/` 目錄（自動探索）、一個 `presets/` 目錄、一個 `examples/` 目錄，以及一個 `assets/` 目錄（掛載於 `/plugins/<id>/assets/<file>`）。一份 `cdui.plugin.toml` 資訊清單宣告 id、版本、`requires_codefyui`、內容目錄、課程 metadata，以及——只在你需要時——[安全性](#安全性三個層級)一節描述的 `[security]` 宣告。若你的節點只做純運算，直接刪掉那一段就好；大多數外掛都是如此。

:::warning 破壞性變更（v0.3）
章節外掛包 `c1`–`c6` 已重新封裝為三個方向外掛包 `foundations` / `deep` / `rl`，而且每個 Edu 節點的型別 id 都加上了一個破折號（`EduKNN` → `Edu-KNN`）。引用舊有 `cN:EduFoo` 型別的已儲存圖必須更新為 `<pack>:Edu-Foo`，並以 `cdui plugin install foundations deep rl` 重新安裝這些外掛包。
:::

## 本地開發

開發外掛時，不必每次迭代都先推上 GitHub。用 **link** 連結你的工作目錄，CodefyUI 會就地載入：

```bash
cdui plugin link ./my-plugin     # 就地註冊本地目錄（不複製）
# ...編輯 nodes/ 或 frontend/...
cdui plugin reload               # 讓執行中的伺服器套用變更
cdui plugin unlink my-plugin     # 解除連結——你的檔案不會被刪除
```

更簡單的方式是用 **`dev`**，一個指令完成連結＋監看，每次存檔都自動熱重載：

```bash
cdui plugin dev ./my-plugin      # 連結＋監看；每次變更自動重載
```

請在另一個終端機執行伺服器（`cdui start` 或 `cdui dev`）。`dev` 會輪詢外掛的 manifest、`nodes/`、`presets/` 與 `frontend/`；`--once` 只連結並重載一次（不監看），`--interval` 可調整輪詢間隔。`link`、`dev`、`reload` 會連到伺服器設定的連接埠（`CODEFYUI_PORT`，預設 `8000`），因此跑在非預設埠時不需額外旗標。

`link` 會從你的 `cdui.plugin.toml` 讀取 id，並把該目錄的絕對路徑以 `source_kind = "local"` 記入 lockfile，因此探索會直接走訪你的工作目錄。連結的外掛會跳過 AST 安全閘門（這是你自己的程式碼，並會印出警告）；`unlink` 只移除 lockfile 條目，絕不刪除你的檔案。編輯 Python 節點後，執行 `cdui plugin reload`（或 `cdui plugin dev`）即可重載。**連結中的外掛，前端變更也會自動重載**——只要安裝著連結的外掛，編輯器就會偵測重載並就地重新掛載外掛 UI，不需手動重新整理瀏覽器。

:::tip 開發資料隔離
透過 `scripts/dev.py` 執行外掛指令——或設定 `CODEFYUI_USER_DATA_DIR`——可讓某個 clone 的 lockfile 留在 repo 內（`.codefyui_dev/`），而非全機共用的 user-data 目錄，避免多個 clone 互相覆蓋。
:::

## REST API

| 端點 | 方法 | 說明 |
|----------|--------|-------------|
| `/api/plugins` | GET | 列出已安裝的外掛包。 |
| `/api/plugins/{id}` | GET | 取得某外掛的資訊清單 (manifest) 與 README。 |
| `/api/plugins/reload` | POST | 熱重載所有節點與預設模組來源。 |
