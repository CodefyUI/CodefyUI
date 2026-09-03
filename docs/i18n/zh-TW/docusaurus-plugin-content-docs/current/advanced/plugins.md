---
sidebar_position: 3
title: 外掛
description: 安裝教育節點的外掛包，並學習如何撰寫與發布你自己的外掛。
---

# 外掛包

教育（「Edu」）節點以可安裝的**外掛包**形式提供，**依方向**組織，因此每一個都對應到一個動手實作的教科書模組，並在你逐步學習時累進安裝。

```bash
cdui plugin sync                           # 安裝所有你還沒決定過的內建外掛包
cdui plugin install foundations deep rl   # 或者一個一個挑
cdui plugin list
cdui plugin info deep                      # manifest, lessons covered, node names
cdui plugin search attention               # query the catalog
cdui plugin install foo/bar                # third-party pack from GitHub
cdui plugin uninstall deep                 # 會被記住：sync 不會再把它裝回來
```

## 有哪些可用的外掛包

| 外掛包 | 動手實作模組 | 節點 |
|------|------------------|-----------|
| `foundations` | I1 Data Representation · I2 Classical ML | Edu-ColumnStats、Edu-KNN、Edu-LinearRegression、Edu-LogisticRegression、Edu-TokenEmbedding、Edu-FFN |
| `deep` | I3 Vision · I4 Sequences | Edu-CrossAttention、Edu-ResBlock、Edu-SelfAttention、Edu-MultiHeadAttention、Edu-Patchify |
| `rl` | I5 Reinforcement Learning | Edu-PolicyGradient |
| `edu` | I1 Data Representation · I2 Classical ML（動手實作） | FilterRows、SlidingWindow2D、SentenceEmbedding、Classifier、AdvancedClassifier、FFNLayer、ActivationLayer、TrainAndEvaluate |
| `stats` | —（任何資料集） | Stats-Describe、Stats-GroupByAggregate、Stats-Histogram、Stats-Percentile、Stats-Correlation、Stats-ConfusionMatrix、Stats-TableView、Stats-ChartView |

`stats` 是這裡的例外：它不是教科書的配套，而是給第三方外掛作者的實作範例。它只用 numpy 與 torch，以[第 0 級](#安全性三個層級)安裝且**整份 manifest 沒有 `[security]` 區段**；它的 [README](https://github.com/CodefyUI/CodefyUI/blob/main/plugins/stats/README.md) 正式記載了資料類外掛需要的兩份契約——表格如何在連接埠之間傳遞，以及 `chart` 輸出如何宣告與繪製。

另外有三個官方外掛放在各自的儲存庫裡。它們一樣可以用目錄中的名稱安裝，但和上面的外掛包不同：它們是從 GitHub 下載的，所以安裝時會請你確認。

| 外掛 | 這是什麼 | 安裝 |
|------|------------------|-----------|
| `graph-copilot` | AI 對話助手：以對話建立與修改節點圖，執行你核准的隔離實驗與參數搜尋，並保留可攜的實驗紀錄。需要一個 LLM 供應者（Codex、Ollama、OpenAI 或 Anthropic）。 | `cdui plugin install graph-copilot` |
| `self-learning` | 把一個自由描述的機器學習問題變成逐步教材：先由 LLM 在畫布上建出可實際執行的圖並驗證它跑得起來，再由外掛擷取每一步的截圖，產出繁體中文 Markdown 教材、可列印頁面、圖檔與一份入門練習。 | `cdui plugin install self-learning` |
| `official-template` | 給外掛作者的可運作起始模板：兩個範例節點、一個 preset、一張範例流程圖、一個資產檔、一套範例測試，以及一個 React 工具面板。裝起來看外掛能做什麼，Fork 它來寫自己的。 | `cdui plugin install official-template` |

每個 Edu 節點都把單一課程概念分解成一連串具名步驟，由 [Teaching Inspector](/usage/teaching-inspector) 一次渲染一列——`Edu-ColumnStats` 將母體標準差公式呈現為 `sum → divide → deviations² → variance → sqrt`；`Edu-PolicyGradient` 暴露 `softmax → gather → log → baseline → loss`；`Edu-Patchify` 讓 `unfold → permute → flatten` 變得可見。在 Settings popover 中開啟 **Verbose mode** 即可擷取它們。

## 外掛包如何儲存

- **內建方向外掛包**位於 repo 內的 `plugins/<id>/`，並就地啟用（不複製）。
- **第三方外掛包**會以固定 SHA 的 tarball 下載到 `<USER_DATA>/plugins/<id>/`，並在安裝前經過 **AST 驗證**（見[安全性](#安全性三個層級)）。
- 位於 `<USER_DATA>/plugins/installed.json` 的 lockfile 會記錄每一次安裝——包含你授權了哪些能力——因此 `cdui start` 會在下次啟動時重新探索它們。它同時也是「已安裝」的定義：要蓋過一個目錄被手動刪掉的外掛包，或蓋過一個你用 `cdui plugin link` 連結的外掛，都必須加上 `--force`，不會再默默覆寫——換掉這兩者都是該由鍵盤前的人來做的決定。

外掛節點會加上命名空間，以避免衝突並讓圖能自我說明——內建節點使用像 `Conv2d` 這樣的裸名稱，而外掛節點則會像 `foundations:Edu-KNN` 這樣加上限定。

### 升級後補上新外掛包——`cdui plugin sync`

真正啟用一個外掛包的是 lockfile，而升級並不會去寫它。所以當某個版本**新增**了內建外掛包時，它的檔案會隨升級落到你的磁碟上，卻沒有任何東西去載入它：那些節點同時處於「可以安裝」與「完全看不到」的狀態。`cdui plugin sync` 就是這個補課動作——它會安裝所有你還沒做過決定的內建外掛包，動手前先確認一次，並且逐一回報結果，所以某個外掛包的 `python_deps` 在學校網路下載不下來時，不會把其他外掛包一起拖垮。

```bash
cdui plugin sync --dry-run   # 只告訴我還有哪些沒裝
cdui plugin sync             # 全部安裝（確認一次）
cdui plugin sync --yes        # 不詢問——腳本、CI、教室映像檔
cdui plugin sync --prune      # 順手清掉已不再發行的外掛 lockfile 項目
```

有兩件事它刻意不做。它不會在啟動時自動執行，`cdui update` 也不會主動問你要不要跑：因為某個版本剛好帶了某段程式碼就替你啟用它，是一個關於同意的決定，不是升級的細節。它也絕不會把你移除過的外掛包裝回來——`cdui plugin uninstall` 會把這次移除記錄在 lockfile 裡（`plugins` 旁邊的 `removed` 對應表），讓「我從沒見過這個外掛包」與「我把這個外掛包丟掉了」不再是同一個狀態。`cdui start` 與 `cdui plugin list` 也因為同樣的理由，不會再提起你移除過的外掛包。想反悔就用名稱重新安裝：`cdui plugin install stats` 會清掉那筆記錄，sync 之後也會重新把它算進去。

## 安全性——三個層級

外掛包是在 CodefyUI 行程內執行的 Python。第三方外掛包安裝前，包內任何位置的每一個 `.py` 檔——`nodes/`、`examples/`、`tests/`、`docs/`、`assets/`，或其他任何子目錄——都會被 AST 閘門走訪，決定它可以 import 什麼。不會因為目錄名稱而被排除：外掛載入器可以從包內任何地方 import（節點檔裡寫 `from ..tests import helper` 是可行的），所以掃描範圍必須涵蓋載入器觸及得到的每一個角落。閘門有三種答案，而中間那一種才是重點。

| 層級 | 外掛如何取得 | 涵蓋範圍 |
|------|----------------------|----------------|
| **0——預設** | 不需宣告 | 純運算：`math`、`statistics`、`collections`、`itertools`、`functools`、`json`、`re`、`dataclasses`、`typing`、`enum`、`decimal`、`random`、`numpy`、`torch`、`pandas`——外加路徑輔助函式（見下）。所有第一方外掛包都在這一級。 |
| **1——宣告能力** | manifest 寫 `[security] capabilities = [...]`，並在安裝時由使用者確認 | 每個能力對應一組具名模組。 |
| **2——信任作者** | `[security] allowed_modules = [...]` **加上** `cdui plugin install --trust-author` | 任何東西，包括 `subprocess`、`ctypes` 與 `importlib`。 |

### 能力清單

| 能力 | 解鎖 | 你正在同意的事 |
|------------|---------|--------------------------|
| `network` | `requests`、`urllib`、`http`、`socket`、`ssl`，以及它們背後的原始 C 模組（`_socket`、`_ssl`） | 這個外掛可以與任何主機收發資料——**並把下載到的內容寫入磁碟**，因為 `urllib.request.urlretrieve(url, dest)` 只要一行。 |
| `filesystem` | `pathlib`、`tempfile`、`shutil`、`zipfile`、`tarfile`、`gzip`、`bz2`、`lzma`、`codecs`、`sqlite3`（含 `_sqlite3`）、`glob`、`fileinput`、`readline` | 這個外掛可以使用檔案**函式庫**。這不是寫入的邊界：單純的 `open(p, "w")` 是內建函式，完全不需要任何宣告（見[這不是什麼](#這不是什麼)）。 |
| `process-env` | `os`、`ntpath`、`posixpath`、`genericpath`、`nt`、`posix` | 這個外掛拿到**整個 `os` 模組**：讀取*並修改*此行程的環境變數（**包含其中的 API 金鑰**）、啟動其他程式（`os.execv`、`os.spawnve`、`os.startfile`），以及刪除或重新命名檔案。這個名字是大家索取它的理由，但授予的範圍比名字大。 |

除此之外都不是能力。`subprocess`、`sys`、`importlib`、`ctypes`、`pickle`、`marshal`、`dill`、`shelve`、`runpy`、`code`、`signal`、`atexit`、`webbrowser`、`threading`、`asyncio`、`multiprocessing` 一律只能走第 2 級：**沒有任何能力會交出一個「本身就是用來執行程式碼、或伸手進入直譯器」的模組。** 請注意這句話的精確之處——`process-env` 授予 `os`，而 `os` 會啟動行程。任何能力都不會給你的，是一個為執行程式碼而生的模組。

### 路徑輔助函式屬於第 0 級

`os.path.join` 是字串處理，因此不需要任何能力——但僅限於**唯一一種**綁定輔助函式本身的寫法，而且只限於那些真的是純字串函式的名稱：

```python
from os.path import join, basename   # 可以，第 0 級
from os.path import expandvars       # 需要 "process-env"——會讀 os.environ
from os.path import exists, getsize  # 需要 "process-env"——真的會 stat()
from os.path import genericpath      # 需要 "process-env"——那是模組
from os import path                  # 需要 "process-env"——綁定的是 ntpath
import os / import os.path           # 需要 "process-env"
import ntpath / posixpath            # 需要 "process-env"
import nt / posix                    # 需要 "process-env"——os.py 賴以建構自身的原始模組
```

第 0 級的清單就是：`join`、`basename`、`dirname`、`split`、`splitext`、`splitdrive`、`normpath`、`normcase`、`isabs`、`commonpath`、`commonprefix`，以及 `sep` / `altsep` / `extsep` / `pathsep` / `curdir` / `pardir` / `defpath` 這些常數。

被拒絕的那幾行不是吹毛求疵——`os.path` 是一個真正的模組，而它的表面大部分都不是字串處理：

- `os.path` **就是** `ntpath` / `posixpath`，這兩個模組在模組層級執行 `import os` 與 `import sys`，並把兩者都留成一般屬性——所以 `path.os.remove(p)` 會刪掉檔案，`path.sys.modules['subprocess'].run([...])` 會執行指令。
- `os` 本身**就是** `nt`（Windows）或 `posix`（POSIX）——CPython 自己的 `os.py` 執行 `from nt import *` / `from posix import *`，`os.remove`、`os.environ`、`os.system` 都是從這裡來的。直接以名稱 import 這個原始模組，中間沒有任何攔截，會拿到一模一樣的介面。
- `expandvars("%WANDB_API_KEY%")` 會回傳該環境變數的值——正是 `process-env` 存在要攔的東西——而 `expanduser("~")` 會回傳你的家目錄。
- `exists`、`isfile`、`isdir`、`getsize`、`getmtime` 這一類會對你指定的任何路徑呼叫 `stat()`；`abspath`、`realpath`、`relpath` 則會依工作目錄解析，因而洩漏 CodefyUI 安裝在哪裡。

第 0 級清單上的每一個名稱都是靠**實際呼叫**驗證的，不是靠讀原始碼——在 Windows 上 `abspath` 會走到 `nt._getfullpathname`，而一份只找 `os.` 用法的原始碼稽核看不到它。

### 宣告，以及被詢問

```toml
[security]
capabilities = ["network"]
```

```console
$ cdui plugin install alice/metric-logger

> 安裝外掛：alice/metric-logger
  來源：https://github.com/alice/metric-logger
  版本：default branch (a1b2c3d)
  Metric Logger 0.4.0
  Ships each run's metrics to a collector.
  Python 套件：httpx>=0.27
  繼續？ [y/N]: y

> 此外掛要求下列能力
    network -> 連線網路——可與任何主機收發資料，並把下載到的內容寫入磁碟（requests、urllib、http、socket）
  能力是宣告，不是沙箱：授權後外掛就能使用該類模組，CodefyUI 不會再逐一攔截。
  要授權嗎？ [y/N]: y
  Resolving alice/metric-logger
  Downloading alice/metric-logger@a1b2c3d
    [##########] 100% 0.1/0.1 MB
  Unpacking metric-logger
  Scanning metric-logger for unsafe code
  Installing packages: httpx>=0.27
  Installing metric-logger
  Recording metric-logger
  + 熱重載完成
  + 安裝完成：metric-logger (a1b2c3d)
```

第一個 `y` 以上的所有內容，都只讀自 manifest，而且讀的是這次安裝將使用的那一個 commit——這個外掛是什麼、會往你的 venv 裡加哪些套件、它要求在白名單之外匯入哪些模組，以及它有沒有附帶 JavaScript。所以這兩個問題都是在儲存庫還沒下載半個位元組之前就問完的，也都可以回答「不要」。第二個 `y` 之後才是安裝本身，一步一步進行；那些步驟行來自共用的安裝流程，所以它們在[外掛中心](#外掛中心)裡讀起來也完全一樣。

- **沒有終端機時**（腳本、CI、以管線輸入的安裝）答案一律是**否**，訊息會指出 `--accept-capabilities`——它會直接授予 manifest 宣告的那一組而不詢問。`-y` / `--no-confirm` **不會**連帶授權：那個旗標跳過的是「要從這個 URL 安裝嗎？」，而同意一段會連線網路的程式碼是另一個問題。
- **授權內容會被記錄**在 `<USER_DATA>/plugins/installed.json`，並由 `cdui plugin list` 與 `cdui plugin info` 顯示。
- **`cdui plugin update` 不會重複詢問**——只要新版要求的是你已授權範圍的子集；一旦它多要了一項能力就會**停下來**，而這正是更新流程真正攔得到的供應鏈風險形狀。

### 每一級都成立的規則

`torch.load(...)` 仍然必須明確寫出 `weights_only=True`；dunder 存取（`__class__`、`__globals__`、`__subclasses__`……）、frame 走訪（`f_globals`、`gi_frame`……），以及**內建函式** `eval` / `exec` / `compile` / `__import__`——不論是裸呼叫或透過 `builtins` 模組——不論宣告了什麼都一律拒絕。**能力永遠買不到反射能力。**

攔的是那些內建函式，不是那個字：只是剛好同名的**方法**屬於一般程式碼，在每一級都會通過，所以 `torch.compile(model)` 與 `model.eval()` 對外掛而言是允許的。這是刻意的——拒絕它們一直是個長年的誤判——也正是為什麼規則問的是「這個 `eval` 是誰的」，而不是去比對這個字。

但這不代表能力永遠買不到執行程式的權力。`os.system(...)` 與 `os.popen(...)` 只在**以呼叫形式出現時**被拒絕——所以 `f = os.system` 之後再 `f(cmd)` 就繞過了這條規則——而一旦授予 `process-env`，`os.spawnve` / `os.execv` / `os.startfile` 根本不會被拒絕。這與上方 `process-env` 那一列所述是同一件事；之所以在這裡重講一次，是因為這一段先前的版本宣稱了相反的事。

### 預設關閉、第 2 級會解除的屬性名稱

跟上面每一條規則不同——上面那些不論宣告了什麼都一律拒絕，在任何層級都沒有例外——這裡是另一份固定的屬性名稱清單，攔的是第 0 級函式庫自己的東西，在第 0 級與第 1 級被拒絕，到第 2 級則會解除。`numpy.zeros(3).dump(path)` 會把資料原封不動 pickle 到任何路徑，內容還大半是攻擊者可控的；`torch.hub.load(...)` 會下載並執行遠端的 `hubconf.py`；`.savetxt`、`.tofile`、`.load_state_dict_from_url`、`.tensorboard` 以及其他十幾個都是同樣的形狀——它們是 Tier-0 import 回傳值上的**方法**，不是它自己的 import，所以能力閘門（只看得懂 `import` 敘述）根本看不到它們。沒有任何能力解除得了這些東西——它們所在的模組本來就屬於第 0 級，所以任何能力光是點名它都不會多給什麼——和[畫布內腳本政策](/advanced/python-script-node)本來就有的那份清單相同。

這條規則不看接收者是誰，所以是雙向的：外掛**自己的**方法只要剛好同名，一樣會被擋下——你自己類別上的 `self.save(...)`，會被擋得跟 `numpy.array(...).save(...)`一模一樣，這正是腳本政策早就加諸在腳本自己的 `obj.save()` 上的同一種代價。單獨在第 0 級或第 1 級底下，這代表一個類別完全不能定義名叫 `save`、`dump`、`hub`，或清單上其他任何一個名字的方法。

**`--trust-author` 會把整份清單解除。** 一旦外掛以 `--trust-author` 加上 `[security] allowed_modules` 安裝，`.dump` / `.hub` / `.save` 以及清單上其他項目，就又變回普通的屬性名稱——一個已經被信任可以用 `subprocess`、`ctypes` 的外掛，再多攔一個 `arr.dump()` 什麼也保護不到，而且不解除的話，根本不可能寫出一個帶有 `save` 方法的外掛。這跟[每一級都成立的規則](#每一級都成立的規則)裡的每一條都不同——那些不論在哪一級都毫無例外地拒絕：它們攔的是**反射能力**，沒有任何能力或信任層級買得到；而 `.dump` 與 `.hub` 是檔案寫入與遠端程式碼抓取，`--trust-author` 早就用更短的路徑，給了等同或更大的授權。

### 請附原始碼，不要附位元組碼

外掛壓縮檔裡只要是 Python 匯入系統載入得了的檔案，都必須是可讀的**原始碼**。
安裝時掃描的是整個目錄（不只 `nodes/`），且枚舉依據是載入器接受的副檔名
（`importlib.machinery.all_suffixes()`）而不是 `*.py`：`.py` 與 `.pyw` 會被掃描，
`.pyc`、`.pyo`、`.pyd`、`.so`、`.dylib` 則在安裝時**點名拒絕**，而且不論你從哪個平台安裝都一樣。

拒絕是誠實的答案，不是偏好：`.pyc` 要反組譯才能掃，編譯好的擴充模組則原則上就不可能做 AST 掃描。
否則就是匯入一份閘門從未打開過的程式碼——而這正是以前發生的事：`nodes/` 裡只有 `helper.pyc`
而沒有 `helper.py` 的套件，會在伺服器啟動時被匯入，以完全信任執行，沒有宣告任何能力，
也不需要 `--trust-author`，而且從頭到尾沒被看過一眼。

編譯快取不受影響。CPython 寫的快取是 `__pycache__/<name>.cpython-311.pyc`，
它的主檔名不是合法識別字，任何 `import` 語句都叫不出它，所以會被跳過；
攻擊者放的 `__pycache__/payload.pyc` **叫得出來**，因此會被拒絕。

### 這不是什麼

**這是防護欄，不是沙箱**——與[畫布內腳本政策](/advanced/python-script-node)同一套說法，而且在這裡更值得重講一次，因為這是**別人的**程式碼真正執行的地方。

- **閘門讀的是你外掛自己的 `import` 敘述。** 它不會去讀你 import 的函式庫，也無法判斷一個被允許的函式實際上做了什麼。
- **能力攔的是 *import*，不是 *行為*。** 有兩個後果值得直接講明，而不是留給你自己踩到：
  - **`filesystem` 並不會攔截寫檔。** `open(p, "w")` 是內建函式、不需要 import，在第 0 級什麼都不宣告就能過。我們考慮過攔它然後放棄了：模式字串經常是算出來的（`open(p, "w" if overwrite else "r")`），所以這個檢查只要一個變數就能繞過，卻會誤傷誠實的外掛——是一個沒有相應安全價值的誤判。
  - **`network` 隱含了寫檔能力**，透過 `urllib.request.urlretrieve(url, dest)`。
- **能力涵蓋的是黑名單上的模組根名稱，不是整個類別。** `requests` 有被攔；`httpx` 從來就不在黑名單上，所以一個 import 它的外掛什麼都不用宣告就能連網。把 PyPI 上每一個 HTTP 用戶端都列進清單，不是清單做得到的事。
- **「沒有任何能力會交出一個本身就是用來執行程式碼的模組」講的是能力**對照表**，不是對「被授權的外掛能碰到什麼」的保證。** 標準函式庫的模組彼此 import，並把結果留成一般屬性，所以 `import shutil`（屬於 `filesystem`）之後，`shutil.sys.modules['subprocess'].run(...)` 就只差一行。閘門攔的是它認得的模組名稱**作為 import 出現時**；它不會去走訪自己放行進來的物件圖。請注意這並沒有提高任何權限——同樣這一行在本功能之前的任何 CodefyUI 上、什麼都不宣告就能執行，因為 `shutil` 在那裡一樣可以 import。這是閘門的極限，不是分級制度帶來的代價。
- **有兩條路徑刻意完全跳過閘門。** 內建外掛包隨這個 repo 一起發行、由 PR 審查；`cdui plugin link` 載入的是**你自己的**工作目錄，並且會印出警告說明。`cdui project restore` 也會以非互動方式授予專案 manifest 宣告的能力——它本來就帶著 `--trust-author`，所以不會增加額外曝險，但這代表一份專案檔本身也是一個信任決定。
- **任何能寫入 `installed.json` 的東西，都能預先批准下一次更新。** lockfile 正是 `cdui plugin update` 用來「這個能力已授權過、不必再問」的依據，所以能編輯它的程式碼（包含已取得 `filesystem` 的外掛，或任何用 `open` 的外掛）都可以在自己的條目裡加上一項能力，讓下一次更新靜默接受。這屬於入侵後的持久化，而非第一步的提權——但 lockfile 是一份信任存放區，它的保護程度就等於你帳號的保護程度。
- **宣告是作者的意圖聲明。** 它提高了順手攻擊的成本，也讓你在同意前有東西可讀。真正該問的問題仍然是：「我信任寫這個東西的人嗎？」

### 從舊版升級

不需要做任何事。在能力機制出現之前寫入的 lockfile 條目沒有 `capabilities` 欄位，讀起來就是「未授權任何能力」——與它原本的行為完全一致。既有的外掛包重新驗證後行為不變。

## 外掛中心

**在應用程式裡安裝外掛，跑的就是終端機那一套安裝。** `cdui plugin install` 與外掛中心是同一個函式（在 `backend/app/core/plugins/`）的兩個前端，所以一次安裝的順序、什麼算失敗、失敗又叫什麼名字，都只決定一次——主控台與面板不可能在這三件事上互相矛盾。下面這些端點就是它今天的全部：`cdui plugin install` 是目前在用它們的用戶端，而編輯器裡的面板會隨著外掛中心 UI 一起到來。

**它是一場有兩個回合的對話。** `POST /api/plugins/inspect` 讀取一個來源——型錄名稱、`owner/repo`，或一個網址——並回答一個人要做決定所需要的一切：這個外掛是什麼、它會往你的 Python 環境加什麼、它宣告了哪些能力、它要求對哪些模組關閉安全掃描、它有沒有附帶會在你的編輯器裡執行的 JavaScript，以及你是不是已經裝了它。過程中不下載任何東西，也不安裝任何東西；manifest 只在「單一個」已解析的 commit 上讀取，答案則存放在一個 `inspection_id` 底下。接著 `POST /api/plugins/install` 依 id 安裝「那一份」檢查結果，並帶上只有人能給的答案（`accept_capabilities`、`trust_author`，以及用來蓋過既有安裝的 `force`）。伺服器永遠不會從請求內容裡拿 manifest、commit 或能力清單——所以你同意的那份 manifest 就是被安裝的那份，而一個在兩個回合之間多要了一項能力、換了 id 或往白名單加了模組的 tarball，會被拒絕而不是被安裝。安裝以工作（job）的形式執行：`202` 加一個 `job_id`，接著 `GET /api/plugins/jobs/{job_id}/events` 會從指定 cursor 重播這個工作的記錄並長輪詢後續，`POST /api/plugins/jobs/{job_id}/cancel` 則會把它停下來，而且乾淨到不會留下任何寫到一半的東西。

**確認畫面就是那三個層級，一一對應。**[第 0 級](#安全性三個層級)沒有東西要問。第 1 級就是檢查結果裡的 `capabilities`——一項一列——第 2 級則是它的 `allowed_modules`，那是一個關於**作者**的獨立決定，以 `trust_author` 傳遞。兩者都不是沙箱：能力一旦授權，外掛就能匯入那一組模組，CodefyUI 不會再逐一攔截；這也正是為什麼授權之前該讀的是[這不是什麼](#這不是什麼)。

**安裝只能從本機操作。** 每一個會安裝、檢查或移除的端點——inspect、install、cancel、update、delete——都同時需要工作階段 token **以及**伺服器綁定在回送（loopback）位址：安裝外掛等於把別人的程式碼放到這個行程即將匯入的地方，檢查來源等於憑呼叫端一句話去連 GitHub，而刪除則是把別人的外掛拿走。刻意對區網提供服務的教室或實驗室環境，可用 `CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1` 重新開放。reload 與 enable／disable 需要 token 但不受回送位址限制——它們處理的是這台機器已經有、而且你已經同意過的程式碼；讀取則一律開放，包括某個工作的事件記錄，那正是一個在安裝途中才打開的第二個分頁用來跟進度的東西。

**步驟與失敗訊息是英文的。**`Resolving …`、`Downloading …`、`Unpacking …`、`Scanning … for unsafe code`、`Installing packages: …`、`Installing …`、`Recording …`，以及一次被拒絕或失敗的安裝所帶的每一句話，都出自共用的安裝流程；它只有一套用字，而不是每個前端各一套。它們周圍的介面有翻譯，這些沒有。

**一次安裝可能以 `needs_restart` 收尾，那不是失敗。** 外掛的 `[python_deps]` 以「只增不改」的方式安裝，並套用一份把執行中伺服器已載入的每個套件都釘住的 constraints 檔，所以外掛要什麼都不可能降級你這個工作階段正握著的東西。當解析器說這件事沒辦法在線上完成時，這個工作會以 `needs_restart` 結束，並附上要在伺服器停掉後執行的那一行 `command`。這個外掛本身沒有任何問題，而且對同一台執行中的伺服器再問一次也只會得到同樣的結果。（`cdui plugin install` 同樣會印出那行指令，並以 `3` 離開。）

**移除只會刪掉這次安裝下載過的東西，不會多刪。**`DELETE /api/plugins/{id}` 會刪掉下載型外掛包的目錄；內建外掛包的檔案會保留——那是發行版的一部分——並且會被記住已移除，所以 `cdui plugin sync` 不會再理它，直到你再次以名稱安裝為止；而你用 `cdui plugin link` 連結的目錄，會原封不動留在作者放它的地方。它絕對不會移除的，是這個外掛的 Python 套件：從「已經匯入這些套件的那個行程」裡面反安裝它們，正是讓直譯器載入到一半還在服務請求的方法。所以回應會把這件事說出來——`python_deps_left` 列出它們，`uninstall_command` 則是伺服器停掉之後可以執行的那一行：

```bash
uv pip uninstall --python <CodefyUI venv 的 python> httpx
```

如果目錄刪不掉——在 Windows 上最常見的原因是有東西正開著其中的檔案——那就什麼都不會被移除：lockfile 條目留著，外掛仍然是已安裝狀態，而回應是 `409` `files_locked`，帶著作業系統自己的那句話，以及還留在那裡的目錄。關掉正在使用它的東西，或把伺服器停掉，然後再移除一次。

**更新會去問外掛自己的儲存庫，並以三種方式之一回答。**`POST /api/plugins/{id}/update` 會在該儲存庫目前的 commit 上重新讀一次 manifest。`200` `{"status": "up_to_date", "sha": …}`——你手上的 commit 就是那邊的 commit。`202` 加一個 `job_id`——新版本要的東西你都已經授權過了，所以它已經在安裝了。`200` `{"status": "needs_consent", …}`——它多要了東西，回應內容會帶著 `/inspect` 回傳的同一份檢查結果，外加 `capabilities_added` 與 `allowed_modules_added`，那就是一次更新的確認畫面的全部內容。要完成這一種，把那份檢查結果以 `inspection_id`、`accept_capabilities` 與 `trust_author` 送回 `POST /api/plugins/install` 即可——而且不用帶 `force`：伺服器已經記下這份檢查結果來自更新按鈕，不會再要求你對「你自己要求換掉的外掛」說一次「對，換掉它」。

有兩件事更新不會做。它絕不會裝上一個**不同的**外掛：如果某個儲存庫的 manifest 現在宣告的是另一個 id，它會以 `400` `not_updatable` 被拒絕而不是被抓下來——因為更新 `metric-logger` 卻拿到 `metric-logger-ng`（甚至可能蓋掉你原本就有的同名外掛），那不叫更新。如果你要的就是那個改過名字的儲存庫，請把它當成一個新外掛安裝。另外，如果你先前把這個外掛停用了，更新之後它仍然是停用的：重新啟用是一個決定，而「從同一個儲存庫重裝同一個外掛」不是做這個決定的地方。

內建外掛包與連結的目錄同樣回答 `400` `not_updatable`，並附上該怎麼做的提示：隨這個版本發行的外掛包要用 `cdui update` 更新，而連結的目錄本來就是作者磁碟上此刻的樣子。

**同一時間只有一次安裝，而且跨兩個中心都算。** 在已經有一個安裝在跑的時候再啟動一次安裝——或一次更新，那也是安裝——會得到 `409` 與可以跟進的 `job_id`；在某個外掛自己的安裝還在進行時去移除、啟用或停用**那個**外掛，也是一樣：lockfile 條目在安裝途中被改寫，正是一個外掛最後留在磁碟上卻沒有任何東西指向它的原因。別的外掛的安裝則不會擋住這三件事，因為兩個外掛是兩個目錄、兩把 lockfile 鑰匙。

## 撰寫你自己的外掛

最快的起手式是 **`cdui plugin new`**，一個指令就能產生可直接編輯的外掛骨架：

```bash
cdui plugin new my-plugin          # 純後端骨架
cdui plugin new my-plugin --ui     # 另含一個接好 SDK 的 React 前端
```

它會產生 manifest、一個範例節點、一個測試（內含 `cdui_plugins.<id>` 命名空間 shim，讓本地 `pytest` 可直接執行），並在加上 `--ui` 時產生一個 Vite + React 的 `ui/`，其 `src/sdk/` 即為型別化的外掛 SDK。外掛會建立在 `./my-plugin/`；用下方的 `cdui plugin dev` 連結後即可開始編輯。

若需更完整的參考，可 Fork **[官方外掛模板](https://github.com/CodefyUI/CodefyUI-Plugin-Official)**——一個可運作、採 MIT 授權的外掛，包含兩個範例節點、一張範例圖、一套測試，以及一份完整註解的資訊清單 (manifest)。它的 README 逐欄解說每個欄位與 AST 安全閘門。它在目錄中的名稱是 `official-template`，可以直接用名稱安裝。

```bash
# Install the template itself to see the pattern live
cdui plugin install official-template

# After forking — any repository, by owner/repo or by URL
cdui plugin install your-username/your-fork
```

一個外掛包可隨附下列任意內容：一個 `nodes/` 目錄（自動探索）、一個 `presets/` 目錄、一個 `examples/` 目錄，以及一個 `assets/` 目錄（於 `/plugins/<id>/assets/<file>` 提供）。一份 `cdui.plugin.toml` 資訊清單宣告 id、版本、`requires_codefyui`、內容目錄、課程 metadata，以及——只在你需要時——[安全性](#安全性三個層級)一節描述的 `[security]` 宣告。若你的節點只做純運算，直接刪掉那一段就好；大多數外掛都是如此。

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

連結外掛的 `[python_deps]` 會依照與下載型外掛包相同的規則安裝：只增不改，並套用那份把執行中伺服器已載入的每個套件都釘住的 constraints 檔。因此 `cdui plugin link` 也帶有安裝路徑的離開碼——`3` 表示它要求的某個套件無法裝進執行中的伺服器（會印出停掉伺服器後要執行的指令），`130` 表示 `Ctrl+C`——而不再是把套件管理程式的原始離開碼直接丟回來。

:::tip 開發資料隔離
透過 `scripts/dev.py` 執行外掛指令——或設定 `CODEFYUI_USER_DATA_DIR`——可讓某個 clone 的 lockfile 留在 repo 內（`.codefyui_dev/`），而非全機共用的 user-data 目錄，避免多個 clone 互相覆蓋。
:::

## REST API

| 端點 | 方法 | 驗證 | 說明 |
|----------|--------|------|-------------|
| `/api/plugins` | GET | 開放 | 列出每一個已安裝的外掛包，不論啟用與否。 |
| `/api/plugins/catalog` | GET | 開放 | 把「這個版本可用名稱安裝的東西」與「你已經安裝的東西」合併——每個外掛一列，每列都說明它處於哪個狀態。 |
| `/api/plugins/generation` | GET | 開放 | 編輯器輪詢用的重載計數器，用來得知節點面板變了。 |
| `/api/plugins/{id}` | GET | 開放 | 取得某外掛的資訊清單 (manifest)、節點與 README。 |
| `/api/plugins/jobs/{job_id}/events` | GET | 開放 | 某個安裝工作在 `?cursor=` 之後的記錄與進度；加上 `?wait=` 可長輪詢後續。 |
| `/api/plugins/reload` | POST | token | 重新探索節點、預設模組與外掛包。 |
| `/api/plugins/{id}/enable` | POST | token | 啟用一個已安裝的外掛。 |
| `/api/plugins/{id}/disable` | POST | token | 停用它，但不移除。 |
| `/api/plugins/inspect` | POST | token + 回送位址 | 在單一 commit 上讀取一個來源，並說明安裝它的代價。不會安裝任何東西。 |
| `/api/plugins/install` | POST | token + 回送位址 | 安裝某次檢查所描述的內容——回傳 `202` 與 `job_id`。 |
| `/api/plugins/jobs/{job_id}/cancel` | POST | token + 回送位址 | 要求執行中的安裝停下來。 |
| `/api/plugins/{id}/update` | POST | token + 回送位址 | 取得該外掛自己的儲存庫現在有什麼。三種答案：`202` `{job_id}`、`200` `{status: "up_to_date", sha}`，或 `200` `{status: "needs_consent", inspection, capabilities_added, allowed_modules_added}`——最後這一種由用戶端以 `POST /install {inspection_id, accept_capabilities, trust_author}` 完成，不必帶 `force`。 |
| `/api/plugins/{id}` | DELETE | token + 回送位址 | 移除它，並說明這樣做留下了什麼。 |

**開放**是編輯器會輪詢的讀取，和這個 app 裡其他讀取一樣。**token** 是每個會造成變更的呼叫都要帶的工作階段標頭。**token + 回送位址**則是在此之上，只要伺服器沒有綁定在回送（loopback）位址就一律拒絕，除非 `CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1` 另有指示——這條線為什麼畫在這裡，見[外掛中心](#外掛中心)。
