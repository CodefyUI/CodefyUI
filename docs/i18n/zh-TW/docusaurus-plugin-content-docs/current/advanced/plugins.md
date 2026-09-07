---
sidebar_position: 3
title: 外掛
description: 安裝教育節點的外掛包，並學習如何撰寫與發佈你自己的外掛。
---

# 外掛包

教育（「Edu」）節點以可安裝的**外掛包**形式提供，**依方向**組織，因此每一個都對應到一個動手實作的教科書模組，並在你逐步學習時累進安裝。

```bash
cdui plugin sync                           # 安裝所有你還沒決定過的內建外掛包
cdui plugin install foundations deep rl   # 或者一個一個挑
cdui plugin install edu stats              # 實作練習、敘述統計
cdui plugin list
cdui plugin info deep                      # manifest, lessons covered, node names
cdui plugin search attention               # query the catalog
cdui plugin install foo/bar                # third-party pack from GitHub
cdui plugin disable deep                   # 停用，但不刪除檔案
cdui plugin enable deep                    # 重新啟用，不需再下載
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

另外有三個官方外掛放在各自的儲存庫裡。它們一樣可以用型錄列出的名稱安裝，但和上面的外掛包不同：它們是從 GitHub 下載的，所以安裝時會請你確認。

| 外掛 | 這是什麼 | 安裝 |
|------|------------------|-----------|
| `graph-copilot` | AI 對話助手：以對話建立與修改節點圖，執行你核准的隔離實驗與參數搜尋，並保留可攜的實驗紀錄。需要一個 LLM 供應者（Codex、Ollama、OpenAI 或 Anthropic）。 | `cdui plugin install graph-copilot` |
| `self-learning` | 把一個自由描述的機器學習問題變成逐步教材：先由 LLM 建出可運作的圖並驗證能執行，再由外掛擷取每一步的截圖，產出繁體中文 Markdown 教材、可列印頁面、圖表與一份起始練習。 | `cdui plugin install self-learning` |
| `official-template` | 給外掛作者的可運作起始模板：兩個範例節點、一個預設模組、一張範例流程圖、一個資產檔、一套範例測試，以及一個 React 工具面板。安裝後可查看外掛能做什麼，也可 fork 後撰寫自己的外掛。 | `cdui plugin install official-template` |

每個 Edu 節點都把單一課程概念分解成一連串具名步驟，由[教學檢視器](/usage/teaching-inspector)一次渲染一列——`Edu-ColumnStats` 將母體標準差公式呈現為 `sum → divide → deviations² → variance → sqrt`；`Edu-PolicyGradient` 暴露 `softmax → gather → log → baseline → loss`；`Edu-Patchify` 讓 `unfold → permute → flatten` 變得可見。在「設定」popover 中開啟**顯示內部步驟**即可擷取它們。

## 外掛包如何儲存

- **內建方向外掛包**位於 repo 內的 `plugins/<id>/`，並就地啟用（不複製）。
- **第三方外掛包**會以固定 SHA 的 tarball 下載到 `<USER_DATA>/plugins/<id>/`，並在安裝前經過 **AST 驗證**（見[安全性](#安全性三個層級)）。
- `<USER_DATA>/plugins/installed.json` 的 lockfile 會記錄每次安裝及已授予的能力，讓 `cdui start` 能在下次啟動時重新探索外掛。對透過 `cdui` 執行的命令，`<USER_DATA>` 是 `<install dir>/.codefyui_dev/`：除非已匯出 `CODEFYUI_USER_DATA_DIR`，否則 `cdui start`、`cdui dev` 與每個 `cdui plugin` 命令都會將它設為該目錄。因此，預設安裝使用 `~/CodefyUI/.codefyui_dev/plugins/installed.json`。平台 user-data 目錄（`%LOCALAPPDATA%\codefyui`、`~/.local/share/codefyui` 或 `~/Library/Application Support/codefyui`）只適用於直接啟動且未設定 `CODEFYUI_USER_DATA_DIR` 的 `uvicorn app.main:app`。lockfile 也用來判定外掛是否已安裝。如果目錄已被手動刪除，或要取代透過 `cdui plugin link` 連結的目錄，重新安裝時必須使用 `--force`。

外掛節點會加上命名空間，以避免衝突並讓圖能自我說明——內建節點使用像 `Conv2d` 這樣的裸名稱，而外掛節點則會像 `foundations:Edu-KNN` 這樣加上限定。

### 升級後補上新外掛包——`cdui plugin sync`

真正啟用一個外掛包的是 lockfile，而升級並不會寫入它。因此，版本**新增**內建外掛包時，檔案雖然會隨升級寫入磁碟，卻不會載入：節點可安裝，但不會顯示。`cdui plugin sync` 用來補上這些外掛包。它會安裝所有你尚未決定是否安裝的內建外掛包，執行前只確認一次，並逐一回報結果。因此，即使某個外掛包的 `python_deps` 無法透過學校網路下載，也不會影響其他外掛包。

```bash
cdui plugin sync --dry-run   # 只告訴我還有哪些沒裝
cdui plugin sync             # 全部安裝（確認一次）
cdui plugin sync --yes        # 不詢問——腳本、CI、教室映像檔
cdui plugin sync --prune      # 順手清掉已不再發行的外掛 lockfile 項目
```

它刻意不做兩件事。第一，它不會在啟動時自動執行，`cdui update` 也不會詢問是否執行。版本新增的程式碼是否啟用，必須由使用者同意，不能視為一般升級細節。第二，它不會重新安裝你已移除的外掛包。`cdui plugin uninstall` 會在 lockfile 中記錄移除狀態（`plugins` 旁邊的 `removed` 對應表），以區分「從未處理過」和「已主動移除」。因此，`cdui start` 與 `cdui plugin list` 也不會再列出已移除的外掛包。若要取消移除狀態，請按名稱重新安裝：`cdui plugin install stats` 會清除該紀錄，之後 sync 也會重新計入。

## 安全性——三個層級

外掛包是在 CodefyUI 行程內執行的 Python。第三方外掛包安裝前，包內任何位置的每一個 `.py` 檔——`nodes/`、`examples/`、`tests/`、`docs/`、`assets/`，或其他任何子目錄——都會由 AST 閘門走訪，決定它可以 import 什麼。任何目錄都不會因名稱而排除：外掛載入器可以從包內任何位置 import（節點檔裡寫 `from ..tests import helper` 是可行的），所以掃描範圍必須涵蓋載入器能觸及的所有位置。閘門分為三個層級，其中第 1 級需要特別說明。

| 層級 | 外掛如何取得 | 涵蓋範圍 |
|------|----------------------|----------------|
| **0——預設** | 不需宣告 | 純運算：`math`、`statistics`、`collections`、`itertools`、`functools`、`json`、`re`、`dataclasses`、`typing`、`enum`、`decimal`、`random`、`numpy`、`torch`、`pandas`——外加路徑輔助函式（見下）。所有第一方外掛包都在這一級。 |
| **1——宣告能力** | manifest 寫 `[security] capabilities = [...]`，並在安裝時由使用者確認 | 每個能力對應一組具名模組。 |
| **2——信任作者** | `[security] allowed_modules = [...]` **加上** `cdui plugin install --trust-author` | 任何模組，包括 `subprocess`、`ctypes` 與 `importlib`。 |

### 能力清單

| 能力 | 解鎖 | 你正在同意的事 |
|------------|---------|--------------------------|
| `network` | `requests`、`urllib`、`http`、`socket`、`ssl`，以及它們背後的原始 C 模組（`_socket`、`_ssl`） | 這個外掛可以與任何主機收發資料——**並把下載到的內容寫入磁碟**，因為 `urllib.request.urlretrieve(url, dest)` 只要一行。 |
| `filesystem` | `pathlib`、`tempfile`、`shutil`、`zipfile`、`tarfile`、`gzip`、`bz2`、`lzma`、`codecs`、`sqlite3`（含 `_sqlite3`）、`glob`、`fileinput`、`readline` | 這個外掛可以使用檔案**函式庫**。這不是寫入的邊界：單純的 `open(p, "w")` 是內建函式，完全不需要任何宣告（見[這不是什麼](#這不是什麼)）。 |
| `process-env` | `os`、`ntpath`、`posixpath`、`genericpath`、`nt`、`posix` | 這個外掛拿到**整個 `os` 模組**：讀取*並修改*此行程的環境變數（**包含其中的 API 金鑰**）、啟動其他程式（`os.execv`、`os.spawnve`、`os.startfile`），以及刪除或重新命名檔案。這個名字是大家索取它的理由，但授予的範圍比名字大。 |

除此之外都不是能力。`subprocess`、`sys`、`importlib`、`ctypes`、`pickle`、`marshal`、`dill`、`shelve`、`runpy`、`code`、`signal`、`atexit`、`webbrowser`、`threading`、`asyncio`、`multiprocessing` 一律只能使用第 2 級：**沒有任何能力會直接允許專門用來執行程式碼或存取直譯器的模組。** 這項說明只針對能力對照表：`process-env` 會授予 `os`，而 `os` 可以啟動行程；但它不會直接授予專門執行程式碼的模組。

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

這些行會被拒絕，因為 `os.path` 是真正的模組，而且大部分功能並非字串處理：

- `os.path` **就是** `ntpath` / `posixpath`，這兩個模組在模組層級執行 `import os` 與 `import sys`，並把兩者都留成一般屬性——所以 `path.os.remove(p)` 會刪掉檔案，`path.sys.modules['subprocess'].run([...])` 會執行指令。
- `os` 本身**就是** `nt`（Windows）或 `posix`（POSIX）——CPython 自己的 `os.py` 執行 `from nt import *` / `from posix import *`，`os.remove`、`os.environ`、`os.system` 都是從這裡來的。直接以名稱 import 這個原始模組，中間沒有任何攔截，會取得相同介面。
- `expandvars("%WANDB_API_KEY%")` 會回傳該環境變數的值——正是 `process-env` 要限制的內容——而 `expanduser("~")` 會回傳你的家目錄。
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

第一個 `y` 以上的所有內容都只讀自 manifest，而且讀的是這次安裝將使用的那一個 commit：外掛的用途、會加入 venv 的套件、要求在白名單之外 import 的模組，以及是否隨附 JavaScript。因此，這兩個問題都會在下載 repository 之前詢問，也都可以回答 `no`。第二個 `y` 之後才會逐步執行安裝；那些步驟行來自共用的安裝流程，所以它們在[外掛中心](#plugin-center)也完全相同。

- **沒有終端機時**（腳本、CI、以管線輸入的安裝）答案一律是**否**，訊息會指出 `--accept-capabilities`——它會直接授予 manifest 宣告的那一組而不詢問。`-y` / `--no-confirm` **不會**連帶授權：那個旗標跳過的是「要從這個 URL 安裝嗎？」，而同意一段會連線網路的程式碼是另一個問題。
- **授權內容會被記錄**在 `<USER_DATA>/plugins/installed.json`，並由 `cdui plugin list` 與 `cdui plugin info` 顯示。
- **`cdui plugin update` 不會重複詢問**——只要新版要求的是你已授權範圍的子集；一旦它多要了一項能力就會**停下來**，而這正是更新流程真正攔得到的供應鏈風險形狀。

### 每一級都成立的規則

`torch.load(...)` 仍然必須明確寫出 `weights_only=True`；dunder 存取（`__class__`、`__globals__`、`__subclasses__`……）、frame 走訪（`f_globals`、`gi_frame`……），以及**內建函式** `eval` / `exec` / `compile` / `__import__`——不論是裸呼叫或透過 `builtins` 模組——不論宣告了什麼都一律拒絕。**任何能力都不會允許反射。**

攔的是那些內建函式，不是那個字：只是剛好同名的**方法**屬於一般程式碼，在每一級都會通過，所以 `torch.compile(model)` 與 `model.eval()` 對外掛而言是允許的。這是刻意的——拒絕它們一直是個長年的誤判——也正是為什麼規則問的是「這個 `eval` 是誰的」，而不是去比對這個字。

但這不代表能力絕不會允許執行其他程式。`os.system(...)` 與 `os.popen(...)` 只在**以呼叫形式出現時**被拒絕——所以 `f = os.system` 之後再 `f(cmd)` 就能繞過這條規則——而授予 `process-env` 後，`os.spawnve` / `os.execv` / `os.startfile` 都不會被拒絕。這與上方 `process-env` 那一列所述是同一件事；此處再次說明，是因為這一段先前的版本宣稱了相反的事。

### 預設關閉、第 2 級會解除的屬性名稱

除了上述在所有層級都成立的規則外，還有一份固定的屬性名稱清單：第 0 級與第 1 級會拒絕這些 Tier-0 函式庫屬性，第 2 級則解除限制。`numpy.zeros(3).dump(path)` 可把大部分內容由攻擊者控制的 pickle 寫入任意路徑；`torch.hub.load(...)` 會下載並執行遠端 `hubconf.py`；`.savetxt`、`.tofile`、`.load_state_dict_from_url`、`.tensorboard` 與其他十多個項目都有相同風險。它們是 Tier-0 import 所回傳值上的**方法**，不是獨立的 import，因此只檢查 `import` 敘述的能力閘門不會發現。任何能力都無法解除這項限制，因為這些方法所在的模組已經屬於第 0 級，點名能力不會授予新模組。這與[畫布內腳本政策](/advanced/python-script-node)使用的是同一份清單。

這條規則不判斷接收者，因此外掛**自己的**方法只要同名也會被拒絕：自訂類別上的 `self.save(...)` 與 `numpy.array(...).save(...)` 受到相同限制，這也和腳本政策對腳本自身 `obj.save()` 採取的限制一致。只使用第 0 級或第 1 級時，類別完全不能定義名為 `save`、`dump`、`hub` 或清單中其他名稱的方法。

**`--trust-author` 會解除整份清單的限制。** 外掛以 `--trust-author` 加上 `[security] allowed_modules` 安裝後，`.dump` / `.hub` / `.save` 與其餘項目都恢復為一般屬性名稱。已信任可使用 `subprocess` 與 `ctypes` 的外掛，再限制 `arr.dump()` 並無額外保護，而且會使外掛無法定義名為 `save` 的方法。這與[每一級都成立的規則](#每一級都成立的規則)不同：那些規則拒絕的是**反射**，任何能力或信任層級都不會允許；`.dump` 與 `.hub` 涉及檔案寫入與遠端程式碼下載，而 `--trust-author` 已經授予同等或更高的權限。

### 請附原始碼，不要附位元組碼

外掛 tarball 中任何可由 Python import 系統載入的檔案，都必須是可讀的**原始碼**。安裝程式會掃描整個目錄，不只 `nodes/`，並依載入器接受的副檔名（`importlib.machinery.all_suffixes()`）枚舉，而不是只找 `*.py`：`.py` 與 `.pyw` 會被掃描；`.pyc`、`.pyo`、`.pyd`、`.so`、`.dylib` 則會在安裝時依名稱**拒絕**，且不受安裝來源平台影響。

這項拒絕反映實際掃描限制，而不是政策偏好。`.pyc` 必須反編譯才能掃描，編譯好的擴充模組則無法進行 AST 掃描。若允許這些檔案，就會 import 未經閘門檢查的程式碼。先前的行為正是如此：若外掛包的 `nodes/` 只有 `helper.pyc` 而沒有 `helper.py`，伺服器啟動時仍會以完整權限 import；不必宣告能力、不必使用 `--trust-author`，也不會先經過掃描。

編譯快取不受影響。CPython 產生的快取路徑為 `__pycache__/<name>.cpython-311.pyc`，其檔名 stem 不是合法識別字，無法由 `import` 敘述指名，因此會被略過。攻擊者提供的 `__pycache__/payload.pyc` 可以被指名，所以會被拒絕。

### 這不是什麼

**這是防護欄，不是沙箱。**[畫布內腳本政策](/advanced/python-script-node)也採用相同說明；此處必須重申，因為外掛會執行**第三方**程式碼。

- **閘門讀的是你外掛自己的 `import` 敘述。** 它不會去讀你 import 的函式庫，也無法判斷一個被允許的函式實際上做了什麼。
- **能力限制的是 *import*，不是 *行為*。** 必須明確說明兩個後果：
  - **`filesystem` 並不限制寫入檔案。** `open(p, "w")` 是內建函式，不需 import，在第 0 級且未宣告任何能力時也能通過。專案曾考慮限制它，但模式字串經常由程式計算（`open(p, "w" if overwrite else "r")`），一個變數就能避開檢查，卻會拒絕正常外掛，因此沒有相應的安全價值。
  - **`network` 隱含了寫檔能力**，透過 `urllib.request.urlretrieve(url, dest)`。
- **能力涵蓋的是黑名單上的模組根名稱，不是整個類別。** `requests` 有被攔；`httpx` 從來就不在黑名單上，所以一個 import 它的外掛不需宣告任何能力就能連網。清單無法列舉 PyPI 上的每一個 HTTP 用戶端。
- **「沒有任何能力會交出一個本身就是用來執行程式碼的模組」只描述能力對照表，不保證已授權的外掛能存取哪些物件。** 標準函式庫的模組會彼此 import，並把結果保留為一般屬性；因此授予 `filesystem` 並執行 `import shutil` 後，外掛即可透過 `shutil.sys.modules['subprocess'].run(...)` 執行程式。閘門只會拒絕以匯入敘述出現的已知模組名稱，不會走訪已允許物件的物件圖。這不會提高既有權限：在加入本功能之前，任何 CodefyUI 外掛都能在未宣告能力時執行同一行，因為當時也允許 import `shutil`。這是閘門的限制，不是分級制度新增的風險。
- **有兩條路徑刻意完全跳過閘門。** 內建外掛包隨這個 repo 一起發行、由 PR 審查；`cdui plugin link` 載入的是**你自己的**工作目錄，並且會印出警告說明。`cdui project restore` 也會以非互動方式授予專案 manifest 宣告的能力——它本來就帶著 `--trust-author`，所以不會增加額外曝險，但這代表一份專案檔本身也是一個信任決定。
- **任何可寫入 `installed.json` 的程式碼，都能預先核准下一次更新。** lockfile 是 `cdui plugin update` 判斷能力已授權、不必再次詢問的依據。因此，可編輯 lockfile 的程式碼（包含已取得 `filesystem` 的外掛，或任何使用 `open` 的外掛）都能在自己的條目加入能力，讓下一次更新直接接受。這屬於入侵後的持久化，而非第一步提權；但 lockfile 是信任存放區，其保護程度取決於使用者帳號的保護程度。
- **宣告是作者的意圖聲明。** 它會提高快速攻擊的成本，也提供安裝前可檢查的資訊。真正需要判斷的是：「你是否信任作者？」

### 從舊版升級

不需要做任何事。在能力機制出現之前寫入的 lockfile 條目沒有 `capabilities` 欄位，讀起來就是「未授權任何能力」——與它原本的行為完全一致。既有的外掛包重新驗證後行為不變。

## 外掛中心 {/* #plugin-center */}

**外掛中心與 `cdui plugin install` 使用相同的後端實作。** 因此，兩者使用相同的安裝步驟、失敗判定與錯誤代碼。

### 使用外掛中心 {/* #using-the-plugin-center */}

**開啟外掛中心。** 使用側邊欄**自訂與外掛**分頁之**外掛**區段的**外掛中心...**按鈕，或**設定 → 外掛 → 外掛中心**中的**開啟**按鈕。「設定」列包含「安裝教學節點套件與 GitHub 上的外掛。」文字，以及「*已安裝 N 個，可安裝 M 個*」摘要。套件中心與外掛中心可以同時開啟；按 **Escape** 會關閉最上層的視窗。

**外掛清單。** 清單包含型錄中可按名稱安裝的所有內建與官方外掛，以及所有已安裝的外掛。篩選條件為**全部**、**已安裝**與**可安裝**。每張卡片會顯示名稱、狀態、版本、來源、repository 與 pin（`ref @ sha`）、章節、節點數量，以及 Python 依賴套件。來源為**內建**、**官方**，或透過 `cdui plugin link` 註冊之目錄使用的**本機連結**；其他第三方 repository 不顯示來源標籤。可用操作取決於[安裝狀態](#install-states)。

**安裝型錄以外的外掛。** 在**從 GitHub 安裝**中輸入 `owner/repo`、`owner/repo@ref`、GitHub URL 或型錄名稱，再選取**檢視**。其他格式會在送出前遭到拒絕，並顯示「請輸入內建套件名稱、owner/repo[@ref] 或 GitHub URL。」

**檢視與同意。** 選取卡片上的**安裝**或輸入欄位中的**檢視**，會讀取單一個已解析 commit 上的 manifest。需要同意時，清單頂端會顯示**安裝前請確認**卡片。卡片包含名稱、版本、說明、作者、新註冊的節點、Python 依賴套件、commit pin，以及 manifest 有提供時的 HTTP 或 HTTPS **首頁**連結。每個必要決定都有 checkbox。**這個外掛要求：**會列出每項宣告的能力及其存取範圍；**同意授予這些能力**會記錄[第 1 級](#安全性三個層級)同意。**我信任這位作者。允許使用：...**會列出 `allowed_modules`，並記錄[第 2 級](#安全性三個層級)同意。隨附瀏覽器程式碼時，卡片會加上「包含會在編輯器中以完整權限執行的 JavaScript。」警告。所有必要的 checkbox 勾選前，**安裝**會保持停用。內建外掛包不需要同意，可直接從自己的資料列安裝。已安裝的外掛會顯示取代警告與**重新安裝**按鈕。

**安裝進度。** 右側面板會顯示目前步驟、進度條、最近的 log，以及**取消安裝**。步驟可能包括*正在解析來源*、*正在下載*、*正在解壓縮*、*正在檢查程式碼*、有 `[python_deps]` 時的 pip 步驟、*正在複製檔案*、*正在寫入安裝紀錄*與*正在載入節點*。最終狀態可能是*已安裝*、*已更新*、附伺服器提示的*失敗*、*已取消安裝*、*needs_restart* 或*與伺服器失去聯繫。請重新整理以確認外掛狀態。* `needs_restart` 表示未安裝任何外掛檔案；面板會顯示要在伺服器停止後執行的 `uv pip install` 指令。詳見[安裝如何進行](#how-an-install-runs)。失去聯繫表示瀏覽器與伺服器中斷連線；請重新整理以取得目前狀態。關閉外掛中心不會取消 job，其他分頁也能繼續追蹤。型錄外掛安裝失敗後，面板也會顯示 `cdui plugin install <repo>[@ref]`，以便在終端機使用相同流程並查看完整 log。

**套用變更。** 安裝、更新、啟用、停用或解除安裝後，面板會重新載入型錄、節點定義與外掛 UI。節點面板與外掛面板不需重新載入頁面即可反映新狀態。

**啟用、停用、更新、解除安裝。** **停用**會從節點面板移除已安裝外掛的節點，並停止提供其 bundle 與 assets，但不刪除檔案。**啟用**會重新啟用外掛，不需再次下載。終端機中的 `cdui plugin enable|disable <id>` 提供相同操作。**更新**只適用於從 GitHub 安裝的外掛；內建外掛包透過 `cdui update` 更新，本機連結的目錄則使用目前檔案。**解除安裝**會先詢問：「要解除安裝「*名稱*」嗎？使用它節點的圖將無法執行；它安裝的 Python 套件會保留。」本機連結的目錄只提供**啟用**與**停用**，因為 `cdui plugin link` 管理其註冊。

**從其他電腦操作。** 伺服器未綁定至回送位址時，footer 會顯示「只能在執行伺服器的那台電腦上安裝。」**檢視**、**安裝**、**更新**與**解除安裝**會停用，並顯示相同 tooltip；**啟用**與**停用**仍可使用。設定 `CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1` 可允許遠端外掛操作。受影響的 routes 請見[安裝如何進行](#how-an-install-runs)。比這個面板舊的伺服器會顯示「這台伺服器不支援外掛中心。請更新 CodefyUI 後重新啟動。」

### 安裝狀態 {/* #install-states */}

`GET /api/plugins/catalog` 會為每個外掛指定六種狀態之一。狀態會決定狀態 pill 與按鈕。狀態優先順序為執行中的 job、lockfile 條目，以及沒有 lockfile 條目時的 `removed` 紀錄。

| 狀態 | Pill | 意義 | 按鈕 |
|-------|------|---------|---------|
| `available` | 未安裝 | 存在於型錄中，沒有 lockfile 條目。 | 安裝 |
| `removed` | 已移除 | 沒有 lockfile 條目，但解除安裝留下了 `removed` 紀錄，因此 `cdui plugin sync` 不會重新加入。計入**可安裝**。 | 安裝（會清除紀錄） |
| `installing` | 安裝中 | 這個外掛的 job 正在執行。計入**已安裝**。 | 無 |
| `installed` | 已安裝 | 有 lockfile 條目、磁碟上有檔案，而且已啟用。 | 停用、更新（僅限 GitHub 安裝）、解除安裝 |
| `disabled` | 停用 | 有 lockfile 條目、磁碟上有檔案，但已關閉：節點未註冊、bundle 與 assets 不會提供。 | 啟用、解除安裝 |
| `missing_files` | 檔案遺失 | 有 lockfile 條目，但其目錄不存在，例如移動 checkout 或中斷解除安裝後。計入**已安裝**。 | 安裝；由於 lockfile 條目仍在，伺服器會回覆 `409` `already_installed`，檢視卡片接著提供**重新安裝**。解除安裝會移除該條目。 |

本機連結的目錄（`source_kind` 為 `local`）在每種狀態下都只提供**啟用**與**停用**。側邊欄與「設定」中的計數只包含 `installed` 與 `disabled` 外掛。

### 安裝如何進行 {/* #how-an-install-runs */}

**檢查與安裝是分開的 request。** `POST /api/plugins/inspect` 會將型錄名稱、`owner/repo` 或 URL 解析至單一 commit，並回傳外掛說明、Python 依賴套件、宣告的能力、`allowed_modules`、瀏覽器程式碼狀態與目前的安裝狀態。它會讀取 manifest，但不會下載外掛 archive 或安裝檔案。結果存放在 `inspection_id` 底下。`POST /api/plugins/install` 接受該 `inspection_id`、`accept_capabilities` 與 `trust_author`；取代既有安裝時另帶 `force`。它不接受 manifest、commit 或能力清單。因此，如果 archive 的 manifest 在檢查後新增能力、變更 id 或新增 allowed module，伺服器會拒絕安裝。安裝會以 job 執行：request 回傳 `202` 與 `job_id`；`GET /api/plugins/jobs/{job_id}/events` 會重播 cursor 之後的事件，並長輪詢新事件；`POST /api/plugins/jobs/{job_id}/cancel` 會取消 job 並移除未完成的寫入。

**確認欄位對應三個安全層級。** [第 0 級](#安全性三個層級)不需要同意。第 1 級會列出檢查結果之 `capabilities` 中的每個值。第 2 級會列出 `allowed_modules`，並要求以 `trust_author` 傳送獨立的作者信任決定。這兩個層級都不是沙箱。能力獲得授權後，外掛可以 import 該組模組，不會再次顯示提示。授予存取權前，請閱讀[這不是什麼](#這不是什麼)。

**外掛安裝預設限制為本機用戶端。** inspect、install、cancel、update 與 delete routes 都要求工作階段 token，而且伺服器必須綁定至回送位址。這些 routes 可以取得外部程式碼、將其安裝至伺服器行程，或移除已安裝的外掛。區網上的教室或實驗室伺服器可用 `CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1` 允許遠端存取。reload、enable 與 disable 需要 token，但不要求回送位址，因為它們只處理現有的本機檔案。讀取 routes 保持開放，包括 job 事件，因此其他分頁可以監控執行中的安裝。

**GitHub API 請求上限。** 未驗證身分的 GitHub API 存取，限制為每個 IP 位址每小時 60 次 request。共用 NAT 後方的電腦會共用配額。配額用盡時，面板會顯示「已達 GitHub 請求上限，請稍後再試，或在伺服器設定 CODEFYUI_GITHUB_TOKEN。」（`502` `github_rate_limited`）。請在 `cdui start` 前匯出 `CODEFYUI_GITHUB_TOKEN`，或在執行 `cdui plugin install|info|update` 的 shell 中匯出；只需具備 public repository 的讀取權限。每個 request 都會從環境讀取 token，因此加入 token 後不必重新啟動伺服器。token 只會以 bearer header 傳送給 GitHub；redirect 時會移除，也不會出現在 log 或錯誤訊息中。

**安裝步驟與後端失敗訊息使用英文。** 共用後端會輸出 `Resolving …`、`Downloading …`、`Unpacking …`、`Scanning … for unsafe code`、`Installing packages: …`、`Installing …` 與 `Recording …`。同一後端輸出的拒絕與失敗訊息也使用英文。周圍的介面有翻譯，但這些訊息沒有。

**`needs_restart` 不代表失敗。** 外掛的 `[python_deps]` 只能新增套件，並套用 constraints 檔以固定執行中伺服器已載入的套件。依賴解析如果無法在即時安裝期間符合這些 constraints，job 會以 `needs_restart` 結束，並回傳要在停止伺服器後執行的確切 `command`。同一台伺服器仍在執行時重複安裝，會得到相同結果。`cdui plugin install` 也會印出該指令，並以離開碼 `3` 結束。

**解除安裝行為取決於外掛來源。** `DELETE /api/plugins/{id}` 會刪除已下載外掛的目錄。內建外掛檔案屬於發行版，因此會保留；伺服器會將外掛記錄為已移除，讓 `cdui plugin sync` 不會還原，直到再次按名稱安裝。透過 `cdui plugin link` 註冊的目錄也不會變更。Python 依賴套件不會移除，因為解除安裝執行中伺服器已 import 的模組，可能使行程處於不一致的狀態。response 會在 `python_deps_left` 中列出保留的依賴套件，並提供停止伺服器後執行的 `uninstall_command`：

```bash
uv pip uninstall --python <the CodefyUI venv's python> httpx
```

如果無法刪除目錄，操作不會進行任何變更：lockfile 條目會保留、外掛仍維持已安裝狀態，伺服器則回傳 `409` `files_locked`、作業系統錯誤與仍存在的目錄。Windows 上的常見原因是其他行程仍開啟某個檔案。請關閉該行程或停止伺服器，再重試解除安裝。

**更新會回傳三種結果之一。** `POST /api/plugins/{id}/update` 會從外掛紀錄的 repository 讀取目前 manifest。`200` `{"status": "up_to_date", "sha": …}` 表示已安裝的 commit 已相符。`202` 與 `job_id` 表示更新不需要額外同意，因此已開始安裝。`200` `{"status": "needs_consent", …}` 表示更新要求額外存取權。該 response 會包含與 `/inspect` 相同的 inspection 資料，以及確認畫面使用的 `capabilities_added` 與 `allowed_modules_added`。若要繼續，請將其 `inspection_id`、`accept_capabilities` 與 `trust_author` 傳送至 `POST /api/plugins/install`。不要傳送 `force`；伺服器會記錄該 inspection 是為更新所建立，並允許取代。

更新不能變更外掛 id。如果 repository 目前的 manifest 宣告不同的 id，伺服器會回傳 `400` `not_updatable`，且不下載外掛。這可防止更新取代另一個外掛，包括已使用新 id 安裝的外掛。請以新 id 個別安裝更名後的外掛。更新也會保留已停用外掛的停用狀態。

內建外掛包與本機連結目錄也會回傳 `400` `not_updatable`，並在 `hint` 中提供替代操作。請用 `cdui update` 更新內建外掛包。本機連結目錄已直接使用目前檔案。

**外掛中心與套件中心同時只能執行一個安裝。** 任何安裝執行時，再啟動外掛安裝或更新會回傳 `409` 與其 `job_id`。外掛正在安裝時，解除安裝、啟用或停用同一個外掛也會回傳 `409`；這可避免同時變更其 lockfile 條目。另一個外掛的安裝不會阻擋這三種操作，因為不同外掛使用不同目錄與 lockfile key。

### 拒絕代碼 {/* #refusal-codes */}

大多數安裝 routes 的拒絕回應使用 `{"detail": {"code": "...", ...}}`，其中包含 code 與用戶端所需的欄位。面板與 `cdui` 會依 code 控制流程，並可將顯示訊息本地化。有兩個例外會回傳純文字 `detail`：回送位址閘門的 `403`（「Installing plugins is only allowed from the computer that runs the server. Set CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1 to override.」），以及對未安裝外掛呼叫 `enable` 或 `disable` 時回傳的 `404`。

| 狀態碼 | Code | 來源 | 意義 |
|--------|------|------|---------|
| 400 | `unparseable_source` | inspect | 值不是型錄名稱、`owner/repo[@ref]` 或 GitHub URL。 |
| 400 | `unknown_catalog_name` | inspect | 型錄中沒有相符的裸名稱。`known` 會列出可用名稱。 |
| 400 | `reserved_id` | inspect, update | manifest `id` 與內建外掛或 `catalog` 等保留 route 名稱衝突。response 包含 `id`。 |
| 400 | `invalid_manifest` | inspect, update | manifest 沒有 `[plugin]` table、使用 `1` 以外的 `schema_version`、包含無效的 id 或 `[security]` 值、包含無效 TOML，或不是文字檔。 |
| 400 | `consent_required` | install | 一項以上宣告的能力尚未獲得同意。`missing_capabilities` 會列出這些能力。 |
| 400 | `trust_author_required` | install | manifest 有 `allowed_modules`，但 `trust_author` 不是 `true`。response 包含 `allowed_modules`。 |
| 400 | `not_updatable` | update | 外掛是內建、本機連結、沒有已記錄的 repository，或其 repository 現在宣告另一個 id。`hint` 會提供替代操作。 |
| 404 | `not_found` | inspect, update | GitHub 上沒有相符的 repository 或 ref。 |
| 404 | `unknown_job` | events, cancel | 只保留最近一個 job，因此要求的 job 無法取得。response 包含 `job_id`。 |
| 404 | `inspection_expired` | install | inspection 已過期。請重新檢查來源。response 包含 `inspection_id`。 |
| 404 | `not_installed` | update, DELETE | 該 id 底下沒有已安裝的外掛。 |
| 409 | `already_installed` | install | 外掛已安裝。請使用 `force: true` 重試，這等同面板的**重新安裝**操作。response 包含 `plugin_id`。 |
| 409 | `busy` | install, update, DELETE, enable, disable | 有安裝正在執行：任何外掛都會阻擋 `install` 與 `update`；外掛會阻擋自己的 DELETE、enable 與 disable 操作。response 包含 `job_id`。 |
| 409 | `pack_install_running` | install, update | 套件中心正在使用兩個中心共用的安裝 slot。response 包含 `job_id`。 |
| 409 | `inspect_busy` | inspect, update | 另一個 inspection 正在執行。請等它完成後再試。 |
| 409 | `files_locked` | DELETE | 無法刪除目錄，且沒有變更任何狀態。這通常表示 Windows 上仍開啟某個檔案。response 包含 `error` 與 `hint`。 |
| 502 | `github_rate_limited` | inspect, update | GitHub 回傳 403 或 429。請等待配額重設，或設定 `CODEFYUI_GITHUB_TOKEN`。 |
| 502 | `github_unreachable` | inspect, update | 另一項錯誤導致伺服器無法連線至 GitHub。 |
| 503 | `unavailable` | inspect, install, events, cancel, update | 外掛 service 未啟動。`GET /catalog` 仍可使用，面板則顯示「這台伺服器不支援外掛中心。」 |

## 撰寫你自己的外掛

最快的方法是 **`cdui plugin new`**，一個指令就能產生可直接編輯的外掛骨架：

```bash
cdui plugin new my-plugin          # 純後端骨架
cdui plugin new my-plugin --ui     # 另含一個接好 SDK 的 React 前端
```

它會產生 manifest、一個範例節點、一個測試（內含 `cdui_plugins.<id>` 命名空間 shim，讓本地 `pytest` 可直接執行），並在加上 `--ui` 時產生一個 Vite + React 的 `ui/`，其 `src/sdk/` 即為型別化的外掛 SDK。外掛會建立在 `./my-plugin/`；用下方的 `cdui plugin dev` 連結後即可開始編輯。

若需更完整的參考，可 fork **[官方外掛模板](https://github.com/CodefyUI/CodefyUI-Plugin-Official)**——一個可運作、採 MIT 授權的外掛，包含兩個範例節點、一張範例圖、一套測試，以及一份完整註解的 manifest。它的 README 逐欄解說每個欄位與 AST 安全閘門。型錄中的名稱是 `official-template`，可以直接按名稱安裝。

```bash
# Install the template itself to see the pattern live
cdui plugin install official-template

# After forking — any repository, by owner/repo or by URL
cdui plugin install your-username/your-fork
```

請將外掛內容放在 manifest 旁的固定目錄中：`nodes/`（自動探索）、`presets/`、`examples/`、`assets/`（於 `/plugins/<id>/assets/<file>` 提供），以及 `frontend/`（參閱[外掛前端擴充](/advanced/plugin-frontend-extensions)）。這些目錄名稱無法設定，而且 scaffold 的 `[content]` table 會被忽略。`cdui.plugin.toml` manifest 會宣告 id、版本、課程 metadata，以及[安全性](#安全性三個層級)所述的任何 `[security]` 設定。節點只使用第 0 級 imports 時，請省略 `[security]`。

### manifest 欄位參考 {/* #manifest-reference */}

只有安裝或載入會使用的欄位才會驗證。其他欄位可能未經驗證就顯示或儲存。

| 欄位 | 驗證 | 功能 |
|-------|-----------|--------------|
| `[plugin] id` | 是 | 必須符合 `^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$`。它是安裝目錄、節點 namespace（`<id>:<NODE_NAME>`，保留連字號），也是安裝與解除安裝時所用的名稱；只有 import 它的 Python package 會轉成 snake_case（`cdui_plugins.my_plugin`）。 |
| `[plugin] schema_version` | 是 | 必須為 `1`；其他值都會被拒絕（「Unsupported plugin schema_version」）。 |
| `[plugin] name`, `version`, `description` | 否 | 顯示在外掛中心的卡片與檢視卡片、`cdui plugin info`，以及 `GET /api/plugins`。`name` 預設退回 id。 |
| `[plugin] homepage` | 否 | 檢視卡片上的**首頁**連結；只接受 http(s) URL。 |
| `[plugin] authors`（list）或 `author`（string） | 否 | 只顯示在檢視卡片上（「作者：...」）。 |
| `[plugin] requires_codefyui`, `license` | 否 | 儲存但不會強制執行、檢查或輸出。 |
| `[security] capabilities` | 是 | 只能從 `network`、`filesystem` 與 `process-env` 中選取的 string list——[第 1 級](#安全性三個層級)。任何未知名稱都會讓整份 manifest 被拒絕。 |
| `[security] allowed_modules` | 是 | module name list——[第 2 級](#安全性三個層級)，只有使用 `--trust-author` 或勾選檢視卡片上的「我信任這位作者」才能安裝。若是單一 bare string 而非 list，會被拒絕。 |
| `[python_deps]` | 安裝時 | `name = "constraint"` pair，會在複製檔案前以 `uv pip install` 安裝。以 operator 開頭的 constraint 會原樣使用（`">=0.27"`）；bare version 會被 pin（`"1.2.0"` 變成 `==1.2.0`）；空字串表示任何版本。extras、URL 與 `git+` source 都會被拒絕。 |
| `[frontend] entry` | 載入時 | 必須以 `frontend/` 開頭的相對 POSIX 路徑（`"frontend/index.js"`）；其他值一律視為「沒有 frontend」。 |
| `[lessons] chapters`, `lessons` | 否 | string list：卡片上的**章節：**一列，以及 `cdui plugin info`。 |
| `[content]` | 否 | 忽略——參閱上文。 |

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

請在另一個終端機執行伺服器（`cdui start` 或 `cdui dev`）。`dev` 會輪詢外掛 manifest、`nodes/`、`presets/` 與 `frontend/`；`--once` 會連結並重載一次而不監看，`--interval` 則設定輪詢間隔。會要求伺服器重載的命令——`link`、`unlink`、`dev`、`reload`、`enable`、`disable`、`install` 與 `uninstall`——會將 POST request 傳送至 `127.0.0.1:<port>`。這些命令會使用所在 shell 的 `CODEFYUI_PORT`；未設定時使用 `8000`。`cdui start --port <n>` 只會為其啟動的伺服器行程設定該變數。使用非預設 port 時，請先匯出 `CODEFYUI_PORT=<n>`，再執行這些命令或以相同方式解析 port 的 `cdui project publish`。否則，lockfile 可能已更新，但重載 request 會回報 `Server not running`。

`link` 會從你的 `cdui.plugin.toml` 讀取 id，並把該目錄的絕對路徑以 `source_kind = "local"` 記入 lockfile，因此探索會直接走訪你的工作目錄。連結的外掛會跳過 AST 安全閘門（這是你自己的程式碼，並會印出警告）；`unlink` 只移除 lockfile 條目，絕不刪除你的檔案。編輯 Python 節點後，執行 `cdui plugin reload`（或 `cdui plugin dev`）即可重載。**連結中的外掛，前端變更也會自動重載**——只要安裝著連結的外掛，編輯器就會偵測重載並就地重新掛載外掛 UI，不需手動重新整理瀏覽器。

連結外掛的 `[python_deps]` 會依照與下載型外掛包相同的規則安裝：只增不改，並套用 constraints 檔，固定執行中伺服器已載入之每個套件的版本。因此 `cdui plugin link` 也使用安裝流程的離開碼：`3` 表示它要求的套件無法安裝至執行中的伺服器（會印出伺服器停止後要執行的指令），`130` 表示 `Ctrl+C`。它不再直接回傳套件管理程式的原始離開碼。

:::tip 每個安裝各有一份 lockfile
`cdui` 命令由 `scripts/dev.py` 實作。除非已設定 `CODEFYUI_USER_DATA_DIR`，否則 `cdui start`、`cdui dev`、`cdui run` 與 `plugin` / `project` / `cache` / `packs` 指令群組會將它設為目前安裝的 `<install dir>/.codefyui_dev/`（參閱[外掛包如何儲存](#外掛包如何儲存)）。因此，每個 clone 都有不同的 lockfile 與本機連結外掛註冊。在一個 clone 中連結的外掛，不適用於從另一個 clone 啟動的伺服器。若要使用其他位置，請匯出 `CODEFYUI_USER_DATA_DIR`；既有值具有優先權。
:::

## REST API

| 端點 | 方法 | 驗證 | 說明 |
|----------|--------|------|-------------|
| `/api/plugins` | GET | open | 列出所有已安裝的外掛，包括已停用的外掛。 |
| `/api/plugins/catalog` | GET | open | 合併型錄條目與已安裝外掛。每個外掛處於[六種狀態](#install-states)之一；response 也包含 `active_job`、`remote_install_allowed` 與 `generation`。 |
| `/api/plugins/generation` | GET | open | 回傳編輯器重新整理節點面板時輪詢的重載計數器。 |
| `/api/plugins/{id}` | GET | open | 回傳一個外掛的 manifest、節點與 README。 |
| `/plugins/{id}/frontend/{path}` | GET | open | 當已啟用外掛的 manifest 宣告 `[frontend].entry` 時，提供其 `frontend/` 中的檔案。此 route 不在 `/api` 底下。每個 request 都會讀取 lockfile，因此檔案會在安裝或重載後可用，並在停用或移除後回傳 `404`。response 使用 `Cache-Control: no-cache`。 |
| `/plugins/{id}/assets/{file}` | GET, HEAD | open | 提供已啟用外掛 `assets/` 中的檔案。此 route 使用相同的啟用規則，但不要求 frontend manifest entry。media type 依副檔名決定，預設為 `application/octet-stream`。 |
| `/api/plugins/jobs/{job_id}/events` | GET | open | 回傳 `?cursor=` 之後的安裝 job 事件；`?wait=` 會長輪詢之後的事件。 |
| `/api/plugins/reload` | POST | token | 重新探索節點、預設模組與外掛。 |
| `/api/plugins/{id}/enable` | POST | token | 啟用已安裝的外掛。 |
| `/api/plugins/{id}/disable` | POST | token | 停用已安裝的外掛，但不解除安裝。 |
| `/api/plugins/inspect` | POST | token + loopback | 在單一 commit 上檢查來源並回傳安裝與權限需求，但不進行安裝。 |
| `/api/plugins/install` | POST | token + loopback | 安裝 `inspection_id` 所識別的結果；回傳 `202` 與 `job_id`。 |
| `/api/plugins/jobs/{job_id}/cancel` | POST | token + loopback | 取消執行中的安裝 job。 |
| `/api/plugins/{id}/update` | POST | token + loopback | 檢查外掛已記錄的 repository 是否有更新。回傳 `202` `{job_id}`、`200` `{status: "up_to_date", sha}`，或 `200` `{status: "needs_consent", inspection, capabilities_added, allowed_modules_added}`。對 `needs_consent`，用戶端會傳送不含 `force` 的 `POST /install {inspection_id, accept_capabilities, trust_author}`。 |
| `/api/plugins/{id}` | DELETE | token + loopback | 解除安裝外掛，並回報保留的檔案或依賴套件。 |

**open** 表示不需要憑證的讀取 route。**token** 要求工作階段 token header。**token + loopback** 還要求伺服器綁定至回送位址，除非已設定 `CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1`。受影響的操作請見[外掛中心](#plugin-center)。
