---
sidebar_position: 7
title: PythonScript 節點
description: 直接在畫布上寫 Python——函式介面、Tier-0 匯入政策、誠實的安全模型，以及統計常用範例。
---

# PythonScript 節點

CodefyUI 裡其他每個節點，都是某個人寫好、安裝進來的 Python 檔案。**PythonScript** 則是內容由你在瀏覽器裡直接輸入的節點。它存在的理由很簡單：統計與研究的需求永遠跑在節點庫前面，而為了四行 numpy 去寫一個[自訂節點](./custom-nodes.md)或發一個[外掛套件](./plugins.md)，手續實在太多。

從節點面板拖出它（分類 **Utility**），在節點上雙擊打開 **程式碼** 分頁，然後寫：

```python
def run(inputs, params):
    x = inputs["in1"]
    return {"out1": x.mean()}
```

## 函式介面

```python
def run(inputs: dict, params: dict) -> dict
```

| | |
|---|---|
| `inputs` | **有連線**的每個輸入連接埠各一個鍵：`in1`、`in2`……。沒接線的連接埠不會出現在字典裡，所以在你自己的設計中屬於選用的連接埠請用 `inputs.get("in2")`。 |
| `params` | 這個節點參數的複本（也包含 `code` 本身）。改動它不會有任何效果——節點交給你的是複本。 |
| 回傳 | 以輸出連接埠為鍵的字典：`{"out1": ..., "out2": ...}`。**直接回傳單一值時會當作 `out1`**，所以只有一個輸出時 `return x` 就夠了。 |
| 多餘的鍵 | 回傳了不對應任何連接埠的鍵會被捨棄，並在執行紀錄中說明，而不是默默吞掉。 |

`run` 必須定義在模組層級、名稱完全一致。少了它時編輯器會立刻提醒，不會等到執行失敗才說。

`async def run` 會**在檢查階段就被拒絕**：節點是在沒有事件迴圈的工作執行緒上呼叫 `run`，不會有人 await 它，輸出連接埠只會拿到一個 coroutine 物件。

除了允許的函式庫之外，命名空間裡還有兩個名稱：

* `device`——本次執行解析出來的運算裝置，讓 `torch.zeros(3, device=device)` 落在流程其他部分所在的裝置上。
* `should_stop()`——本次執行的協作式停止旗標。腳本無法從外部被中斷（見下方的安全模型），所以想要能被停止的長迴圈必須自己詢問：

  ```python
  def run(inputs, params):
      total = 0.0
      for row in inputs["in1"]:
          if should_stop():
              break
          total += float(row.sum())
      return total
  ```

## 連接埠

`input_ports` 與 `output_ports`（各 1–8）決定節點有幾個連接點。**程式碼** 分頁的連接埠區塊可以同時設定數量與每個連接埠的 `DataType`；同樣的值以逗號分隔字串存在 `input_types` / `output_types` 參數裡，所以流程 JSON 仍然易讀。

* 輸入連接埠一律是**選用**的：需要哪些由腳本自己決定，所以宣告四個連接埠只接兩條線仍是合法的流程。
* 輸出連接埠預設是 `ANY`，可以接到任何地方。改成實際型別（`TENSOR`、`SCALAR`、`STRING`……）之後，流程驗證器就會幫你檢查接線——腳本寫好之後很值得補上。
* 調降連接埠數量時，接在已消失連接埠上的連線會被**移除**，受影響的下游節點也會被標記為需要重跑。

## 快取

**這個節點永遠不會被快取。它每次執行都會重跑，這是刻意的，即使上游什麼都沒變也一樣。**

`code` 確實是一個普通參數，執行快取也確實把它算進金鑰——改了腳本就會重跑。但那只涵蓋「腳本被改過」的情況；真正會命中快取的，是腳本沒改、輸入也沒變的那一次，而快取命中會直接回傳節點記錄下來的輸出，**完全不執行它**。對純轉換來說，這樣看不出差別；但如果腳本除了回傳值以外還「做了些什麼」，那些事在第二次就悄悄沒發生。

只有你的腳本知道自己屬於哪一種，所以節點採取保守的答案。有兩條副作用路徑，任何腳本都走得到，而且不可能在不毀掉這個節點的前提下封起來：

* **就地修改輸入。** `inputs` 是淺複製，所以裡面的值就是上游節點自己的物件。`inputs["in1"].add_(1)`、對 MODEL 的參數做 `p.data.zero_()`、或 `inputs["in1"].append(x)`，改動的都是每一個下游節點共用的狀態。這就是普通的數值運算——沒有任何規則能合理地把它擋下來。
* **行程層級的全域狀態。** `torch.manual_seed`、`numpy.random.seed`、`torch.set_default_dtype`、`torch.set_grad_enabled` 全都碰得到，而且每一個都會改變這次執行「其餘部分」算出來的東西。

透過輸入連接埠拿到的東西是第三條路徑：連接埠型別是 `ANY`，所以自訂節點或外掛可以把寫檔器、記錄器或已開啟的檔案握柄交給你的腳本，而 Tier-0 政策完全看不到——Tier-0 限制的是腳本可以 import 哪些**函式庫**，不是別人**交給它**的物件能做什麼。

代價是實在的：純轉換的腳本每次都會重跑，而且因為退出快取會[往下游傳播](../usage/running-graphs.md#從不快取的內容)，它餵的所有節點也一樣。如果某個步驟很貴而且確實是純的，它該寫成[自訂節點](./custom-nodes.md)——自訂節點可以為自己宣告 `cacheable = True`，因為它的作者看得到它全部的程式碼。

## 輸出與錯誤

腳本裡的 `print()` 會被擷取（每次執行上限 64 KB），以節點的紀錄行出現在**執行紀錄**中。這不是全域的 stdout 轉向——直接寫入 `sys.stdout` 的函式庫仍會輸出到伺服器主控台——因為節點共用執行緒池，劫持整個行程的 stdout 會把其他節點同時印出的內容一起吃掉。

發生例外時，訊息會標出**你腳本裡的行號**：

```
PythonScript failed at line 4: ZeroDivisionError: division by zero
```

標出的是你程式碼裡最深的那一層：在 `statistics.mean([])` 裡爆掉會指向呼叫它的那一行，在你自己寫的輔助函式裡爆掉則指向那個函式。腳本在失敗前印出的內容也會附在訊息後面。流程的錯誤處理模式（fail-fast／continue／retry）與其他節點完全一致。

## Tier-0 政策

這裡有兩層，而只有其中一層是真正的界線。

1. **AST 關卡。** 程式碼在**每次編輯**時就會檢查，而且是在編譯之前——用的是把關外掛套件的同一套 AST 檢查器（`backend/app/core/plugin_validator.py`），只是切換成允許清單模式。被拒絕時你會立刻在編輯器下方看到紅色訊息並標出該行，而不是在訓練流程跑了十分鐘後才失敗。它的規則認的是**名稱**，這讓它跑得快、也讓它是個好編輯器提示，但不是一道好牆。
2. **執行期的模組代理**（`backend/app/core/script_proxy.py`）。腳本實際執行的命名空間裡**沒有**主行程真正的模組物件，放的是受限的代理物件；它判斷的是一個屬性**解析出來的東西**：只要拿回來的模組不在 Tier-0 清單上，不管那個屬性叫什麼都會被拒絕。這一層才是真正守得住的界線。

前後三次安全審查都穿過了第一層，每一次都是從沒人列進清單的名字進去的——先是 `__loader__`，再來是 `torch.os`，然後是 `collections._sys`（那就是真正的 `sys` 模組，而 `sys.modules['os']` 是其餘的一切）。這不是運氣不好，這就是「認名稱的清單」必然的結果。第二層問的是**我剛剛拿到的是什麼**，所以底線別名、`sys.modules` 下標、綁到區域變數上、乃至明年函式庫改版，全都是同一條規則。

**可匯入的模組**（`backend/app/core/script_policy.py` 的 `TIER0_MODULES`）：

```
collections   itertools   json   math   numpy   re   statistics   torch
```

這些模組也已經以同名**預先綁定**在命名空間裡，所以不寫 import 也能直接用 `math.floor(x)`；喜歡別名的話，`import numpy as np` 也可以。兩種寫法綁到的都是代理物件而不是模組本身——`import` 走的是同一道守衛。

**可觸及的子套件**是逐一列入允許清單（`TIER0_MODULE_PATHS`），不是由根套件自動涵蓋：

```
collections.abc   numpy.linalg   numpy.random
torch.nn          torch.nn.functional   torch.signal   torch.signal.windows
```

這些根套件底下的其他東西——`torch.utils`、`torch.package`、`torch.fx`、`torch.serialization`、`numpy.f2py`、`numpy.testing`、`numpy.lib.format`——一律拒絕，因為它們每一個都藏著一行就能逃出去的路。如果你需要的做法剛好少了某一個，那是刻意畫的線而不是疏漏：說一聲，可以單獨評估。

### 代理會拒絕什麼

* **清單以外的任何模組，不管用什麼方式碰到。** `collections._sys`、`statistics.random`、`json.codecs`、`torch.cuda.tunable.mp`（那其實是標準函式庫的 `multiprocessing`）——判斷依據是模組自己的身分，所以沒有別名可找、也沒有名字要補。**被允許**套件底下的子模組（`numpy.linalg`、`torch.nn.functional`、`torch.signal.windows`）會包成巢狀代理，正常可用。
* **函式庫的私有屬性與雙底線屬性**：`re._parser`、`statistics._sum`、`numpy.__version__`。函式庫的私有名稱正是它自己那些 import 的所在。請改用公開 API——但請注意，在 Tier-0 之下函式庫的**版本號根本拿不到**：`torch.__version__` 和 `numpy.__version__` 是雙底線屬性，而 `torch.version` / `numpy.version` 是子套件允許清單沒有收錄的模組。需要依函式庫版本分支的腳本，應該寫成[自訂節點](./custom-nodes.md)。
* **對模組取下標**——`m['os']`——這樣一份「模組的對照表」就不能成為繞過屬性規則的路。
* **對函式庫屬性指派或刪除**：`torch.zeros = 我的函式` 以前會把 `torch` 改給行程裡其他所有節點。現在兩層都會拒絕。
* 把模組當成函式呼叫、對模組做迭代，以及下面列出的每一個被封鎖屬性。

* **本身就是能力、而不是資料的值**：`pathlib.Path`（類別或實例）、開啟的檔案物件、`mmap`、`os.DirEntry`——以及任何由被封鎖模組所**定義**的東西，例如 `subprocess.Popen` 或 `importlib.find_spec`，不論被允許的函式庫用什麼名字把它重新匯出。檢查看的是值的型別，所以子類別與任何別名都涵蓋在內。

其餘一般的值會**原樣**回傳：`torch.zeros(3)` 就是普通的張量，不是代理。這是刻意的——連資料一起代理，等於在每個腳本的每一次 `.mean()` 前面都加上一層 Python 檢查。這同時也是剩下那項殘留風險的形狀，詳見安全模型那一節。

**會被關卡拒絕，並附上替代路徑說明：**

* 其他任何匯入——`os`、`sys`、`pathlib`、`subprocess`、`socket`、`urllib`、`requests`，以及 `pandas`、`sklearn` 這類並不危險、只是不在清單上的模組。相對匯入同樣被拒絕。
* `exec`、`eval`、`compile`、`__import__`、`open`、`input`、`globals`、`locals`、`vars`、`dir`、`breakpoint`、`exit`——以**裸呼叫**形式，以及透過定義它們的模組（`builtins.eval`）。只是剛好同名的方法並不是那個內建函式，因此是允許的：`re.compile(r'\d+')` 與 `model.eval()` 在規則學會問「這個 `compile` 是誰的」之前一律被拒絕。它們也永遠不可能是內建函式——代理會以物件識別（identity）拒絕真正的 `compile` / `eval` / `open`，不論它被叫成什麼名字，而 `builtins` 也在下方的拒絕屬性清單上，所以第 0 級腳本根本沒有路徑碰到它。**有一個刻意的例外：** `.compile` 只在 `re` 上被允許。`torch.compile(m)`——以及 PyTorch 2.2 起在 `nn.Module` 上新增的 `m.compile()` 方法——仍然被拒絕，因為它們會把運算圖交給 TorchInductor，而後者會產生 C++/Triton 並呼叫真正的編譯器。那和拒絕 `torch.utils.cpp_extension` 擋的是同一扇門。已安裝的外掛不受影響，兩者都可以呼叫。
* 直接使用模組機制的名稱——`__loader__`、`__spec__`、`__builtins__`、`__package__`——以及雙底線屬性存取（`__class__`、`__globals__`、`__subclasses__`、`__code__`、`__traceback__`……）。這份清單請讀成**我們已知的逃逸手法，一項一項列出來**，而不是「反射這一整類都處理好了」的保證。清單上的每一項都曾是真的能逃出去的路：`__loader__.load_module('nt')` 不用任何 import 就能拿到真正的 `os` 模組。
* **走訪執行框架（frame）**，不論接收端是誰：`tb_frame`、`tb_next`、`f_back`、`f_globals`、`f_locals`、`f_builtins`、`f_code`、`gi_frame`、`gi_code`、`cr_frame`，以及這一族的其餘成員。被接住的例外身上帶著 traceback，traceback 身上帶著它被丟出時的那個 frame，而**呼叫**你的那個 frame 屬於 CodefyUI 自己：`e.__traceback__.tb_frame.f_back.f_globals` 會交出這個節點自己的模組全域變數，裡面就有 `importlib` 與 `builtins`。到了那一步，腳本原本拿到的是哪一套 builtins 已經不重要了——所以這些名稱是直接拒絕，而不是設法清理。
* **你不能 import 的模組名稱，被當成屬性來取用**：`torch.os`、`torch.sys`、`torch.serialization.pickle`、`json.codecs.sys`、`numpy.f2py.subprocess`。函式庫自己也會 import 東西，所以一份只看 `import` 陳述句的允許清單，只要你向某個被允許的模組指名要，它就會把被封鎖的模組直接遞出來。這些名稱取自 import 規則用的同一份封鎖清單，只排除 `torch.signal`（那是 torch 自己的訊號處理命名空間，不是標準函式庫的同名模組），並額外加上 `builtins`——它不能放進那份共用封鎖清單，因為那份清單同時也是代理的「這個值是哪個模組**定義**的」規則，而 `builtins` 定義了函式庫遞回來的每一個 `int`、`str` 與 `float`，`math.pi` 也包含在內。這一類現在由代理從結構上擋住；名稱規則留下來，是因為它是編輯器在你打字時就能顯示的版本。
* **函式庫的私有屬性**——`collections._sys`、`statistics.random._os`、`re._parser`——只要接收端是那八個被允許的模組之一就會被拒絕。你自己類別裡的 `self._cache` 是普通的 Python，仍然合法。
* **對函式庫指派**：`torch.zeros = 我的函式`、`del numpy.mean`。
* 允許的函式庫**內部**通往檔案系統、網路、編譯器或其他行程的那些門：`torch.hub`（會下載並執行遠端的 `hubconf.py`）、`torch.utils.cpp_extension`（編譯並執行 C++）、`torch.distributed`、`torch.multiprocessing`、`numpy.savetxt` / `loadtxt` / `fromfile` / `fromregex` / `tofile` / `save` / `dump` / `memmap`、`numpy.ctypeslib`，以及 `script_policy.py` 裡 `TIER0_DENIED_ATTRS` 的其餘項目。這份清單裡的 `dump` 是唯一長在**陣列**上、而不是長在模組上的：`numpy.zeros(3).dump(path)` 會把陣列直接 pickle 寫進你指定的任何路徑，每一種 numpy 陣列與純量型別都帶著它，而且寫進去的內容大致由你決定。因為這些規則是**不管接收端是誰**、只認屬性名稱，所以單一條目就一次擋掉它所有的寫法——代價是 `json.dump` 也一起被擋（`json.dumps` 不受影響，而 `json.dump` 本來就需要一個腳本拿不到的檔案物件）。用 import 把它們的名稱帶進來、或用字面值 `getattr` 取得，同樣會被拒絕；`os.system` / `.popen` / `.spawnv` 現在也**當成屬性**擋掉，而不只是擋呼叫——因為 `f = obj.system` 之後再 `f(cmd)`，只差一行指派就繞過了任何「認呼叫」的規則。
* 在 `json` 以外的任何東西上使用 `.load(...)` / `.loads(...)`——包括只是**讀取**這個屬性，例如 `f = torch.load`。這些函式會執行來源檔案裡的程式碼。這條規則刻意訂得很鈍。接收端會透過 import 別名**以及**單純的指派來解析（`b = torch; b.load(x)`），而只要是檢查器解析不出來的接收端——`(lambda: torch)().load(x)`、`things[0].load(x)`——一律拒絕，而不是放行；代價是你自己寫的 `obj.load()` 輔助方法也會一起被拒絕。

`json.load` 與 `json.loads` 是最後這條規則的例外——`json` 本來就是 Tier-0 模組，解析 JSON 正是它的用途。而且它是**唯一**的例外：八個允許的模組裡，只有 `json`、`numpy`、`torch` 有 `.load`，另外兩個正是 pickle 那兩道門。

:::note `weights_only=True` 不再是 Tier-0 的通融寫法
這一頁先前告訴你「真的要載入時就寫 `torch.load(path, weights_only=True)`」。現在那樣寫會被拒絕。執行期的代理交出去的是**屬性**而不是呼叫：它看不到關鍵字引數，所以只要 `torch.load` 拿得到，`f = torch.load; f(p)` 連同任何引數就都拿得到。Tier 0 本來就沒有檔案存取（`open` 同樣被禁），所以現在兩層一致地說不，而不是讓關卡承諾執行期不認的事。要載入檢查點，請用[自訂節點](./custom-nodes.md)或內建的載入節點。
:::

**哪一層守哪一條。** builtins 允許清單（所以 `open`、`eval` 這些不只是不能改，而是根本不存在）、受守衛的 `__import__`，以及上面每一條模組／屬性規則，在代理層都有執行期的鎖。仍然**只在 AST 階段**檢查的是代理看不到的那些反射手法：雙底線屬性（`__class__`、`__globals__`、`__code__`……）、走訪執行框架（`f_globals`、`gi_frame`、`tb_frame`……），以及用計算出來的名字呼叫 `getattr`。這些活在一般 Python 物件上、而不是函式庫表面上，所以關卡仍然必須在編譯之前跑。

**綠色標記不是保證。** 編輯器跑的是 AST 關卡，所以關卡挑不出毛病的腳本，仍然可能在執行途中被拒絕——`numpy.f2py.crackfortran.myeval` 與 `torch.utils.collect_env.run` 都是透過被允許的模組碰到的 numpy／torch 自家子套件，這兩行裡沒有任何一個字看起來像規則，代理則直接拒絕。真的發生時，你會在執行紀錄裡看到同一段政策訊息，並附上你的行號。

### 需要清單以外的東西？

那正是另外兩條路存在的理由，而且它們更適合：

* [自訂節點](./custom-nodes.md)——放在你自己專案裡的檔案，你寫的、你能檢查。
* [外掛套件](./plugins.md)——可安裝、有版本，並依它自己的[三級政策](./plugins.md#安全性三個層級)執行：純運算不需要任何宣告，`network` / `filesystem` / `process-env` 在 manifest 宣告並於安裝時由使用者確認，超出這三者的則需要 `--trust-author`。

## 安全模型——這是防護欄，不是沙箱

在把 CodefyUI 開到網路介面上之前，請先讀這一節。

這套政策擋的是**容易的**逃逸手法。它**不是**沙箱，也沒有打算變成沙箱：

* **這道界線真正保證的是什麼。** 腳本拿不到頂層套件不在那八個允許模組裡的任何模組——不管是透過 import、屬性、私有別名、下標、綁到區域變數，還是字面值 `getattr`——因為檢查的是**拿回來的物件**，而不是被打出來的名字。在這之上，本身就是已知檔案 IO 能力、或由被封鎖模組所定義的值，也會用同一種「看型別」的檢查擋掉。承諾就是這兩句；以下全都是**沒有**承諾的事。它們是規則本身的性質，而不是「今天這一版函式庫」的狀態；這正是這一頁上一版寫錯的地方：它宣稱「掃過之後找不到還能碰到的封鎖模組」，而在寫下那句話的當下，`collections._sys` 回傳的就是真正的 `sys` 模組。認名稱的掃描，永遠只能描述它剛好走過的那些名字。
* **它框住的是腳本能碰到哪些函式庫，而不是那些函式庫能做什麼。** numpy 與 torch 本身就大到內建了檔案 IO、下載功能與一個 C++ 編譯器。那份拒絕清單關掉的是我們知道的門；它是架在兩套龐大 API 之上的黑名單，只能視為提高逃逸的成本，絕不是「檔案與網路碰不到」的保證。
* **不是模組的能力，改用型別擋掉。** 類別不是模組，所以上面那條模組規則從來看不到 `pathlib.Path`——而 `Path` 就是任意檔案讀取**與寫入**。它曾能透過 `numpy.f2py.crackfortran.Path`、`numpy.f2py.f2py2e.rules.Path`、`torch.fx.graph_module.Path`、`torch.package.package_exporter.Path` 拿到，還能在 `numpy.testing.NUMPY_ROOT` 拿到一個**實例**：同一個類別、五種寫法，沒有一個是任何清單會收錄的名字。現在有第二條規則，對「值」問的是第一條規則對「模組」問的同一件事——**這到底是什麼東西**——並拒絕本身是（或屬於）已知檔案 IO 型別的值（`pathlib.PurePath`、`io.IOBase`、`mmap.mmap`、`os.DirEntry`，含子類別），以及由本政策封鎖的模組所**定義**的值（`subprocess.Popen`、`threading.RLock`、`importlib.find_spec`……不論被哪個函式庫重新匯出）。
* **子套件是逐一列入允許清單，不是看根套件。** 「只要頂層套件被允許就放行」是先前的規則，實際量過之後才發現它太寬鬆：numpy 與 torch 各自都帶了一些本質上是工具、而不是數值運算的子套件，而下面每一項都只要一行——`torch.utils.collect_env.run("cmd /c whoami")`（torch 自己包的 `subprocess`，而且 `shell=True`）、`numpy.f2py.crackfortran.myeval(...)`（貨真價實的 `eval()`）、`torch.package`（把原始碼寫出去再執行它）、`numpy.lib.format.open_memmap(p, mode="w+")`（在任意路徑建立檔案）、`torch.fx.symbolic_trace(m).to_folder(p)`（寫出真正的 `.py`）、`torch.serialization.mkdtemp()`。它們都不是能力**型別**，也都不是被封鎖模組所定義的——它們是 numpy 與 torch 自己的函式，做的正是那些子套件存在的目的，所以型別規則看不到它們。現在允許清單是一份**完整模組名稱**的清單（`TIER0_MODULE_PATHS`），子套件只有列在上面才碰得到。
* **殘留風險：型別清單是有限的，模組清單也是。** 這條能力規則涵蓋檔案 IO 型別、任何被封鎖模組（以及它們的 C 實作模組 `nt`、`posix`、`_io`、`_thread`……）所定義的東西，還有本政策從命名空間移除的那些 builtins——後者是用物件本身的身分比對，所以 `os.getcwd` 或 `open` 被重新匯出成別的名字時，它仍是同一個函式、同樣被拒絕。它**不**涵蓋：型別由被允許函式庫自己定義的能力，以及任何呼叫的回傳值——代理交出去的是屬性，函式拿回傳值做什麼已經超出它的範圍。**兩種掃描**，兩者都已寫成測試裡的常駐檢查。第一種走過政策允許的每一個屬性（深度 4）——15 個模組、2,052 個一般值——找不到任何能力。第二種檢查所有碰得到的可呼叫物件有沒有看起來像檔案路徑的參數。它原本只掃到模組層（747 個可呼叫物件），找到一個真正的目標：`numpy.fromregex`——它會打開你給的路徑。但「只掃到模組層」本身就是缺陷：它從來沒有往下走進那些模組交出來的**類別**，而 `numpy.ndarray.dump(path)` 正是住在那裡——把陣列直接 pickle 寫進任何路徑，每一種陣列與純量型別都帶著它，大約五十種寫法，而且在它敞開的整段期間，編輯器的檢查一路都是綠燈。現在這項掃描也會走過類別方法（462 個類別、15,851 個方法）。兩個目標都已關閉，每次提交跑的就是擴充後的這個版本。把型別規則關掉再掃一次，同樣找不到——這才是誠實的解讀：自從子套件允許清單落地之後，那些型別規則變成「目前後面沒有東西」的第二道鎖，而不是真正在擋的那一層。在它落地之前，同樣的掃描橫跨 330 個模組，型別規則當時擋下了七個真實可用的能力。這兩個數字描述的都是今天的函式庫，不是規則本身的性質。
* **反射手法仍然只由關卡擋。** `__globals__`、`__class__`、frame 屬性，以及用計算出來的名字呼叫 `getattr`，是由 AST 檢查器拒絕的，沒有別的東西擋——因為它們活在一般物件上，而不是函式庫表面上。
* 腳本在 **CodefyUI 伺服器行程內**執行，使用你的使用者權限。沒有任何容器隔離。
* 已經能在你畫布上打字的攻擊者，只要夠有決心，仍可能找得到出口。這套政策提高的是隨手執行程式碼的成本，並不能讓這個面向對有備而來的對手變得安全。
* 沒有任何 CPU 或記憶體限制，而且失控的腳本不只影響自己的節點。節點是在直譯器的**預設執行緒池**上執行，所以某個腳本裡的 `while True:` 會餓死這個行程裡**所有**節點的執行，直到伺服器重啟。「停止」是協作式的、只在節點**之間**檢查，無法中斷節點內部的迴圈——長迴圈必須自己呼叫 `should_stop()`。
* `code` 參數跟其他參數一樣會存進流程 JSON。**打開來路不明的流程並按下執行，等於執行對方的 Python。** 政策檢查是這中間唯一的一道防線，這正是它在編譯之前就跑、而不是等到匯入時才跑的原因。

**這個節點首次發布以來已修掉的問題**：污染模組。允許清單裡的模組以前交出去的是本尊，所以 `torch.zeros = 別的東西` 會改掉行程裡其他所有節點看到的版本。現在兩層都會拒絕。

真正的界線是**誰能碰到這個編輯器**。CodefyUI 預設只綁定本機；除非你信任該網路上的所有人，否則請維持原狀。

## 匯出

「匯出為 Python」會把腳本以「一行原始碼一個字串常量」的形式寫進產生的檔案，並附上來源註解：

```python
def n02_pythonscript(ctx):
    "PythonScript - node 'py1'."
    params = {
        # ---- 'code' of node 'py1': the script this node runs, verbatim ----
        'code': (
            'def run(inputs, params):\n'
            '    x = inputs["in1"]\n'
            '    return {"out1": x.mean(dim=0)}\n'
        ),
        'input_ports': 1,
        'output_ports': 1,
    }
    return _call('PythonScript', 'py1', params, ctx, inputs={'in1': in1})
```

它易讀、可改——改一行，匯出的流程就照著改動跑——而且仍然只是字串常量，所以你程式碼裡的 `'''` 永遠不可能變成程式本體。

## 統計常用範例

### 每個通道的平均與標準差

一個 TENSOR 進、兩個 TENSOR 出（`output_ports: 2`、`output_types: TENSOR,TENSOR`）：

```python
def run(inputs, params):
    x = inputs["in1"]                     # (N, C, H, W)
    flat = x.reshape(x.shape[0], x.shape[1], -1)
    return {
        "out1": flat.mean(dim=(0, 2)),
        "out2": flat.std(dim=(0, 2)),
    }
```

### 標籤批次的類別分佈

一個 TENSOR 進、一個 STRING 出：

```python
import collections

def run(inputs, params):
    labels = inputs["in1"].flatten().tolist()
    counts = collections.Counter(labels)
    total = sum(counts.values())
    lines = [
        f"class {int(k)}: {v} ({100 * v / total:.1f}%)"
        for k, v in sorted(counts.items())
    ]
    print("\n".join(lines))          # 同時會進到執行紀錄
    return {"out1": "\n".join(lines)}
```

### 穩健統計摘要（中位數、四分位距、離群值數量）

```python
import statistics

def run(inputs, params):
    values = sorted(float(v) for v in inputs["in1"].flatten().tolist())
    q1, _, q3 = statistics.quantiles(values, n=4)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {
        "out1": {
            "median": statistics.median(values),
            "iqr": iqr,
            "outliers": sum(1 for v in values if v < low or v > high),
        }
    }
```

### 比較兩個張量

兩個 TENSOR 進（`input_ports: 2`）、一個 SCALAR 出：

```python
def run(inputs, params):
    a, b = inputs["in1"], inputs["in2"]
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return float((a - b).abs().max())     # 單一值 -> out1
```

### 在本次執行的裝置上建立張量

`device` 是本次執行解析出來的運算裝置，讓建立張量的腳本跟流程其他部分待在同一個裝置上：

```python
def run(inputs, params):
    return torch.zeros(4, 4, device=device)
```

## 限制

| | |
|---|---|
| 腳本長度 | 100,000 個字元。超過這個長度請改寫成自訂節點。 |
| 連接埠 | 每邊 1–8 個。 |
| 擷取的輸出 | 每次執行 64,000 個字元，超過後截斷並附註。 |
| 非同步 | `run` 必須是一般的 `def`；`async def run` 會在檢查階段被拒絕，`asyncio` 也無法匯入。 |
| 狀態 | 每次執行都會拿到全新的命名空間，所以你自己的模組層級變數不會延續，而函式庫代理也拒絕被寫入。真正需要保留的狀態請交給流程本身。 |
