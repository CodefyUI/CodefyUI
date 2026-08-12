---
sidebar_position: 3.7
title: 儲存與載入模型
description: ModelSaver 寫出什麼、ModelLoader 又願意讀回什麼 —— state_dict 與 full_model 的差別，以及為什麼載入模型檔是一個信任決定。
---

# 儲存與載入模型

`ModelSaver` 和 `ModelLoader`（兩者都在節點面板的 **IO** 分類）各有兩種模式，而你選的那一組決定了檔案「是什麼」：一袋數字，或者一個 Python 物件。這個差別不只是方便與否。載入一個已存的 Python 物件是唯一有機會執行程式碼的步驟，所以這也是 CodefyUI 裡唯一一處「你打開的檔案對接下來會發生什麼有發言權」的地方。

## 兩種模式

| | `state_dict`（預設） | `full_model` |
|---|---|---|
| 檔案裡有什麼 | 張量，以參數名稱為鍵 | 整個 `nn.Module`，以 pickle 序列化 |
| 載入時你需要 | 同一份架構，接到 `ModelLoader.model` | 什麼都不用 —— 模組會自己重建 |
| 支援 `.safetensors` | 支援 | 不支援（該格式只存張量） |
| 改層名稱後還能用 | 不能 —— 鍵就是屬性名稱 | 可以 |
| 在 CodefyUI 之外可讀 | 可以，任何能讀 torch 檔案的程式都行 | 只有在類別定義可以 import 時才行 |

**`state_dict` 是預設值，也是建議的路徑。** 所有隨附範例都用它。這個模式對檔案沒有附加任何條件：`torch.load` 在 torch 的受限解序列化器下讀取它，所以檔案裡不存在任何可能被執行的東西，而任何具備同一份架構的 Python 程式都能使用它。

**`full_model` 的用途是你不想重建架構的那些情況** —— 把一個檔案交給別人讓他直接跑，或是重新載入一個你已經沒有圖的模型。

## 為什麼 `full_model` 有限制

完整模型檔是一個 **pickle**，而解 pickle 不是「讀資料」：一個 pickle 可以指名一個函式並要求呼叫它。這就是為什麼 `weights_only=True` 是 torch 的預設值，而 CodefyUI 從不把它關掉。

取而代之的是，`ModelLoader` 只在那一次載入期間，把受限解序列化器放寬到剛好三組名稱：

1. **`torch.nn` 自己的層類別。** 用走訪 `nn.Module` 已載入子類別的方式推導出來，只留下 torch 自己定義的那些，所以它跟著你安裝的 torch 版本走，而不是跟著這裡寫死的一份清單走。
2. **CodefyUI 自己的模組類別** —— `GraphModelModule`（每一個層編輯器模型都是它）、`SequentialModel` 的包裝類別（`Reshape`、`SelectIndex`，以及 LSTM／GRU／注意力／transformer 區塊）、`CausalLMModule` 與構成它的那些區塊、diffusion U-Net、VLA 策略網路，還有其餘幾個。這是一份精選的、逐一列名的類別清單，每一個都依照下面那些規則審核過。
3. **兩個 torch 的啟動函式** —— `torch.nn.functional.relu` 與 `torch._C._nn.gelu`。torch 的 transformer 層把自己的啟動函式存成一個*可呼叫物件*屬性，而不是存成一個層，所以一個 `TransformerEncoder` 或 `TransformerDecoder` 的檢查點要載回來，就得靠這兩個名稱。這裡放行的是逐一列名的身分，不是整個 `torch.nn.functional` 命名空間：`handle_torch_function` 也住在那裡，而它會轉派到任意物件的 `__torch_function__`，所以放行整個命名空間等於放行一個通用的呼叫工具，也等於把 torch 以後往那裡新增的任何東西一起放行。

其他一切都會被拒絕，並附上一則指名它停在哪裡的訊息。一個指名 `os.system` 的 pickle 載不進來，因為 `os.system` 不在上面這三組裡的任何一組 —— 而除了那兩個之外的任何函式也一樣不在。

### 「審核過」到底是什麼意思

把一個**類別**列名放行之後，檔案可以對它做兩件事，而這兩件事都必須是無害的：

- **還原它的屬性。** 所以被放行的類別不能定義 `__reduce__`、`__setstate__` 或 `__getnewargs__` —— 任何會把「還原一個屬性」變成「執行某段程式」的東西都不行。
- **用檔案自己挑的參數呼叫它的建構子。** torch 的受限解序列化器對任何被允許的名稱都會執行 `func(*args)`，所以 `cls(...)` 是可以被觸發的。因此被放行的建構子不能碰檔案、不能碰網路、也不能碰全域狀態（用區域的 `torch.Generator` 沒問題，`torch.manual_seed` 不行）。參數亂給導致丟出錯誤是可以接受的 —— 那是一次失敗的載入，不是一次被入侵的載入。

第二點對 torch 自己的類別一直都成立：放行 `nn.Linear` 就等於放行「用檔案挑的尺寸呼叫 `nn.Linear(...)`」。之所以值得寫出來，是因為 CodefyUI 這份清單是由人維護的；而這兩點裡可以用機械方式檢查的部分，每次跑測試都會被驗證。

放行一個**函式**則看四點，四點都必須成立：它是 torch 自己的、它是一個純粹的張量運算、它沒有檔案／網路／行程層面的副作用、它不改動任何全域狀態 —— 這樣一來，用檔案自己挑的任意參數呼叫它，結果只會是回傳一個張量或是丟出錯誤。這是同一套標準，而且它旁邊那個類別的情況其實是*更大*的表面：檔案本來就能經由同一條程式路徑觸發 `nn.Linear(...)`，而一個函式既沒有建構子，也沒有要還原的屬性。

這兩份清單都是從**存檔那一端**列舉出來的 —— 從「CodefyUI 實際上能組出來的模型會存下什麼」列舉 —— 而且兩份都各有一個測試會把那次列舉重跑一遍。所以，當某個類別或某個可呼叫物件開始出現在被存下的模型裡時，失敗的會是測試，而不是安靜地變成一個沒有人打得開的檢查點。

:::note 實際上這代表什麼
CodefyUI 寫出的 `full_model` 檔**可以在 CodefyUI 裡載回來**。但含有[自訂節點](/advanced/custom-nodes)、[外掛](/advanced/plugins)或別人自己腳本裡類別的 `full_model` 檔**不行** —— 那些程式碼沒有經過審查，而放寬到它們是 CodefyUI 不跨越的那條線。`ModelSaver` 會在存檔當下就在它的 **Log** 分頁告訴你剛剛寫出的是哪一種，而不是讓你晚一個節點才發現。
:::

### 一個已知的邊界

- **CodefyUI 寫出的檔案不是自足的。** 讀它需要 CodefyUI 自己的類別可以 import，所以比「開始放寬的那個版本」更舊的 CodefyUI 會拒絕它，而在 CodefyUI 之外用純 torch 讀則需要 `weights_only=False` 加上後端套件在 `sys.path` 上。`state_dict` 檔案這兩個條件都沒有。（只由 torch 原生層組成的檔案 —— 包含 transformer —— 這兩個條件也都沒有；`Log` 那則提示會告訴你剛剛寫出的是哪一種。）

## 如果載入被拒絕，而你信任那個檔案

在 CodefyUI 之外轉換一次，然後把結果當成 `state_dict` 載入：

```python
import torch
model = torch.load("PATH.pt", weights_only=False)   # 這一行就是執行程式碼的那一步
torch.save(model.state_dict(), "NEW_PATH.pt")
```

只對你自己產生的、或來源你信任的檔案這樣做。那個 `weights_only=False` 正是 CodefyUI 不會替你按下的那一步 —— 所以它是一行你自己打出來的程式碼，而不是應用程式裡的一個勾選框。

## 相關頁面

- [重現基準結果](./reproducing-baselines) —— `CheckpointSaver` / `CheckpointLoader`，它們儲存的是訓練*狀態*（模型、最佳化器、排程、epoch）而不是一個模型，而且一律以張量形式儲存。
- [執行圖](./running-graphs) —— 為什麼這兩個節點永遠不會由執行快取代為回答。
- [Python Script 節點](/advanced/python-script-node) —— `torch.load` 在那裡是直接被拒絕的，以及為什麼。
