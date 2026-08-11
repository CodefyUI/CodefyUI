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

取而代之的是，`ModelLoader` 只在那一次載入期間，把受限解序列化器放寬到剛好兩組名稱：

1. **`torch.nn` 自己的層類別。** 用走訪 `nn.Module` 已載入子類別的方式推導出來，只留下 torch 自己定義的那些，所以它跟著你安裝的 torch 版本走，而不是跟著這裡寫死的一份清單走。
2. **CodefyUI 自己的模組類別** —— `GraphModelModule`（每一個層編輯器模型都是它）、`SequentialModel` 的包裝類別（`Reshape`、`SelectIndex`，以及 LSTM／GRU／注意力／transformer 區塊）、`CausalLMModule` 與構成它的那些區塊、diffusion U-Net、VLA 策略網路，還有其餘幾個。這是一份精選的、逐一列名的類別清單，每一個都經過閱讀確認：重建它只會還原屬性，不會執行任何東西。

其他一切都會被拒絕，並附上一則指名它停在哪裡的訊息。一個指名 `os.system` 的 pickle 載不進來，因為 `os.system` 不在上面任何一組裡。

:::note 實際上這代表什麼
CodefyUI 寫出的 `full_model` 檔**可以在 CodefyUI 裡載回來**。但含有[自訂節點](/advanced/custom-nodes)、[外掛](/advanced/plugins)或別人自己腳本裡類別的 `full_model` 檔**不行** —— 那些程式碼沒有經過審查，而放寬到它們是 CodefyUI 不跨越的那條線。`ModelSaver` 會在存檔當下就在它的 **Log** 分頁告訴你剛剛寫出的是哪一種，而不是讓你晚一個節點才發現。
:::

### 兩個已知的邊界

- **`TransformerEncoder` 或 `TransformerDecoder` 層目前還是載不回來。** 原因不在包裝類別 —— 它在清單上 —— 而在於 `nn.TransformerEncoderLayer` 把它的啟動函式存成 `torch.nn.functional.relu`，那是一個*函式*，而清單上沒有任何函式。放寬到函式等於放寬到「解序列化器可能被要求呼叫」的可呼叫物件，那是一個比類別更大的決定，目前還沒有做。這些層請用 `state_dict`。
- **CodefyUI 寫出的檔案不是自足的。** 讀它需要 CodefyUI 自己的類別可以 import，所以比「開始放寬的那個版本」更舊的 CodefyUI 會拒絕它，而在 CodefyUI 之外用純 torch 讀則需要 `weights_only=False` 加上後端套件在 `sys.path` 上。`state_dict` 檔案這兩個條件都沒有。

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
