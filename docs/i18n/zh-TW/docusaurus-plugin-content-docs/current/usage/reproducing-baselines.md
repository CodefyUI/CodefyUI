---
sidebar_position: 3.9
title: 重現標準結果
description: 研究等級的完整走訪 — 用純 GUI 節點、固定 seed、丟上執行佇列，從頭到尾重現 ResNet-18 / CIFAR-10 的標準結果。
---

# 重現標準結果

這份文件大部分在講「怎麼學」。這一頁講的是另一件事：你在畫布上接出來的圖，能不能產生一個你敢寫進論文的數字，而且別人能照著重新算出來。

拿來示範的是 CIFAR-10 的標準結果 — ResNet-18、SGD 加動量、cosine 退火、裁切加翻轉的資料增強。它是影像分類重現性的「hello world」，準確率是大家都知道的常識，而這正是它適合當測試的原因：有一個公開的數字可以被錯過。

以下完全不需要寫 Python。架構、優化器、學習率排程、資料增強鏈，全部都是節點。

## 這個範例

從空白畫布的範例庫點開 **ResNet-18 / CIFAR-10 baseline**，或載入 `examples/Usage_Example/ResNet18-CIFAR10-Baseline/graph.json`。

十九個節點，分成四條線匯進 `TrainingLoop`：

- **資料增強** — `RandomCrop` 接 `RandomHorizontalFlip` 接 `ToTensorTransform` 接 `NormalizeTransform`，接進訓練用 `Dataset` 的 `train_transform` 埠。轉換節點可以直接一個接一個串起來，不需要另外一個 Compose 節點。
- **評估前處理** — 只有 `ToTensorTransform` 接 `NormalizeTransform`，接進測試用 `Dataset` 的 `eval_transform` 埠。測試集永遠不做資料增強。
- **模型** — 一個 `SequentialModel`，它的層編輯器裡放著 70 層的 ResNet-18。
- **優化** — `Optimizer`、`Loss` 和 `LRScheduler`。

`EvaluateModel` 讀訓練好的模型和測試集，回報 top-1 準確率。`CheckpointSaver` 存權重。`Visualize` 畫 loss 曲線。

## 配方

| | |
|---|---|
| 架構 | ResNet-18，CIFAR 版 — 3x3 stride-1 stem、沒有 maxpool、4 個 stage 各 2 個 BasicBlock（64/128/256/512），11,173,962 個參數 |
| 優化器 | SGD，lr 0.1，動量 0.9，Nesterov，weight decay 5e-4 |
| 學習率排程 | `CosineAnnealingLR`，`T_max` 等於總 epoch 數 |
| 損失函數 | 交叉熵，不做 label smoothing |
| 批次大小 | 訓練 128，評估 512 |
| Epoch 數 | 200 |
| 資料增強 | `RandomCrop(32, padding=4)`、`RandomHorizontalFlip(p=0.5)` |
| 正規化 | CIFAR-10 各通道統計值 — 平均 (0.4914, 0.4822, 0.4465)、標準差 (0.2470, 0.2435, 0.2616) |
| 精度 | bf16 autocast |

### 為什麼 stem 不一樣

`torchvision` 裡的 ResNet-18 是為 224x224 的 ImageNet 圖片設計的：先做 7x7 stride-2 卷積，再接 3x3 stride-2 的 max-pool，等於在進第一個殘差塊之前就把輸入縮小成四分之一。這套用在 32x32 的 CIFAR 圖片上，等於一開始就把空間資訊丟得差不多了，同樣的配方會低好幾個百分點。

CIFAR 版把整個 stem 換成單一個 3x3 stride-1 卷積，並且拿掉 max-pool，所以第一個 stage 看到的還是 32x32。所有公開的 CIFAR ResNet 數字都預設是這樣做的。「ResNet-18 跑 CIFAR-10」重現出來只有 91% 而不是 95%，最常見的原因就是這一點。

你可以在層編輯器裡看到這個 stem：雙擊 `SequentialModel`，前三層就是 `Conv2d(3, 64, kernel_size=3, stride=1, padding=1)`、`BatchNorm2d(64)`、`ReLU`。

## 怎麼跑才能重現

關鍵是兩個執行選項，都是送出 run 的時候設定，不是設在任何節點上：

- **`seed`** — 一個整數就決定全部。權重初始化、訓練 loader 的洗牌順序、資料增強的隨機抽樣，都是從它逐一推導出來的，所以結果不會因為哪個節點先執行而改變。
- **`deterministic`** — 要求使用確定性的 kernel，並關掉 cuDNN 的演算法自動選擇。不開的話，cuDNN 每次執行會重新挑卷積演算法，同樣 seed 的兩次執行就會慢慢分岔。在這個工作量下大約損失 4% 的速度，換到一個完全可重複的數字，很划算。

從執行面板送出，或用 API：

```bash
curl -X POST http://127.0.0.1:8000/api/runs \
  -H "Content-Type: application/json" \
  -H "X-CodefyUI-Token: $CODEFYUI_TOKEN" \
  -d '{
        "name": "resnet18-cifar10-seed1337",
        "graph": {"nodes": [], "edges": []},
        "options": {"device": "cuda", "seed": 1337, "deterministic": true}
      }'
```

（`nodes` 和 `edges` 就是範例 `graph.json` 的內容。）

## 實測結果

在下列硬體上，把附的圖原封不動跑兩次、用同一個 seed 量出來的。

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4080（16 GB，compute capability 8.9） |
| 軟體 | PyTorch 2.11.0+cu128、torchvision 0.26.0、Python 3.11 |
| Seed | 1337，`deterministic: true` |
| Epoch 數 | 200 |
| 實際耗時 | 22 分 29 秒（平均每個 epoch 6.62 秒，含測試集那一輪） |
| **測試準確率（top-1）** | **95.48%** |
| 同 seed 重跑 | 95.48% — 差距 **0.00 個百分點** |

第二次是在伺服器重啟之後才送出的，所以跟第一次不共用任何行程狀態。全部 601 個記錄點 — 200 個訓練損失、200 個測試損失、200 個學習率，加上最後的準確率 — 全部**逐位元完全相同**。

這種完全一致就是 `deterministic: true` 換來的。不開的話，同一個 seed 還是會得到很像的曲線，但 cuDNN 會重新挑卷積演算法，兩次執行在數值上就會慢慢分岔。

換**不同** seed 的話，準確率大概會有正負 0.3 個百分點的差距；那是這個配方本身的隨機性，不是壞掉。要抓緊的是同一個 seed。

如果你的數字是低好幾個百分點、而不是低零點幾，先去檢查 stem，再檢查其他東西。

## 關掉瀏覽器也沒關係

一個 run 屬於伺服器，不屬於送出它的那個分頁。這件事值得自己試一次，之後才敢信：

1. 送出 run。
2. 關掉分頁。
3. 過一陣子再打開畫布。

執行面板會重新接上還在跑的工作，並且把中間錯過的事件補播回來。什麼都沒掉，你離開的期間它也沒有停。通道和並行上限請看[執行佇列](./run-queue.md)。

## 停止與接續

按**停止**是「合作式」的：訓練節點會把手上這個 batch 做完、寫一個檢查點、然後回傳它的部分結果。這個檢查點會登記成 run 的產物，中繼資料裡記著**已完成**的 epoch 數。

要接續的話，在圖裡加一個 `CheckpointLoader`，指向那個檢查點，然後把三個還原出來的值繞過它：

- `CheckpointLoader.model` 接 `TrainingLoop.model`
- `CheckpointLoader.optimizer` 接 `TrainingLoop.optimizer`
- `CheckpointLoader.lr_scheduler` 接 `TrainingLoop.lr_scheduler`
- `CheckpointLoader.epoch` 接 `TrainingLoop.start_epoch`

`TrainingLoop.epochs` 是絕對的目標值，不是「再跑幾個」，所以維持 200 就好，接續的 run 會從停下來的地方繼續。

:::warning 排程器要接在 loader 前面，不是後面
接法是 `Optimizer` 進 `LRScheduler`，再由 `LRScheduler` 進 `CheckpointLoader` — 不是 `Optimizer` 進 `CheckpointLoader` 再進 `LRScheduler`。

PyTorch 的排程器會在**建立當下**從優化器抓走 `base_lrs`。如果 loader 已經先把衰減過的學習率還原到優化器上，之後才建立的排程器就會把那個衰減值當成起點，等於又退火了第二次。先用乾淨的優化器建立排程器、再還原它的狀態，`base_lrs` 才會維持在 0.1。這個問題記在 issue #149。
:::

## 自己驗證數字

每個 run 的指標都可以查，`train_loss`、`val_loss`、`lr` 和 `eval_accuracy` 都是逐 epoch 記錄的：

```bash
curl "http://127.0.0.1:8000/api/runs/<run_id>/metrics?format=csv" -o metrics.csv
```

在 `TrainingLoop` 上把 `tensorboard` 打開，還會在 run 的產物目錄下寫出 event 檔，任何 TensorBoard 都讀得起來。

## 值得先知道的坑

- **每個源頭節點都要接一條觸發線。** 執行是從 `Start` 出發、沿著資料線往前走，所以沒有輸入資料線的節點 — `Loss`、每個 `Dataset`、每條轉換鏈的頭 — 除非 `Start` 指到它，否則會被跳過。漏掉一個，執行就會失敗，而且錯誤訊息指的是下游很遠的另一個節點。附的範例四條都接了。記在 issue #201。
- **`EvaluateModel` 不會跟著執行時的裝置走。** 它的 `device` 參數預設是 `cpu`，而且沒有 `auto` 可選。要自己設成 `cuda`，不然評估會很慢。
- **第一次執行會下載 CIFAR-10**（大約 170 MB）到 `backend/data/`。之後就會重複使用。
