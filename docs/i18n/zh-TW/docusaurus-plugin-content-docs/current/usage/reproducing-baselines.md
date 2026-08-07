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

從空白畫布的範例庫點開 **ResNet-18 / CIFAR-10 baseline**，或直接載入它的圖。這一頁提到的所有東西都在 [`examples/Usage_Example/ResNet18-CIFAR10-Baseline/`](https://github.com/CodefyUI/CodefyUI/tree/main/examples/Usage_Example/ResNet18-CIFAR10-Baseline) 底下 — `graph.json`、一份記錄實際跑了什麼的 `README.md`，還有放著原始指標匯出與曲線圖的 `evidence/` 目錄。

十九個節點，分成四條線匯進 `TrainingLoop`：

- **資料增強** — `RandomCrop` 接 `RandomHorizontalFlip` 接 `ToTensorTransform` 接 `NormalizeTransform`，接進訓練用 `Dataset` 的 `train_transform` 埠。轉換節點可以直接一個接一個串起來，不需要另外一個 Compose 節點。
- **評估前處理** — 只有 `ToTensorTransform` 接 `NormalizeTransform`，接進測試用 `Dataset` 的 `eval_transform` 埠。測試集永遠不做資料增強。
- **模型** — 一個 `SequentialModel`，它的層編輯器裡放著 ResNet-18：總共 70 個節點，其中 68 個是實際的層，另外 2 個是 `Input`/`Output` 標記。
- **優化** — `Optimizer`、`Loss` 和 `LRScheduler`。

`EvaluateModel` 讀訓練好的模型和測試集，回報 top-1 準確率。`CheckpointSaver` 存權重。`Visualize` 畫 loss 曲線。

:::tip 這個架構不要用手拉
68 個層節點加 77 條連線，不是任何人應該一個一個拖出來的東西 — 那是好幾百次不能出錯的操作，而且殘差捷徑接錯的時候不會有任何提示，要等到張量形狀對不上才會發現。

請改用層編輯器的 **Import**。它可以直接吃這個範例的 `graph.json`，一次把整個架構載進來；要在自己的圖裡重複使用這個 ResNet-18 也是同一條路：打開你的 `SequentialModel`、按 **Import**、選檔案。**Export** 會把同樣的格式寫回去。層編輯器適合拿來**看**架構和**改**架構，不適合拿來從零開始打字。
:::

## 配方

| | |
|---|---|
| 架構 | ResNet-18，CIFAR 版 — 3x3 stride-1 stem、沒有 maxpool、4 個 stage 各 2 個 BasicBlock（64/128/256/512），11,173,962 個參數 |
| 優化器 | SGD，lr 0.1，動量 0.9，Nesterov，weight decay 5e-4 |
| 學習率排程 | `CosineAnnealingLR`，`T_max` 等於總 epoch 數（見下方「值得先知道的坑」— 兩者不一致會出現提醒，但不會被擋下來） |
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
        "graph": {"nodes": [ ... ], "edges": [ ... ]},
        "options": {"device": "cuda", "seed": 1337, "deterministic": true}
      }'
```

把範例 `graph.json` 裡的 `nodes` 和 `edges` 陣列貼進去 — 請求本體帶的是整張圖，所以在貼上之前，這段指令直接複製去跑不會有任何效果。

## 實測結果

在下列硬體上，把附的圖原封不動跑兩次、用同一個 seed 量出來的。

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4080（16 GB，compute capability 8.9） |
| 軟體 | PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、Python 3.11 |
| Seed | 1337，`deterministic: true` |
| Epoch 數 | 200 |
| 實際耗時 | 從頭到尾 22 分 29 秒（平均每個 epoch 6.62 秒、含測試集那一輪，範圍大約 6.0 到 8.1 秒；剩下的半分鐘是載入資料集和最後那次評估） |
| **測試準確率（top-1）** | **95.48%** |
| 同 seed 重跑 | 95.48% — 差距 **0.00 個百分點** |

第二次是在伺服器重啟之後才送出的，所以跟第一次不共用任何行程狀態。全部 601 個記錄點 — 200 個訓練損失、200 個測試損失、200 個學習率，加上最後的準確率 — 全部**逐位元完全相同**。

能做到這樣靠的是 `deterministic: true`：它要求使用確定性的 kernel，並且不讓 cuDNN 在每次執行時重新挑卷積演算法。它是「盡力而為」而不是保證 — 遇到沒有確定性實作的運算，它會發出警告然後照跑 — 但在這個配方上它確實做到了完全一致，而且是跨兩個行程驗證過的。不開的話，同一個 seed 還是會得到很像的曲線，但兩次執行在數值上就會慢慢分岔。

關於資料切分，有一件事要老實說：這張圖沒有跟測試集分開的驗證集。測試集同時餵給 `TrainingLoop.val_dataloader` 和 `EvaluateModel`，這是 CIFAR-10 標準結果的慣常做法。但沒有任何東西會**根據它做選擇** — early stopping 是關掉的，存下來的檢查點是最後一個 epoch 的模型，不是驗證表現最好的那個 — 所以回報的數字是乾淨的 held-out 量測。曲線標的是「測試」而不是「驗證」也是這個原因。如果你要調超參數，請先從訓練集裡切一份驗證集出來。

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

要接續的話，在圖裡加一個 `CheckpointLoader`，指向那個檢查點，然後把它還原出來的四個輸出接進 `TrainingLoop`：

- `CheckpointLoader.model` 接 `TrainingLoop.model`
- `CheckpointLoader.optimizer` 接 `TrainingLoop.optimizer`
- `CheckpointLoader.lr_scheduler` 接 `TrainingLoop.lr_scheduler`
- `CheckpointLoader.epoch` 接 `TrainingLoop.start_epoch`

`TrainingLoop.epochs` 是絕對的目標值，不是「再跑幾個」，所以維持 200 就好，接續的 run 會從檢查點記錄的那個 epoch 的下一個開始。

如果你在意數字要完全對得起來，有一個邊界要知道：停止的時間點如果落在**某個 epoch 中間**，那個 epoch 已經做的權重更新會留在模型裡，但這個 epoch **不算完成**，因為它的平均損失還不完整。接續的 run 會從 batch 0 重跑那個 epoch。也就是說，學習率排程在停止前後是連續的，但資料掃過的次數不是。停過再接的 run 因此會很接近、但不會逐位元等於一次跑完的 run。在 epoch 之間才按停止就完全沒有這個問題。

:::warning 排程器要接**進** loader，不是繞過它
接法是 `Optimizer` 進 `LRScheduler`，再由 `LRScheduler` 進 `CheckpointLoader.lr_scheduler`。這個輸入是選填的，所以很容易漏接 — 而漏接才是真正會讓你損失東西的地方。

**問題不在 `base_lrs`。** 在已經還原過的優化器上建立排程器，起點一樣是 0.1：`Optimizer.state_dict()` 會把第一個排程器蓋在 param group 上的 `initial_lr` 一起存走，`load_state_dict` 會把它還原回來，而 `LRScheduler.__init__` 是用 `setdefault` 讀它，不會覆蓋掉既有的值。兩種接法都實際量過，跟理論上的 cosine 相符到 1.4e-17。

**接法真正決定的，是檢查點裡存的排程進度會被「還原」還是被「重建」。** `CheckpointLoader` 只能把 `scheduler_state_dict` 還原進有接到它身上的排程器。這個輸入沒接，存下來的狀態就會被丟掉（只留一行 log），排程改成用「從 `base_lrs` 重播 `start_epoch` 次 step」的方式重建。對 `CosineAnnealingLR` 來說重播是精確的 — 所以這一頁的配方兩種接法都安全 — 但由指標驅動的 `ReduceLROnPlateau` 沒辦法重播：它的 `best` 和 `num_bad_epochs` 會無聲歸零，本來下一個 epoch 就要觸發的衰減會被往後推。這個一般性問題記在 issue [#149](https://github.com/CodefyUI/CodefyUI/issues/149)。
:::

## 自己驗證數字

每個 run 的指標都可以查：

```bash
curl "http://127.0.0.1:8000/api/runs/<run_id>/metrics?format=csv" -o metrics.csv
```

`train_loss`、`val_loss` 和 `lr` 是每個 epoch 記一次。**`eval_accuracy` 不是** — `EvaluateModel` 只在 run 結束時寫一個點。所以 200 個 epoch 匯出來是 601 列而不是 800 列，而且沒有準確率對 epoch 的曲線可以畫：產品裡沒有任何節點會產生它。記在 issue [#202](https://github.com/CodefyUI/CodefyUI/issues/202)。

這個範例的 `TrainingLoop` 上 `tensorboard` 已經是打開的，所以每次執行也會在 run 的產物目錄下寫出 event 檔，任何 TensorBoard 都讀得起來。

## 值得先知道的坑

- **每個源頭節點都要接一條觸發線。** 執行是從 `Start` 出發、沿著**資料**線往前走，所以沒有輸入資料線的節點，除非有一條從 `Start` 來的觸發線指到它，否則會被跳過。這個範例剛好有四個這種節點，四條都接了：

  | 觸發線指向 | 為什麼它是源頭 |
  |---|---|
  | `RandomCrop` | 訓練資料增強鏈的頭 |
  | `ToTensorTransform`（評估那條） | 評估前處理鏈的頭 |
  | `SequentialModel` | 模型沒有資料輸入 |
  | `Loss` | 損失函數沒有資料輸入 |

  請注意這張表上**沒有**的東西：兩個 `Dataset` 節點都不是源頭。它們各自被自己的轉換鏈餵進來（`aug-norm` 進 `ds-train.train_transform`、`ev-norm` 進 `ds-test.eval_transform`），往前走的走訪自己就會走到它們 — 對 `Dataset` 接觸發線不會出錯，但也沒有作用。

  漏掉一條觸發線，執行就會失敗，而且錯誤訊息指的是下游很遠的另一個節點。舉例來說，漏掉指向 `SequentialModel` 的那一條，`Optimizer` 和 `LRScheduler` 會跟著一起被剪掉，然後執行會抱怨 `TrainingLoop`。記在 issue [#201](https://github.com/CodefyUI/CodefyUI/issues/201)。

- **`LRScheduler.T_max` 要跟 `TrainingLoop.epochs` 保持一致。** cosine 退火是每個 epoch 走一步，而且剛好在 `T_max` 歸零。`T_max` 設太大，訓練會停在曲線中段、退火退不完，大約會損失一個百分點的準確率；設太小則更麻煩 — 過了 `T_max` 之後 cosine 會**再往上爬**，最後幾個 epoch 反而是在學習率上升的情況下訓練。兩者不一致時 `TrainingLoop` 會提醒你（伺服器記錄檔、執行紀錄面板的執行記錄，以及畫布下方的記錄分頁各一份），但不會強制擋下來：截斷的排程本身是合理的選擇，而且 `CosineAnnealingWarmRestarts` 是拿同一個值當 `T_0`，在那裡「相等」反而代表永遠不會重啟。同一組檢查也涵蓋 `OneCycleLR.total_steps`，它的預設值 1000 是以批次為單位的數字，任何 epoch 設定都到不了。
- **`EvaluateModel` 不會跟著執行時的裝置走。** 它的 `device` 參數預設是 `cpu`，而且沒有 `auto` 可選。要自己設成 `cuda`，不然評估會很慢。記在 issue [#204](https://github.com/CodefyUI/CodefyUI/issues/204)。
- **第一次執行會下載 CIFAR-10**（大約 170 MB）。預設會放到 `backend/data/`；如果有開專案目錄，則會放到 `<專案>/assets/data`。之後就會重複使用。

CIFAR-10 資料集出自 Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*（2009）；它是在執行時下載的，CodefyUI 並沒有隨附散布它。
