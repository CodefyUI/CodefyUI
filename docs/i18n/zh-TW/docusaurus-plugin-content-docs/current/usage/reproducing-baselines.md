---
sidebar_position: 3.9
title: 重現標準結果
description: 研究等級的完整說明 — 使用 GUI 節點、固定 seed 與執行佇列，從頭到尾重現標準 ResNet-18 / CIFAR-10 結果。
---

# 重現標準結果

多數文件著重於學習。本頁說明另一項能力：在畫布上建立的圖可以產生適合寫入論文的數值，而且其他人能依相同步驟重新推導。

本頁使用標準 CIFAR-10 baseline：ResNet-18、帶動量的 SGD、cosine 退火，以及裁切與翻轉資料增強。這是常用的影像分類重現性基準，預期準確率廣為人知，因此適合檢查結果是否偏離公開數值。

以下步驟不需要撰寫 Python。架構、優化器、學習率排程與資料增強鏈都由節點組成。

## 這個範例

從空白畫布的**範例圖庫**開啟 **ResNet-18 / CIFAR-10 baseline**，或直接載入圖檔。本頁使用的檔案都位於 [`examples/Usage_Example/ResNet18-CIFAR10-Baseline/`](https://github.com/CodefyUI/CodefyUI/tree/main/examples/Usage_Example/ResNet18-CIFAR10-Baseline)：`graph.json`、記錄實際執行內容的 `README.md`，以及包含原始指標匯出與曲線圖的 `evidence/` 目錄。

圖中有 19 個節點，分為四組並連到 `TrainingLoop`：

- **資料增強** — `RandomCrop` → `RandomHorizontalFlip` → `ToTensorTransform` → `NormalizeTransform`，連到訓練 `Dataset` 的 `train_transform` port。轉換節點可直接串接，不需要另外使用 Compose。
- **評估前處理** — 只有 `ToTensorTransform` → `NormalizeTransform`，連到測試 `Dataset` 的 `eval_transform` port。測試集不使用資料增強。
- **模型** — 一個 `SequentialModel`，其 layer editor 包含 ResNet-18 圖：共 70 個節點，其中 68 個是模組，另外 2 個是 `Input`／`Output` 標記。
- **優化** — `Optimizer`、`Loss` 與 `LRScheduler`。

`EvaluateModel` 讀取已訓練模型與 test split，並回報 top-1 準確率。`CheckpointSaver` 寫入權重，`Visualize` 繪製 loss 曲線。

:::tip 不要手動放置架構
手動拖曳 68 個層節點並建立 77 條連線，需要數百次精確操作。殘差捷徑若連接錯誤，通常要到 tensor shape 不相容時才會發現。

請改用層編輯器的**匯入**。它可直接讀取此範例的 `graph.json`，一次載入完整架構。若要在自己的圖中重用這個 ResNet-18，請開啟 `SequentialModel`、點擊**匯入**並選擇檔案。**匯出**會寫出相同格式。層編輯器適合檢視及調整架構，不適合從零手動建立整個架構。
:::

## 基準設定 {/* #配方 */}

| | |
|---|---|
| 架構 | ResNet-18，CIFAR 版 — 3x3 stride-1 stem、沒有 maxpool、4 個 stage 各 2 個 BasicBlock（64/128/256/512），11,173,962 個參數 |
| 優化器 | SGD，lr 0.1，動量 0.9，Nesterov，weight decay 5e-4 |
| 學習率排程 | `CosineAnnealingLR`，`T_max` 等於 epoch 數（見下方注意事項；不一致時會發出警告，但不會強制限制） |
| 損失函數 | 交叉熵，不做 label smoothing |
| 批次大小 | 訓練 128，評估 512 |
| Epoch 數 | 200 |
| 資料增強 | `RandomCrop(32, padding=4)`、`RandomHorizontalFlip(p=0.5)` |
| 正規化 | CIFAR-10 各通道統計值 — 平均 (0.4914, 0.4822, 0.4465)、標準差 (0.2470, 0.2435, 0.2616) |
| 精度 | bf16 autocast |

### 為什麼 stem 不一樣

`torchvision` 的 ResNet-18 是為 224x224 ImageNet 圖片設計：先使用 7x7 stride-2 卷積，再使用 3x3 stride-2 max-pool，在第一個殘差塊前將輸入尺寸縮小四倍。若套用於 32x32 CIFAR 圖片，會在一開始移除大部分空間資訊，使相同設定的準確率降低數個百分點。

CIFAR 版本以單一 3x3 stride-1 卷積取代整個 stem，並移除 max-pool，讓第一個 stage 仍接收 32x32 輸入。公開的 CIFAR ResNet 結果都採用這項設定。重現「ResNet-18 on CIFAR-10」時得到 91% 而不是 95%，最常見的原因就是使用了錯誤的 stem。

可在層編輯器中查看 stem。雙擊 `SequentialModel`，前三層是 `Conv2d(3, 64, kernel_size=3, stride=1, padding=1)`、`BatchNorm2d(64)`、`ReLU`。

## 可重現的執行方式 {/* #怎麼跑才能重現 */}

有兩個 run 選項負責這件事，都在送出 run 時設定：

- **`seed`** — 使用一個整數為所有亂數來源設定 seed。權重初始化、訓練 loader 的洗牌順序與資料增強抽樣都會依節點從這個值推導，因此結果不受節點執行順序影響。
- **`deterministic`** — 要求使用決定性 kernel，並停用 cuDNN 演算法 autotuner。若未啟用，cuDNN 會在每個 run 重新選擇卷積演算法，使使用相同 seed 的兩個 run 逐漸產生數值差異。此工作負載的 throughput 約降低 4%，以換取完全可重複的結果。`TrainingLoop` 的**進階**區段也有 `deterministic` 開關；任一處啟用都會套用到整個 run。

在畫布上，請於**設定 → 訓練行為**設定**亂數種子**與**決定性演算法**，再點擊**執行**。也可以從終端機將圖檔送到執行中的伺服器：

```bash
cdui run examples/Usage_Example/ResNet18-CIFAR10-Baseline/graph.json \
  --name resnet18-cifar10-seed1337 --device cuda --seed 1337 --deterministic
```

也可以透過 API：

```bash
curl -X POST http://127.0.0.1:8000/api/runs \
  -H "Content-Type: application/json" \
  -H "X-CodefyUI-Token: $(cat ~/CodefyUI/.codefyui_dev/session.token)" \
  -d '{
        "name": "resnet18-cifar10-seed1337",
        "graph": {"nodes": [ ... ], "edges": [ ... ]},
        "options": {"device": "cuda", "seed": 1337, "deterministic": true}
      }'
```

請用範例 `graph.json` 的 `nodes` 與 `edges` 陣列取代片段中的預留內容。請求本體必須包含完整圖形，因此在貼入這些內容之前，這個片段不會執行任何圖。token 檔案由伺服器在啟動時寫入；位置請見[取得 token](./graph-as-a-function#2-getting-the-token-for-external-scripts)。

## 實測結果

以下結果是在指定硬體上，以未修改的隨附圖和相同 seed 執行兩次所測得。

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4080（16 GB，compute capability 8.9） |
| 軟體 | PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、Python 3.11 |
| Seed | 1337，`deterministic: true` |
| Epoch 數 | 200 |
| 實際耗時 | 從頭到尾 22 分 29 秒（平均每個 epoch 6.62 秒，含測試那一輪，範圍 6.0 到 8.1 秒；其餘半分鐘是載入資料集與最後的評估） |
| **測試準確率（top-1）** | **95.48%** |
| 同 seed 重跑 | 95.48% — 差距 **0.00 個百分點** |

第二個 run 在伺服器重啟後送出，因此沒有與第一個 run 共用行程狀態。601 個指標點全部**逐位元相同**：200 個訓練損失、200 個測試損失、200 個學習率，以及最後的準確率。

`deterministic: true` 會要求決定性 kernel，並防止 cuDNN 在不同 run 之間重新選擇卷積演算法。這項要求會盡力執行，但不提供保證；缺少決定性 kernel 的運算會發出警告並繼續。不過，此基準設定已跨兩個行程驗證為完全一致。若未啟用，相同 seed 仍會產生相似曲線，但兩個 run 的數值會逐漸分離。

資料切分有一項限制：這張圖沒有與測試集分開的驗證集。測試集同時提供給 `TrainingLoop.val_dataloader` 與 `EvaluateModel`，這是 CIFAR-10 baseline 的常見配置。它不會用於選擇模型：early stopping 已關閉，儲存的是最後一個 epoch 的模型，而不是驗證結果最佳的模型。因此，回報值仍是有效的獨立測試量測，曲線也基於這項原因標示為「測試」而不是「驗證」。如果要調整超參數，請先從訓練集切出驗證集。

不同 seed 的結果通常相差約正負 0.3 個百分點。這是此基準設定在不同 run 之間的隨機波動，不是缺陷；相同 seed 的結果才應接近一致。

若結果低數個百分點，而不是只低零點幾個百分點，請先檢查 stem。

## 離開正在執行的工作 {/* #關掉瀏覽器也沒關係 */}

run 由伺服器持有，不依附於送出它的瀏覽器分頁。這一點值得親自試一次，確認可以信任：

1. 送出 run。
2. 關閉分頁。
3. 稍後重新開啟畫布。

**執行任務**面板會重新連線到仍在執行的工作，並重播錯過的事件。離開期間不會遺失資料，也不會暫停 run。通道與並行上限請見[執行佇列](./run-queue.md)。

## 停止與接續

點擊**停止**會採用協作式取消：訓練節點完成目前的 batch、寫入 checkpoint，再回傳部分結果。checkpoint 會登記為 run 的產出檔案，其中的 metadata 會記錄*已完成*的 epoch 數。

若要繼續執行，請在圖中加入 `CheckpointLoader`、指定該 checkpoint，再將四個還原輸出連到 `TrainingLoop`：

- `CheckpointLoader.model` 接 `TrainingLoop.model`
- `CheckpointLoader.optimizer` 接 `TrainingLoop.optimizer`
- `CheckpointLoader.lr_scheduler` 接 `TrainingLoop.lr_scheduler`
- `CheckpointLoader.epoch` 接 `TrainingLoop.start_epoch`

`TrainingLoop.epochs` 是絕對目標，不是要再執行的 epoch 數。維持 200 後，接續的 run 會從 checkpoint 所記錄 epoch 的下一個開始。

若需要精確重現數值，必須注意停止時機。**在 epoch 中途**停止時，該 epoch 已完成的權重更新會保留在模型中，但因 loss average 尚未完成，這個 epoch 不會計入完成數。接續的 run 會從 batch 0 重新執行該 epoch。學習率排程在停止前後保持連續，但資料遍歷不連續。因此，停止後再繼續的 run 會非常接近未中斷 run，卻不會逐位元相同。在 epoch 之間停止可避免這項差異。

:::warning 將排程器接*進* loader，不要繞過它
請將 `Optimizer` 連到 `LRScheduler`，再將 `LRScheduler` 連到 `CheckpointLoader.lr_scheduler`。這個輸入是選用項目，很容易漏接，而漏接才是真正有代價的地方。

**問題不在 `base_lrs`。** 在已還原的優化器上建立排程器時，起始值仍為 0.1。`Optimizer.state_dict()` 包含第一個排程器寫入 param group 的 `initial_lr`；`load_state_dict` 會還原它，而 `LRScheduler.__init__` 使用 `setdefault` 讀取，不會覆寫既有值。兩種接線方式都已與理論 cosine 曲線比較，誤差只有 1.4e-17。

**接線方式決定 checkpoint 中儲存的排程位置會還原或重建。** `CheckpointLoader` 只能將 `scheduler_state_dict` 還原到與它連接的排程器。若這個輸入未連接，儲存的狀態會被捨棄，排程會改為從 `base_lrs` 重播 `start_epoch` 個 step 來重建。`CosineAnnealingLR` 可以精確重播，因此本頁的基準設定在兩種接線方式下都安全。由指標驅動的 `ReduceLROnPlateau` 無法重播；其 `best` 與 `num_bad_epochs` 會重設，可能把原本將於下一個 epoch 發生的衰減延後。

你不必事先知道這一點。這兩個節點都會在伺服器 log、**執行任務**面板顯示的事件紀錄，以及畫布的**執行紀錄**中回報這項狀況。`CheckpointLoader` 會指出已捨棄儲存的排程位置，並列出應連接的輸入；`TrainingLoop` 會指出無法還原的排程。
:::

## 自行驗證數值 {/* #自己驗證數字 */}

每個 run 的指標都可查詢：

```bash
curl "http://127.0.0.1:8000/api/runs/<run_id>/metrics?format=csv" -o metrics.csv
```

`train_loss`、`val_loss` 與 `lr` 每個 epoch 各記錄一次。**`eval_accuracy` 不會如此記錄**；`EvaluateModel` 只在 run 結束時寫入一個資料點。因此，200 epoch 的匯出檔有 601 列，而不是 800 列，也無法繪製 accuracy-by-epoch 曲線，因為產品中沒有節點會發出這項資料。相關追蹤項目為 issue [#202](https://github.com/CodefyUI/CodefyUI/issues/202)。

此範例的 `TrainingLoop` 已將 `tensorboard` 設為 `true`，因此每個 run 也會在其產出目錄中寫入 event 檔案，可由任何 TensorBoard 安裝讀取。

## 注意事項 {/* #值得先知道的坑 */}

- **Trigger 只標示執行起點。** 在此範例中，`Start` 會 trigger `RandomCrop`、評估用的 `ToTensorTransform`、`SequentialModel` 與 `Loss`。不過，只要 root 透過 data edge 連到正在執行的節點，就會執行，不論是否有 trigger 指向它；前述四個節點與兩個 `Dataset` 節點都符合這項條件。詳見[執行圖](./running-graphs#沒有-trigger-的節點仍然可能執行)。因此，移除四條 trigger edge 中的任一條都不會改變執行內容；要排除節點，必須中斷其 data edge。

- **保持 `LRScheduler.T_max` 與 `TrainingLoop.epochs` 相等。** cosine annealing 每個 epoch 前進一步，並在 `T_max` 時降至零。`T_max` 過高時，run 會在曲線尚未完成前結束，無法完整 anneal，準確率約降低一個百分點。`T_max` 過低時，cosine 在超過 `T_max` 後會再次**上升**，使最後幾個 epoch 使用逐漸提高的 learning rate。兩者不一致時，`TrainingLoop` 會在伺服器 log、**執行任務**面板顯示的事件紀錄，以及畫布的**執行紀錄**中發出警告，但不會強制要求相等。截短 schedule 是有效選擇；此外，`CosineAnnealingWarmRestarts` 會將相同值用作 `T_0`，若要求它與 epoch 數相等，就不會發生 restart。相同檢查也適用於 `OneCycleLR.total_steps`；其預設值 1000 代表 batch 數量，沒有 epoch budget 會達到這個數字。以上說明假設使用預設的 `TrainingLoop.scheduler_step = epoch`，本 baseline 也使用這項設定。若改為 `optimizer_step`，`LRScheduler` 上的所有長度都會改以 optimizer step 計算，警告也會將它們與 run 的 step budget 比較，而不是與 `epochs` 比較。
- **`EvaluateModel.device` 預設為 `auto`**，並跟隨 run device。隨附的圖將它固定為 `cuda`，但不是必要設定。
- **第一次執行會下載 CIFAR-10**（約 170 MB）。預設位置是 `backend/data/`；開啟專案目錄時，位置是 `<project>/assets/data`。後續 run 會重用資料。

CIFAR-10 資料集來自 Krizhevsky 的 *Learning Multiple Layers of Features from Tiny Images*（2009）。資料集會在 run 時下載，不會隨 CodefyUI 散布。
