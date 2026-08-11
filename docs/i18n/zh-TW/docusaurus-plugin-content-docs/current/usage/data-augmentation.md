---
sidebar_position: 3.8
title: 資料與資料增強
description: 用節點組出前處理流程、對訓練資料做增強、載入自己的影像，並把指標送到 TensorBoard。
---

# 資料與資料增強

在 CodefyUI 1.5 之前，一個資料集的前處理選項只有三個：縮放、`ToTensor`，以及寫死成 `mean=0.5, std=0.5` 的正規化。這足以讓圖跑起來，卻不足以重現任何一篇論文的視覺任務結果。這一頁講的就是取而代之的東西。

這裡的一切都不會動到既有的圖。舊的 **Transform** 節點行為完全不變，參數也一樣。

## 變換鏈

現在變換本身就是節點，而且節點之間可以互相連接。每個節點接收它前面的所有步驟，再把「到目前為止的步驟」往下傳，所以你在畫布上畫出來的鏈，就是實際執行的順序：

```
RandomCrop -> RandomHorizontalFlip -> ToTensorTransform -> NormalizeTransform
```

它們之間的連線是 `TRANSFORM` 型別的埠，畫成琥珀色。它只能接到其他 `TRANSFORM` 埠，或是接到 `ANY` 埠，所以流程不可能被誤接到需要資料集的地方。

這條鏈產生的正是 `transforms.Compose([RandomCrop(...), RandomHorizontalFlip(...), ToTensor(), Normalize(...)])` — 和你手寫出來的物件一模一樣，匯出的 Python 腳本建出來的也是同一個東西。

### 順序很重要，torchvision 的規則依然適用

- **幾何與色彩步驟放最前面**，此時樣本還是 PIL 影像：`RandomCrop`、`RandomHorizontalFlip`、`RandomRotation`、`ColorJitter`、`RandAugment`、`ResizeTransform`。
- **`ToTensorTransform` 放中間。** 它會轉成範圍 `[0, 1]` 的 `C x H x W` 浮點張量。
- **`NormalizeTransform` 放最後。** 它需要張量。

順序錯掉的鏈會在 DataLoader 裡失敗，並顯示 torchvision 自己的錯誤訊息，訊息會指出是哪一個步驟收到了它無法處理的輸入。

### 節點一覽

| 節點 | 作用 | 主要參數 |
| --- | --- | --- |
| `ResizeTransform` | 縮放成正方形 | `size`；`interpolation`（進階） |
| `ToTensorTransform` | PIL 影像轉成 `[0, 1]` 張量 | — |
| `NormalizeTransform` | 每個通道做 `(x - mean) / std` | `preset`、`mean`、`std` |
| `RandomCrop` | 先補邊，再隨機取一個視窗 | `size`、`padding`；`padding_mode`（進階） |
| `RandomHorizontalFlip` | 左右鏡射 | `p` |
| `RandomRotation` | 旋轉一個隨機角度 | `degrees`；`expand`、`fill`（進階） |
| `ColorJitter` | 調整亮度／對比／飽和度／色相 | `brightness`、`contrast`、`saturation`、`hue` |
| `RandAugment` | 用兩個數字表達一整套增強策略 | `num_ops`、`magnitude`；`num_magnitude_bins`（進階） |
| `ComposeTransform` | 依埠的順序合併數條鏈 | `steps` |

只有在「兩條鏈是分開建立的，但同一條流程要跑完兩條」時才需要 `ComposeTransform` — 節點接節點本身就已經是組合了。

### 正規化預設組合

`NormalizeTransform` 內建了真正重要的統計值，你不用再去查：

| 預設組合 | mean | std | 適用情境 |
| --- | --- | --- | --- |
| `Half` | `(0.5,)` | `(0.5,)` | 預設值，也是在有預設組合之前 CodefyUI 一直採用的做法。把 `[0, 1]` 映射到 `[-1, 1]`。 |
| `ImageNet` | `(0.485, 0.456, 0.406)` | `(0.229, 0.224, 0.225)` | **任何 torchvision 預訓練模型。** 那些權重就是用這組數字訓練出來的；換成別的會讓輸入分布偏離權重的預期。 |
| `CIFAR-10` | `(0.4914, 0.4822, 0.4465)` | `(0.2470, 0.2435, 0.2616)` | 重現 CIFAR-10 的 baseline。 |
| `CIFAR-100` | `(0.5071, 0.4865, 0.4409)` | `(0.2673, 0.2564, 0.2762)` | 重現 CIFAR-100 的 baseline。 |
| `Custom` | 自訂 | 自訂 | 其他情況。只有選這個時才會出現 `mean` 與 `std` 欄位。 |

只給一個值時會廣播到所有通道，所以 `Half` 對單通道的 MNIST 和三通道的 CIFAR 都是正確的。

## 把鏈接到資料集

**Dataset** 與 **ImageFolderDataset** 各有兩個變換輸入：

- `train_transform` — 當 `split` 是 `train` 時使用。資料增強應該放在這裡。
- `eval_transform` — 其他分割都用它；當 `train_transform` 沒接時，訓練分割也會退回用它。

這個退回機制刻意只有單向。測試分割永遠不會拿到帶增強的鏈，因為一個被隨機扭曲過的測試集，每看一次量到的都是不同的東西。唯一的例外是 **ImageFolderDataset** 的 `(none)` 分割，因為那裡根本沒有分割可以退回；詳見下方說明。

至於沒有變換輸入的資料集 — **HuggingFaceDataset**、**KaggleDataset**，或你自己寫的 — 請改把鏈接到 **Transform** 節點的 `transform` 輸入。一旦那個輸入接上了，該節點的三個參數就會被忽略。

### CIFAR-10 的標準配方

CIFAR-10 的標準起手式，四個節點：

```
RandomCrop(size=32, padding=4)
  -> RandomHorizontalFlip(p=0.5)
  -> ToTensorTransform
  -> NormalizeTransform(preset="CIFAR-10")
  -> Dataset(name="CIFAR10", split="train").train_transform
```

評估用的分割則是同一條鏈，但去掉前兩個隨機步驟。

## 更多資料集

**Dataset** 現在提供 `MNIST`、`FashionMNIST`、`CIFAR10`、`CIFAR100`、`SVHN` 與 `STL10`。六個都會在第一次使用時下載到同一個 `data_dir`；在專案目錄模式下，那就是 `assets/data/`。

### 你自己的影像

**ImageFolderDataset** 讀的是 torchvision `ImageFolder` 預期的結構：

```
my-dataset/
  train/
    cat/  img001.png  img002.png ...
    dog/  img101.png ...
  val/
    cat/  ...
    dog/  ...
```

- `path` — 放置各個分割的資料夾。相對路徑會相對於同時放著 `models/` 與 `images/` 的資料目錄；絕對路徑則照用。
- `split` — 要載入哪個子資料夾。如果類別資料夾直接放在 `path` 底下、沒有分割這一層，就選 `(none)`。選 `(none)` 時沒有分割可以用來區分那兩個變換輸入，所以哪一個有接上就用哪一個；兩個都接上時，以 `train_transform` 為準。

標籤由資料夾名稱依字母順序決定，所以在任何機器上 `cat` 都是 0、`dog` 都是 1。這個節點另外輸出一個依標籤順序排列的 `classes` 列表。

### 其他形式的資料

如果你的資料不是「一個類別一個資料夾」 — 例如一份記錄路徑的 CSV、自訂的封裝格式、或即時生成的分布 — 請寫一個自訂節點（見 [自訂節點](../advanced/custom-nodes.md)）或使用 [PythonScript 節點](../advanced/python-script-node.md)。只要你回傳的物件有一個公開、可寫入、且會在 `__getitem__` 中被套用的 `transform` 屬性，這一頁所有的變換鏈就都能用在它身上。

## 可重現性

資料增強是隨機的，所以「可重現的執行」也必須重現同一組增強。在執行選項裡給定 **種子（seed）** 就會做到：同一個種子會產生同樣的裁切、翻轉與色彩偏移，每次都一樣。

有四個細節值得知道：

- **隨機串流是隔離的。** 一條鏈的隨機性只取決於執行種子，以及把它掛上去的那個節點的身分，除此之外別無其他。改動模型或 dropout 比例，都不會改變你拿到的裁切。不過一旦 `num_workers` 大於 0，batch size 和 worker 數量仍然會決定「哪個樣本從串流的哪個位置取值」— 因為每個 worker 各有自己的串流。所以這兩個設定會改變某張影像實際拿到的變換，但只要設定不變，重跑一次就會完全重現同樣的結果。
- **它仍然會變化。** 可重現不等於凍結：樣本之間彼此不同，第 2 個 epoch 也和第 1 個不同。這正是資料增強的意義所在，而且在有設 `num_workers` 與沒設的情況下都成立。
- **沒給種子的執行完全不受影響。** 沒有種子時，流程維持使用 PyTorch 自己的亂數來源，完全沒有額外成本。額外的簿記只有在「執行有要求種子」**而且**「鏈裡確實有隨機步驟」時才會裝上。
- **匯出的腳本會帶著種子。** 「匯出為 Python」會把畫布上的種子寫進產生的檔案，成為 `GRAPH_SEED`，也就是該腳本 `--seed` 的預設值。所以匯出的圖會重現畫布上的結果 — 同樣的裁切、同樣的翻轉 — 不需要任何人記得加參數。要覆寫就用 `--seed 123`，或用 `--no-seed` 讓每次執行都取用新的亂數。決定性核心的開關也一樣會跟著走，對應 `--deterministic` / `--no-deterministic`。

## TensorBoard

**TrainingLoop** 的 **進階** 區有一個 **tensorboard** 參數。打開它，執行時除了寫進 CodefyUI 自己的圖表之外，也會把指標寫成 TensorBoard 事件檔 — 同樣的序列，來自同一個呼叫點，所以兩者不可能對不上。

檔案會放在該次執行專屬的資料夾裡，和 `models/`、`images/` 並列：

```
<資料根目錄>/runs/<執行 id>/tb/<節點 id>/
```

圖裡的每個訓練節點都有自己的葉子目錄，所以「預訓練迴圈」和「微調迴圈」在 TensorBoard 裡會畫成兩條獨立的曲線，而不是一條來回鋸齒的線。

該路徑會登記成這次執行的產出檔案，你可以從 **執行紀錄** 面板複製出來直接開啟：

```bash
tensorboard --logdir <從執行紀錄面板複製的路徑>
```

CodefyUI **不需要** 依賴 TensorBoard 就能寫出這些檔案 — 它自己編碼事件格式，所以開啟這個功能不會讓 CodefyUI 的安裝變重。你只有在要 *檢視* 它們的時候才需要安裝 TensorBoard：

```bash
pip install tensorboard
```

有兩件事是它刻意不做的。當這次執行沒有地方可以登記產出檔案時（匯出的腳本、CLI contract runner），它什麼都不寫，因為一個沒有任何東西指向它的資料夾，就是任何清理機制都找不到的垃圾。另外，非有限的數值會被丟棄而不是寫入，因為一個 NaN 會毀掉同一張圖表裡其他所有序列的 y 軸自動縮放 — 這個數值在執行本身的指標儲存區裡仍然保留著。

**這些記錄的壽命和它所屬的那次執行一樣長。** 當一次執行掉出「保留最近 N 筆」的範圍時（`CODEFYUI_RUN_RETENTION_KEEP_LAST`，預設保留 200 筆已完成的執行 — 見 [執行佇列](./run-queue.md)），它的資料列會被刪除，它的 `tb/` 資料夾也會一起被刪掉。這就是 `runs/` 不會隨著安裝的使用時間無限膨脹的原因 — 否則每次執行的每個訓練節點都會留下一個資料夾。所以如果某條曲線值得保存到兩百次執行之後，請把那個資料夾另外複製一份出來，或是把上限調高。

## 把指標匯出成 CSV

每次執行的指標都可以下載成 CSV，在 **執行紀錄** 面板有兩個入口：每一列上的 **CSV** 按鈕，以及展開某次執行後、圖表旁邊的 **下載 CSV**。兩者產生的檔案相同，一列一個資料點，包含序列名稱、step 與數值。
