---
sidebar_position: 8
title: 訓練記憶體
description: 讓單張顯示卡跑更大的訓練——混合精度、梯度累積、指定某一張 GPU、記憶體不足時會發生什麼，以及伺服器自己的記憶體上限。
---

# 訓練記憶體

大部分訓練跑不完，原因不是數學算錯，而是顯示卡的記憶體不夠了。本頁說明 CodefyUI 提供的四個把這個天花板往上推的工具，以及一件它刻意不幫你做的事。

以下所有功能**預設都是關閉的**。在這些功能出現以前存下來的圖，行為和以前完全一樣。

## 混合精度

`TrainingLoop` 的**進階**區塊裡有一個 **precision** 參數：

| 值 | 做什麼 | 什麼時候用 |
| --- | --- | --- |
| `fp32` | 什麼都不做，全程 32 位元。 | 預設值。永遠正確，也永遠最耗記憶體。 |
| `bf16` | 前向傳遞與 loss 在 `autocast(torch.bfloat16)` 底下執行，不需要 gradient scaler。 | **首選。** Ampere 以後的顯卡——RTX 30xx、40xx、50xx、A100、H100。 |
| `fp16` | 同上，但用 `float16`，另外搭配 `GradScaler`。 | 不支援 bfloat16 的顯卡——Volta 與 Turing，也就是 GTX 16xx 和 RTX 20xx。 |

在深的模型上，把顯示卡塞滿的是 activation 而不是權重，而 autocast 會把 activation 砍半。參數在兩種情況下都維持 float32，所以記憶體主要花在權重上的模型，效果會比記憶體主要花在特徵圖上的模型小。

**為什麼新卡要用 bf16 而不是 fp16。** bfloat16 保留了 float32 的指數範圍，把位元花在尾數上。因此小的梯度不會下溢，loss scaler 沒有事情可做，也不會有被跳過的更新。float16 只有五個指數位元，所以需要一個 scaler 在 `backward()` 之前把 loss 乘大、在更新之前再把倍率除回去——而且只要有梯度溢位，就會退一步並整個跳過那次更新。

**每個裝置實際做得到什麼。** 這個選擇會在執行開始前先跟裝置對照過；做不到的裝置會退回 `fp32` 並發出警告，而不是讓執行失敗：

- 不支援 bfloat16 的 CUDA 裝置要求 `bf16` 時會拿到 `fp32`。
- MPS 除了 `fp32` 以外都會拿到 `fp32`。Apple 的 autocast 覆蓋範圍會隨 torch 版本而異，CodefyUI 無法在你的機器上驗證。
- CPU 三種都支援。兩種 16 位元模式在 CPU 上都不會比較*快*——重點是讓這堂課能在你面前這台機器上跑起來。

節點的設定訊框與 `metrics` 輸出都會回報 `precision`（實際跑的），以及在兩者不同時回報 `precision_requested`（你要求的）。驗證階段跑在和訓練一樣的 autocast 底下，所以兩條 loss 曲線可以互相比較。

### 續跑一個 fp16 的訓練

loss scale 是訓練狀態的一部分。`CheckpointSaver` 與 `CheckpointLoader` 用 `grad_scaler_state` 這個 port 攜帶它，`TrainingLoop` 兩側各有一個：

```
TrainingLoop.grad_scaler_state  →  CheckpointSaver.grad_scaler_state
CheckpointLoader.grad_scaler_state  →  TrainingLoop.grad_scaler_state
```

`fp32` 與 `bf16` 兩邊都不用接——沒有 scale 需要保存。弄丟這個狀態不會致命：新的 scaler 在幾百步之內就會重新找到它的水位，只是這中間那些步是在錯誤的倍率下走的。

## 梯度累積

**accumulate_steps**（同樣在**進階**區塊）會先跑 N 批、把每一批的 loss 除以 N，然後才做一次優化器更新。

這樣送到優化器的梯度，就是一個大 N 倍的批次的梯度——當 loss 是平均值、而且每個小批次都是滿的時候，這是精確相等而不是近似。所以：

> batch_size 8 搭配 accumulate_steps 4 **就是** batch_size 32，而 activation 記憶體只要四分之一。

這是有測試撐著的主張，不只是說明：測試會用手算出整批的梯度，並要求累積版本落在完全相同的權重上。

有三個互動值得知道：

- **梯度裁剪**發生在更新的時候，對象是整個累積起來的梯度。改成逐小批次裁剪的話，N 個裁剪過的梯度加起來會是門檻值的 N 倍。
- **`max_steps` 算的是優化器更新次數**而不是批次數，所以不管 `accumulate_steps` 設多少，它代表的學習量都一樣。`metrics` 輸出兩個都會回報：`total_steps`（優化器更新）與 `total_batches`（前向傳遞）。
- **回報的 loss 沒有被除過。** `/N` 只是組裝梯度的細節，不會進到圖表裡，所以同一次執行在 accumulate_steps 1 和 4 之下畫出來的曲線是一樣的。

如果某個 epoch 的批次數不是 `accumulate_steps` 的倍數，最後會剩下一個不完整的視窗，而那個視窗仍然會被更新——那些批次的前向與反向傳遞已經算過了。它和其他視窗一樣除以 N，所以短視窗會走出一個按比例縮小的步伐，這才是對較少樣本誠實的處理方式。

累積視窗不會跨越 epoch 邊界，而按下 **Stop** 時會丟掉還沒更新的視窗，而不是在離開前再多走一步。

## 指定某一張 GPU

在有一張以上 CUDA 裝置的機器上，每個裝置下拉選單——**設定**裡的全域選擇器，以及每個節點自己的 **device** 參數——都會逐張列出：

```
CPU
NVIDIA CUDA          （torch 目前指向的那一張）
NVIDIA CUDA #0       cuda:0
NVIDIA CUDA #1       cuda:1
```

只有一張 GPU 的機器只會顯示 `NVIDIA CUDA`，因為在那裡 `cuda` 和 `cuda:0` 是同一塊硬體，兩個都給等於在請使用者做一個沒有內容的選擇。

每一張卡都有**自己的執行佇列**，所以 `cuda:0` 上一個跑六小時的工作，不會拖到送到 `cuda:1` 的執行。請參閱[執行佇列](/usage/run-queue)。

**指定了不存在的編號**——兩張卡的機器上寫 `cuda:3`，或是在工作站存下來、在筆電上打開的圖——會退回*目前的 CUDA 裝置*，並發出一則寫明卡數的警告。不會退回 CPU：你要求的是在 GPU 上訓練，用一次安靜的四十分鐘 CPU 執行來回答那個要求，是更糟的意外。

根本無法尋址的裝置字串——`cuda:`、`cuda:abc`（例如沒展開的 `${CUDA_INDEX}`）、`cuda:0:1`、`cuda: 0`——會落到同一個地方，理由也一樣：`cuda` 這個前綴仍然是明確要求 GPU，錯的只有編號。在這之前，這些字串會原封不動送到 PyTorch，換來 `RuntimeError: Invalid device string`——而那則錯誤既沒有說是哪張圖，也沒有說是哪個參數。

:::note 分散式訓練不在範圍內
一次執行是單一行程、單一裝置。沒有 `DistributedDataParallel`、沒有多卡資料平行、也沒有任何多機的東西——一次執行用一張卡。多次執行可以透過各裝置的佇列同時佔用多張卡，一張卡一個。

這是刻意畫下的界線，不是還沒補上的缺口。DDP 需要行程啟動、rendezvous、逐 rank 的記錄與逐 rank 的檢查點，而這每一項都會動到 run service、事件串流與 artifact 儲存。做一半會比完全不做更糟。
:::

## 如果還是不夠用

CUDA 記憶體不足會以 **NodeOOMError** 回報：哪一個節點、哪一個裝置、當下配置器手上握著什麼，以及該改什麼。大致長這樣：

```
Node TrainingLoop (n7) ran out of memory on cuda:0.

What to try, cheapest first:
  - reduce batch_size on the DataLoader node
  - set TrainingLoop's precision to bf16 (roughly halves activation memory on Ampere and newer)
  - raise TrainingLoop's accumulate_steps and lower batch_size by the same factor, which keeps the effective batch identical
  - make the model smaller, or shorten the sequence / shrink the image

CUDA memory on cuda:0: 14.82 GiB held by live tensors, 15.44 GiB reserved by
the caching allocator, 15.61 GiB peak this process, 15.99 GiB on the card.

Original error: CUDA out of memory. Tried to allocate 2.00 GiB ...
```

除了這則訊息之外，還會做兩件事，好讓*下一次*執行從乾淨的卡開始：丟掉那個節點快取起來的東西，並把快取配置器的空閒區塊還回去。

**不會重試，也不會幫你把 batch size 調小。** 同樣的配置再跑一次會得到同樣的答案。在你背後把批次砍半，會在你不知情的狀況下改變這次執行產生的數字，那麼同一張圖就會因為當下剩多少 VRAM 而代表兩種不同的意思。你會拿到訊息，然後由你來改。

## 伺服器自己的記憶體

有三個記憶體內的儲存區會在多次執行之間保留張量，三個現在都同時受位元組與筆數上限約束：

| 儲存區 | 放什麼 | 上限 | 設定 |
| --- | --- | --- | --- |
| 執行快取 | 節點輸出，用來做「改一個節點，只有那個子樹重跑」 | **每個開啟的編輯器連線** 1 GB | `CODEFYUI_EXECUTION_CACHE_MAX_MB` |
| 執行輸出儲存區 | 給教學檢視器用的 port 擷取值 | 2 GB | `CODEFYUI_RUN_OUTPUT_STORE_MAX_MB` |
| 節點狀態儲存區 | 每個節點持久保存的 `nn.Module` 權重 | 1 GB | `CODEFYUI_NODE_STATE_STORE_MAX_MB` |

把其中任何一個設成 `0`，就會關閉位元組上限，改由原本的筆數上限負責。

光看筆數從來就不算是記憶體上限：256 筆快取起來的節點輸出，可能是 40 MB 的純量，也可能是 200 GB 的特徵圖，而筆數分辨不出這兩者。淘汰策略是最近最少使用，大小則是以張量的 storage 為準來量測——會遞迴走過 list、tuple 與 dict，而且一個張量和它的 view 只會算一次而不是兩次。

`GET /api/health` 會回報每個儲存區目前用量與上限的對照：

```json
{
  "status": "ok",
  "caches": {
    "execution_cache": {"instances": 2, "entries": 41, "bytes": 918212608, "max_bytes_each": 1073741824},
    "run_output_store": {"runs": 6, "max_runs": 20, "bytes": 244318208, "max_bytes": 2147483648},
    "node_state_store": {"modules": 12, "max_modules": 200, "bytes": 51380224, "max_bytes": 1073741824}
  }
}
```

關於這些數字有兩點要先說清楚，因為一份會偷偷四捨五入的記憶體報告比沒有更糟。模組的大小是在模組被建立時量的，所以模組在訓練過程中長出來的梯度不會重新計入。另外 CPU 與 CUDA 的位元組是加在一起的——上限比較的是同一個數字，而這裡報的就是那個數字。
