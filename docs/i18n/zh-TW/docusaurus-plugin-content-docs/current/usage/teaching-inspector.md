---
sidebar_position: 4
title: 教學檢視器
description: 記錄每個節點的輸出、檢視 input→output 張量差異、比較子圖段落、捕獲梯度，並檢視步驟追蹤。
---

# 教學檢視器

CodefyUI 可以作為一份 **互動式教材** — 讓學生看到流過每一個節點的真實張量。教學檢視器會在執行期間捕獲節點的輸出，並把它們渲染在右側面板。

## 操作說明

1. 把一個 **`TensorInput`** 節點拖到畫布上（Data 類別）。把 `value_mode` 設成 `explicit`，並在內嵌格子裡填入你想讓管線看到的數值。
2. 用任意一串張量運算節點把它串起來（例如 `Reshape → Softmax → Print`）。
3. **新增一個 `Start` 節點**，把它的 trigger 輸出連到你想開始執行的第一個節點 — 通常就是 `TensorInput`。少了這個，圖就是一份草稿，**執行** 會被拒絕（見 [你的第一個圖](./first-graph)）。
4. 開啟工具列的 **⚙ 設定** popover，把 **記錄輸出** 切到 ON，然後點擊 **執行**。每個完成的節點的完整輸出會被保存在伺服器記憶體中，以該次 run 當作索引。
5. 點任何一個節點 — **檢視器** 面板會抓取該節點的輸入與輸出，由上到下堆疊顯示 **shape、dtype、min/max/mean** 與實際數值。數值變動的格子會以熱力色標示。
6. **Shift 選取兩個節點**，使用 **段落比較**（在 ⚙ 設定 → 檢視段內），聚焦在頭部輸入與尾部輸出；畫布會以淺橘色泡泡把它們包起來並加上 **HEAD** / **TAIL** 標籤。
7. 在跑重度訓練前，如果你不想記錄每個 epoch，可以先把 **記錄輸出** 切回 OFF — 先前已捕獲的 run 在伺服器重啟前仍然可以查閱。

:::note
被捕獲的資料存放在 session 當下的記憶體（LRU，最近 20 次 run）。段落標記會跟著 graph JSON 一起儲存。
:::

## 設定 popover 的開關

工具列的 **⚙ 設定** popover 把每個分頁的教學／訓練開關集中在同一處 — 概念類似 VS Code 的 Settings UI：

| 開關 | 用途 |
|--------|---|
| **記錄輸出 (Record outputs)** | 把每個完成節點的完整輸出存到 Inspector 中。預設關閉；跑重度訓練前記得關。 |
| **詳細模式 (Verbose mode)** | 後端把中間算法步驟（attention scores、softmax 溫度等）連同輸出一起記錄 — 餵給 Inspector 的 **Steps** 分頁。 |
| **段落比較 (Compare Segment)** | 把 shift 選取的兩個節點包成 HEAD/TAIL 泡泡，讓 Inspector 只顯示這個子圖的頭尾邊界。 |
| **跨 run 保留權重** | 跨次 Run 保留 `Conv2d`／`Linear`／`Attention` 權重（讓模型真的學得到東西）；關閉時每次 Run 都會重新初始化。 |
| **立即重置全部權重** | 清掉這個分頁所有快取的權重；下一次 Run 會從頭初始化。 |
| **捕獲梯度 (Capture gradients)** | 跑 forward + `.backward()` 並把每層的梯度存起來，給 Inspector 的 **Backward** 分頁。 |
| **自動合成 loss** | 當圖中沒有 `Loss`／`BackwardOnce` 節點時，自動合成一個讓 `.backward()` 跑得起來。 |
| **格線吸附 (Grid snap)** | 拖曳節點時自動吸附到畫布格線。 |
| **顯示節點 tooltip** | 滑鼠停在畫布上的節點時顯示描述卡片。 |
| **節點分類模式** | `入門` 只在側欄顯示基本類別；`全部` 顯示所有類別。 |

## 步驟追蹤（詳細模式）

開啟 **詳細模式** 後，受插樁的節點會發出 `__steps__` 追蹤，Inspector 會一次一列地渲染。教學用的外掛節點大量仰賴這個 — 例如 `Edu-ColumnStats` 會把母體標準差公式顯示為 `sum → divide → deviations² → variance → sqrt`。見 **[外掛](/advanced/plugins)**。

## 梯度捕獲（Backward 分頁）

開啟 **捕獲梯度** 後，引擎會跑一次 forward pass、呼叫 `.backward()`，並把每層的梯度存起來。在 Inspector 中開啟某個節點的 **Backward** 分頁，即可看到每層的梯度大小 — 對診斷梯度消失／爆炸很有用。
