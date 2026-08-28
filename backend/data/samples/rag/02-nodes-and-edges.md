# Nodes, Ports and Edges

Every node has ports. Input ports are on the left, output ports on the right,
and each one declares a data type: Tensor, String, Scalar, List, Dataset,
DataLoader, Model, Optimizer, and a few more. The type is not decoration. When
you drag a wire, the canvas checks whether the two ends agree, and it will not
let you connect a Model output to a Tensor input. Most beginner mistakes are
caught right there, before anything runs.

Edges come in two flavours. A data edge carries a value from an output port to
an input port, and it is what makes a node's result available to the next node.
A trigger edge carries no value at all -- it only says "run this next". Trigger
edges start at a Start node, and they are how the graph decides what actually
executes. A node that produces data but has no trigger path to it stays idle,
which is intentional: it lets you park half-built branches on the canvas
without them slowing every run down.

To run a graph, press Run in the toolbar. Nodes light up as they execute, and a
node that fails turns red and shows its error message in place, so you can see
which step broke rather than reading a stack trace bottom-up. The Print node is
the simplest way to look at a value: wire anything into it and its output shows
up in the Execution Log panel.

For anything bigger than a number, use the Inspector. Click a node after a run
and the Inspector shows what came out of each port -- a tensor's shape and
dtype, a list's length and first few items, a chart or an image if the node
emits one. Between Print for quick checks and the Inspector for shapes, most
debugging is looking at the value on a wire and asking whether it is the value
you expected.

## 中文

每個節點都有埠。輸入埠在左邊、輸出埠在右邊，而且每個埠都會宣告自己的資料型別：
Tensor、String、Scalar、List、Dataset、DataLoader、Model、Optimizer，還有其他
幾種。型別不是裝飾。當你拉一條線的時候，畫布會檢查兩端是否相符，你沒辦法把一個
Model 輸出接到 Tensor 輸入上。大部分初學者的錯誤在這一步就被擋下來了，根本還沒
開始跑。

邊有兩種。資料邊把值從輸出埠帶到輸入埠，讓一個節點的結果能被下一個節點用到。
觸發邊則完全不帶值 -- 它只說「接下來跑這個」。觸發邊從 Start 節點出發，圖要執行
哪些節點就是由它決定的。一個會產生資料、但沒有任何觸發路徑連到它的節點會保持不
動，這是刻意的設計：你可以把做到一半的分支先擱在畫布上，而不會讓每次執行都被它
拖慢。

要執行一張圖，按工具列上的執行。節點會在輪到它時亮起來，失敗的節點會變紅並就地
顯示錯誤訊息，所以你看得出是哪一步壞掉，不必從錯誤堆疊的最底下往回讀。想看某個
值，最簡單的辦法是 Print 節點：把任何東西接進去，結果就會出現在執行紀錄面板裡。

比數字更複雜的東西，就用檢視器。跑完之後點一下節點，檢視器會顯示每個埠輸出了什
麼 -- 張量的形狀與 dtype、列表的長度與前幾個元素，如果節點會產生圖表或影像也會
一併顯示。用 Print 做快速確認、用檢視器看形狀，大部分的除錯就是去看線上的值，然
後問自己：這是我預期的值嗎？
