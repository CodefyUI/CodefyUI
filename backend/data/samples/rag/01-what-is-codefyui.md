# What CodefyUI Is

CodefyUI is a visual, node-based way to build and run machine-learning
pipelines in a browser. Instead of writing a training script from scratch, you
drag nodes onto a canvas, wire them together, and press Run. The graph you draw
is the program: what you see is exactly what executes, in the order the wires
describe.

A graph is made of three things. Nodes are the units of work -- one loads a
dataset, one holds a model, one runs a training loop, one prints a number.
Typed edges carry values between nodes, and the types have to match, so a
mismatch shows up as a connection the canvas refuses to make rather than as an
error ten minutes into a run. A Start node marks where execution begins: it
sends a trigger into the graph, and a node runs when the trigger reaches it,
or when something it feeds is going to run.

That last point is the difference between a diagram and a program. A dataset
node needs no trigger of its own: the training node it feeds has one, so the
dataset is loaded anyway. What stays idle is a branch connected to nothing
that runs, which is how a half-built experiment can sit on the canvas costing
nothing. Two Start nodes give you two independent branches within a single
run, which is a convenient way to keep an experiment beside the thing you are
comparing it against.

CodefyUI is built for three kinds of people. Students see the shape of a
pipeline before they have to write one, and can change a single thing -- a
learning rate, an activation, a layer width -- and watch what moves. Teachers
get a lesson that runs on a classroom laptop, with no environment to set up in
the first twenty minutes. Teams evaluating an approach get a sketch they can
share and run, rather than a notebook that only works on the machine it was
written on.

## 中文

CodefyUI 是一套在瀏覽器裡用「節點圖」來組裝並執行機器學習流程的工具。你不必從
零寫一支訓練腳本，而是把節點拖到畫布上、把它們接起來，然後按下執行。你畫出來的
圖就是程式：看到什麼就跑什麼，執行順序由連線決定。

一張圖由三種東西組成。節點是工作的單位 -- 有的載入資料集，有的持有模型，有的跑
訓練迴圈，有的把數字印出來。邊帶著值在節點之間流動，而且兩端的型別必須相符，所
以接錯的時候畫布會直接拒絕連線，而不是等到跑了十分鐘才報錯。Start 節點標記執行
從哪裡開始：它會往圖裡送出一個觸發訊號；一個節點會被執行，可能是因為觸發訊號到
得了它，也可能是因為它餵給的節點即將執行。

最後這一點正是「示意圖」與「程式」的分野。資料集節點通常沒有自己的觸發線：它餵
給的訓練節點有，所以資料集照樣會被載入。真正不動的，是一條沒有連到任何會執行的
東西的分支 -- 做到一半的實驗就可以這樣擱在畫布上，不花任何代價。放兩個 Start
節點，就等於在同一次執行裡有兩條互相獨立的分支 -- 想把一個實驗和它的對照組並排
放著看時，這很方便。

CodefyUI 是為三種人設計的。學生可以在還沒有能力自己寫出流程之前，先看見流程的
樣子，而且只改一個地方 -- 學習率、激活函數、某一層的寬度 -- 就能觀察到什麼跟著
變。老師拿到的是一堂能在教室筆電上直接跑的課，不必把前二十分鐘花在安裝環境上。
正在評估某個做法的團隊，得到的是一份可以分享、可以執行的草圖，而不是一本只有在
作者那台機器上才跑得動的筆記本。
