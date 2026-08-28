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
sends a trigger into the graph, and only the nodes that trigger reaches are
run.

That last point is the difference between a diagram and a program. A node
sitting on the canvas with no path back to Start is documentation, not work.
Two Start nodes give you two independent runs on one canvas, which is a
convenient way to keep an experiment beside the thing you are comparing it
against.

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
從哪裡開始：它會往圖裡送出一個觸發訊號，只有這個訊號到得了的節點才會被執行。

最後這一點正是「示意圖」與「程式」的分野。畫布上一個沒有任何路徑回到 Start 的
節點，是說明，不是工作。放兩個 Start 節點，就等於在同一張畫布上有兩次互相獨立的
執行 -- 想把一個實驗和它的對照組並排放著看時，這很方便。

CodefyUI 是為三種人設計的。學生可以在還沒有能力自己寫出流程之前，先看見流程的
樣子，而且只改一個地方 -- 學習率、激活函數、某一層的寬度 -- 就能觀察到什麼跟著
變。老師拿到的是一堂能在教室筆電上直接跑的課，不必把前二十分鐘花在安裝環境上。
正在評估某個做法的團隊，得到的是一份可以分享、可以執行的草圖，而不是一本只有在
作者那台機器上才跑得動的筆記本。
