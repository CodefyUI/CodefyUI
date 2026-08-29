# Training Basics

A supervised training run needs five pieces, and CodefyUI gives each of them a
node. A Dataset holds the examples and their correct answers. A DataLoader cuts
that dataset into batches and shuffles them, because updating the model once
per example is slow and updating it once per whole dataset is coarse. A model
turns an input into a prediction. A loss function scores how wrong that
prediction was. An optimizer uses the loss to nudge the model's weights in a
better direction.

Two numbers control the run. Epochs is how many times the loop walks through
the whole dataset; each pass gives the optimizer another chance to improve on
the same examples. The learning rate is how big each nudge is. Too small and
training crawls; too large and the loss jumps around or turns into NaN, because
each step overshoots the minimum it was aiming at. Changing only the learning
rate and watching the loss curve is the single most useful experiment a
beginner can run.

The loss curve is the run's vital sign. A curve falling steeply and then
flattening is healthy: the model learned the easy structure quickly and is now
working on the rest. A flat line from the start usually means the learning rate
is far too small, the data is not connected to the labels, or the model has no
capacity to fit anything. A curve that falls and then rises again is a sign the
learning rate is too high.

Overfitting is what happens when the model stops learning the pattern and
starts memorising the examples. Training loss keeps falling while validation
loss levels off and then climbs -- the model is getting better at the data it
has seen and worse at data it has not. This is why the dataset gets split, and
why the number that matters is the one measured on data the model never
trained on.

## 中文

一次監督式訓練需要五個元件，CodefyUI 為每一個都準備了節點。Dataset 存放範例與
它們的正確答案。DataLoader 把資料集切成一批一批並且打亂順序，因為每看一個範例就
更新一次模型太慢，而看完整個資料集才更新一次又太粗糙。模型把輸入變成預測。損失
函數為「這個預測錯得多離譜」打分數。優化器則依據損失，把模型的權重往比較好的方
向推一點點。

有兩個數字在控制這整個過程。Epoch 是迴圈要把整個資料集走過幾遍；每走一遍，優化
器就多一次在同一批範例上改進的機會。學習率則決定每次推動的幅度有多大。太小，訓
練會慢到像在爬；太大，損失會上下亂跳甚至變成 NaN，因為每一步都衝過了它原本要去
的最低點。只改學習率、然後看損失曲線怎麼變，是初學者能做的最有價值的實驗。

損失曲線是一次訓練的生命徵象。先急速下降、然後趨於平緩，是健康的：模型很快學會
了容易的部分，正在處理剩下的。從頭到尾都是一條平線，通常代表學習率太小、資料和
標籤根本對不起來，或是模型根本沒有能力擬合任何東西。先降下去又爬上來，則是學習
率太高的徵兆。

過擬合是指模型不再學習規律，而開始死記範例。訓練損失持續下降，驗證損失卻先持平
再往上爬 -- 模型在看過的資料上愈來愈好，在沒看過的資料上愈來愈差。這正是為什麼
資料集要切分，也是為什麼真正重要的數字，是在模型從未訓練過的資料上量出來的那一
個。
