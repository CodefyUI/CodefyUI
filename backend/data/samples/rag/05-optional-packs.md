# Optional Packs

A base install of CodefyUI is deliberately small. Everything that would make
the download large -- embedding models, word vector tables, a local text
generator, a GPU build of PyTorch -- lives in an optional pack instead, and you
install only the ones your lesson needs. Packs are managed in the Package
Center, reached from Settings in the toolbar.

There are four packs today. Sentence embeddings brings the
sentence-transformers library plus four small embedding models: an English one,
a multilingual one, a Chinese one, and one tuned for retrieval. Word vectors
ships the 400,000-word GloVe table used to show that vector arithmetic on words
really does behave the way the textbook claims; it needs no Python packages at
all, just the data file. The RAG stack adds a small local instruction-following
model so a retrieval graph can generate an answer without calling out to an
API. GPU PyTorch swaps the CPU build of PyTorch for the CUDA or ROCm build that
matches the machine, and is the one pack whose install restarts the server.

One rule matters more than the rest: a graph run never downloads pack
contents. Four hundred megabytes arriving in the middle of a run, on a
classroom connection, with no progress bar and no way to cancel, is not
something a Run button is allowed to do. A node whose pack is missing stops
with an error naming the pack and pointing at the Package Center, and nothing
is fetched behind your back.

Downloaded files go to a cache directory outside your project, shared by every
graph on the machine, so two projects that use the same model download it once.
Deleting that cache costs a re-download and nothing else. Every model and data
file in the catalogue carries a permissive licence -- Apache-2.0, MIT or
PDDL-1.0 -- and the Package Center shows each item's licence and approximate
size before you agree to fetch it.

## 中文

CodefyUI 的基本安裝刻意做得很小。所有會讓下載變大的東西 -- 嵌入模型、詞向量表、
本地端的文字生成模型、GPU 版的 PyTorch -- 都放在選用套件包裡，你只需要安裝這堂
課用得到的那幾個。套件包在套件中心管理，從工具列的設定進去。

目前有四個套件包。Sentence embeddings 會帶進 sentence-transformers 函式庫，以及
四個小型嵌入模型：一個英文的、一個多語的、一個中文的，還有一個是為檢索調校的。
Word vectors 提供 40 萬詞的 GloVe 詞向量表，用來實際驗證課本上說的「詞向量的加
減真的有意義」；它不需要任何 Python 套件，只需要那個資料檔。RAG stack 會加上一
個小型的本地指令模型，讓檢索流程不必呼叫外部 API 也能生成答案。GPU PyTorch 則把
CPU 版的 PyTorch 換成符合這台機器的 CUDA 或 ROCm 版本，也是四個裡面唯一一個安裝
後會重新啟動伺服器的。

有一條規則比其他都重要：執行圖的時候絕對不會下載套件包的內容。在教室的網路上，
跑到一半突然開始下載四百 MB、沒有進度條也沒辦法取消 -- 這不是一個「執行」按鈕
可以做的事。缺少套件包的節點會直接停下來，錯誤訊息裡會寫出缺的是哪個套件包、並
指向套件中心，不會在你不知情的狀況下偷偷抓東西。

下載下來的檔案會放在專案之外的快取資料夾，整台機器上的所有圖共用，所以兩個專案
用到同一個模型時只會下載一次。刪掉這個快取的代價就只是重新下載一次而已。目錄裡
每一個模型與資料檔的授權都是寬鬆授權 -- Apache-2.0、MIT 或 PDDL-1.0 -- 而且套件
中心會在你同意下載之前，先顯示每個項目的授權與大致大小。
