# Embeddings and RAG

An embedding is a piece of text turned into a list of numbers -- a vector,
usually a few hundred numbers long. The model that produces it is trained so
that texts meaning similar things land near each other, even when they share no
words at all. "How do I run a graph?" and "press the Run button in the toolbar"
have almost nothing in common as strings, but their vectors sit close together.

Closeness is measured with cosine similarity: the cosine of the angle between
two vectors, which is 1 when they point the same way, 0 when they are
unrelated, and negative when they point apart. It ignores length and looks only
at direction, which is what you want -- a long passage and a short question
should be comparable.

Long documents are cut into chunks before they are embedded. One vector has to
stand for everything in the text it covers, so a whole chapter averages into a
vague direction that is close to every question and useful for none. A chunk of
a few hundred characters, with a little overlap so a sentence is not split down
the middle, keeps each vector about one idea.

RAG -- retrieval-augmented generation -- has four stages. Load the documents.
Chunk them. Embed each chunk and put the vectors in an index. Then, at question
time, embed the question, retrieve the nearest chunks, and hand them to a
language model as context along with the question.

The instruction to answer only from the retrieved context is what makes the
whole thing worth building. A language model asked a question with no context
will answer from memory and sound equally confident whether it is right or
wrong. Grounded in retrieved passages, it can quote its source, and it can say
that the documents do not contain the answer -- which is a far more useful
reply than a fluent guess.

## 中文

嵌入就是把一段文字變成一串數字 -- 一條向量，長度通常是幾百個數。產生它的模型在
訓練時被要求：意思相近的文字，向量要落在相近的位置，就算兩段文字一個字都沒重
疊也一樣。「我要怎麼執行一張圖？」和「按工具列上的執行按鈕」這兩句話當成字串看
幾乎毫無交集，但它們的向量靠得很近。

「靠得近」是用 cosine similarity 來量的：兩條向量夾角的餘弦值，方向相同時是 1、
毫無關聯時是 0、方向相反時是負的。它忽略長度、只看方向，而這正是我們要的 -- 一
段長文和一個短問句本來就應該可以互相比較。

長文件在被嵌入之前要先切塊。一條向量必須代表它所涵蓋的全部內容，所以整整一章壓
成一條向量，方向會變得很模糊：對每個問題都有點像，對每個問題也都沒有用。切成幾
百個字元、並且讓相鄰的塊稍微重疊（免得一句話被從中間切斷），能讓每條向量大致只
講一件事。

RAG（檢索增強生成）有四個階段。載入文件。把文件切塊。把每一塊嵌入成向量並放進
索引。然後在有人提問時，把問題也嵌入、取出最接近的幾塊，連同問題一起當作脈絡交
給語言模型。

「只根據取出的脈絡作答」這條指示，才是整件事值得做的原因。語言模型在沒有脈絡的
情況下被問問題，會憑記憶回答，而且答對答錯聽起來一樣有把握。有了取出的段落當依
據，它可以引用出處，也可以說「這些文件裡沒有答案」-- 而這個回答，遠比一個流暢
的猜測有用得多。
