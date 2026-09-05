---
sidebar_position: 7.9
title: 範例專案
description: 官方提供、可以直接 clone 的範例服務 -- 每一個都是獨立完整的 CodefyUI 專案儲存庫，可以執行、發佈與 fork。
---

# 範例專案

每個官方範例都是一個獨立完整的[專案目錄](./project-directories)，以自己的儲存庫放在 CodefyUI 組織底下。clone 一個、在它上面啟動伺服器，你的畫布上就有一個能用的服務 -- 接著把它發佈成真正的 HTTP API，再把儲存庫推到任何你喜歡的地方。它們同時也是誠實的範本：fork 一個，換上你自己的 graph 就行。

每個儲存庫都遵循同樣的三個指令：

```bash
git clone https://github.com/CodefyUI/<example-name>
cd <example-name>
cdui start --project .
```

在伺服器印出的網址開啟編輯器，從工具列載入 graph，然後按下**執行**。每個 README 都會帶你走完整個發佈流程（commit、`cdui project publish`、建立 API key、用 curl 呼叫），並列出該範例在 CodefyUI 本身之外還需要的東西。

## 範例清單

| 儲存庫 | 這個服務做什麼 | 需要什麼 |
| --- | --- | --- |
| [example-word-analogy](https://github.com/CodefyUI/example-word-analogy) | 詞向量類比查詢：輸入三個詞，輸出最接近的類比詞。 | 不需要任何東西 -- 完全離線。 |
| [example-tabular-predictor](https://github.com/CodefyUI/example-tabular-predictor) | 表格資料分類器：輸入特徵列，輸出類別預測。 | 不需要任何東西 -- 完全離線。 |
| [example-llm-document](https://github.com/CodefyUI/example-llm-document) | 文件摘要器：輸入文件內容，輸出摘要。 | 本機安裝的 [Ollama](https://ollama.com)。 |
| [example-mnist-train-serve](https://github.com/CodefyUI/example-mnist-train-serve) | 一個專案、兩張 graph：先在畫布上用 MNIST 訓練一個小型 CNN，再把「你自己訓練出來的」權重當成數字辨識 API 提供出去。 | 第一次訓練時會下載 MNIST（約 60 MB）。 |

## 為什麼它們是各自獨立的儲存庫

已發佈的服務是你自己擁有的程式碼：它值得有自己的歷史、自己的 remote、自己的 CI。把每個範例保持為一個真正的儲存庫（而不是打包在 CodefyUI 裡的幾個檔案），代表你在範例上練習的那條「從 clone 到發佈成 API」的路，跟你之後替自己的服務走的路一模一樣。每個範例儲存庫都能乾淨地通過 `cdui project validate .`，所以任何一個也都可以當成你自己專案做 [CI 驗證](./project-directories)的範本。
