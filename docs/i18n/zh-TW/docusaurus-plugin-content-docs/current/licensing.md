---
sidebar_position: 5
title: 授權常見問題
description: AGPL-3.0 對 CodefyUI 使用者實際的要求是什麼、內部執行會不會觸發第 13 條、自訂節點與外掛怎麼算、商業授權涵蓋哪些範圍，以及貢獻者同意了什麼。
---

# 授權常見問題

CodefyUI 採用**雙軌授權模式**。

- **開源路徑** — [AGPL-3.0-only](https://github.com/CodefyUI/CodefyUI/blob/main/LICENSE)，適用於個人開發者、小型團隊、教育、研究、社群使用，**以及任何能遵守 AGPL-3.0 的其他使用情境**。
- **商業路徑** — 若需要閉源、SaaS、OEM、企業部署，或其他不適合 AGPL-3.0 的條款，請[聯絡維護者](https://github.com/CodefyUI/CodefyUI/issues)。

著作權人為 **CodefyUI**（https://github.com/CodefyUI）及 CodefyUI 貢獻者——見 [NOTICE](https://github.com/CodefyUI/CodefyUI/blob/main/NOTICE)。

:::note 本頁不是法律意見
以下全部是 CodefyUI 專案自己對所採用授權條款的解讀。這樣寫是為了讓你看得到專案的意圖，並能自己拿授權原文對照。它不是法律意見，也不會建立或修改任何授權。若本頁與 [LICENSE](https://github.com/CodefyUI/CodefyUI/blob/main/LICENSE) 有出入，以 LICENSE 為準。如果你的情況需要的是確定性而不是解讀，商業授權存在的目的正是如此。
:::

## 未經修改在內部執行，會觸發 AGPL 第 13 條嗎？

**不會。**

AGPL-3.0 第 2 條（*Basic Permissions*）寫著：

> This License explicitly affirms your unlimited permission to run the unmodified Program.
>
> （本授權明確確認你擁有執行未修改程式的無限制權利。）

而第 13 條的網路條款，前提是「你修改了程式」：

> Notwithstanding any other provision of this License, **if you modify the Program**, your modified version must prominently offer all users interacting with it remotely through a computer network (if your version supports such interaction) an opportunity to receive the Corresponding Source of your version [...]
>
> （儘管本授權另有規定，**若你修改了本程式**，你的修改版本必須向所有透過電腦網路遠端與之互動的使用者，明顯地提供取得該版本 Corresponding Source 的機會 [...]）

所以，一家公司照發行版本安裝 CodefyUI 並跑在內部伺服器上——不論多少員工使用、是否為營利目的、在不在防火牆後面——**對第 13 條沒有任何義務，也不需要商業授權**。這是開源路徑本來就該有的樣子，不是鑽漏洞。

有兩件事因此成立，而且值得明講，因為評估者常常反過來假設：

- **「商業使用」本身不是觸發商業授權的原因。** AGPL-3.0 完全不限制以營利為目的使用軟體。真正觸發商業路徑的，是你需要 AGPL 給不了你的條款——主要是「把修改保持閉源」的能力。
- **散布未經修改的副本同樣沒問題**，只要一併附上授權與原始碼（第 4 / 6 條）。把安裝程式給同事，不需要向專案取得許可。

## 寫自訂節點或外掛，算不算修改本程式？

**這才是真正重要的問題，而誠實的答案是：很可能算，而且專案就是這樣認定的。**

事實面沒有爭議。[自訂節點](/advanced/custom-nodes)是一個 Python 檔案，繼承 `app.core.node_base` 的 `BaseNode`，並被 import 進執行中的後端。[外掛包](/advanced/plugins)用外掛文件自己的話說，就是「在 CodefyUI 行程內執行的 Python」。兩者都 import CodefyUI 自己的 API、都在 CodefyUI 的行程裡執行，離開 CodefyUI 就沒有意義。而 CodefyUI **沒有提供 linking exception**——LICENSE 裡沒有任何條文像 LGPL 或 Classpath exception 那樣，把獨立撰寫的模組切出去。

**專案的解讀：** 一個 import CodefyUI Python API、並在 CodefyUI 行程內執行的自訂節點或外掛包，就 AGPL 而言屬於本程式*修改版本*的一部分。如果你接著讓使用者透過網路遠端使用這個部署，第 13 條的原始碼提供義務就被觸發，而且範圍涵蓋你的節點或外掛程式碼。

實務上這代表什麼、不代表什麼：

| 情境 | 專案的解讀 |
|---|---|
| 你寫了一個自訂節點，只在自己電腦上用。 | 什麼都不會觸發。第 13 條談的是**透過網路遠端互動**的使用者；私人使用不構成 conveying。 |
| 你寫了自訂節點，把 CodefyUI 架在內部伺服器上，同事透過瀏覽器使用。 | 第 13 條被觸發。請向這些使用者提供 Corresponding Source——包含你的節點。在同一個組織內，這通常是一個內部 repository 的連結，而不是公開釋出。 |
| 你公開發布一個外掛包。 | 請用與 AGPL-3.0 相容的條款發布。這是最常見的情況，也是[外掛範本](https://github.com/CodefyUI/CodefyUI-Plugin-Official)預設的做法。 |
| 你想出貨閉源的節點或外掛，或想把修改過的 CodefyUI 當服務跑而不釋出修改。 | 這就是商業授權存在的原因。 |

:::caution 不確定性到底在哪裡
「外掛算不算其宿主的衍生著作」在著作權法上是真的有爭議的問題，各法域見解不同，也還沒有法院針對 AGPL 給出定論。專案把上面的解讀寫出來，是為了讓你知道專案的意圖，不會事後拿另一套說法來找你。它不是法律意見，也不拘束任何其他人的法務。如果你需要的是一個可以依賴的答案，請取得商業授權——它不是回答這個問題，而是讓這個問題消失。
:::

## 我建的圖和訓練出來的模型會受 AGPL 約束嗎？

**依專案的解讀，不會。** 你在畫布上建出來的 `graph.json`、訓練產生的權重，以及任何圖表或匯出檔，都是執行本程式的**輸出**。AGPL-3.0 第 2 條直接處理了這件事：

> The output from running a covered work is covered by this License only if the output, given its content, constitutes a covered work.
>
> （執行受本授權涵蓋之著作所產生的輸出，只有在其內容本身構成受涵蓋著作時，才受本授權涵蓋。）

一份圖的描述、一組學到的權重張量，是你的資料，不是 CodefyUI 的程式碼。AGPL 沒有要求你公開它們。

唯一要分開看的是你餵進去的訓練**資料**以及你下載的預訓練權重——那些帶著它們各自來源的授權，與 CodefyUI 完全無關。

## 商業授權涵蓋什麼？由誰授予？

**由誰授予：** CodefyUI（https://github.com/CodefyUI），也就是 [NOTICE](https://github.com/CodefyUI/CodefyUI/blob/main/NOTICE) 中列名的著作權人。外部貢獻以 Developer Certificate of Origin 1.1 加上一條明示的雙軌授權條款收受——見 [CONTRIBUTING.md](https://github.com/CodefyUI/CodefyUI/blob/main/CONTRIBUTING.md)——以確保貢獻的程式碼可以走任一條路徑。

**涵蓋什麼：** AGPL-3.0 以外的條款，用於開源路徑服務不了的情況。

- 你不想公開的閉源或專有修改。
- 把修改過的 CodefyUI 當成託管服務或 SaaS 產品提供。
- OEM 轉散布，或把 CodefyUI 內嵌到以你自己授權出貨的產品中。
- 出貨閉源的自訂節點或外掛包（見上一題）。
- 內部政策禁止 copyleft 義務的企業部署——不論該義務實際上有沒有被觸發。

**它不做什麼：** 它不會從開源路徑拿走任何東西，也不是技術支援合約。條款與價格逐案商議。

請從[問題追蹤器](https://github.com/CodefyUI/CodefyUI/issues)開始接洽。另見 [COMMERCIAL-LICENSE.md](https://github.com/CodefyUI/CodefyUI/blob/main/COMMERCIAL-LICENSE.md)。

## 貢獻者同意了什麼？商業授權收入會分給貢獻者嗎？

三條基本規則，完整內容見 [CONTRIBUTING.md](https://github.com/CodefyUI/CodefyUI/blob/main/CONTRIBUTING.md)：

- **每一位貢獻者都會被記錄。** git 歷史會永久保留每位貢獻者的名字與 email，GitHub 的 contributors 圖表也由同一份記錄產生，而且貢獻者保有自己作品的著作權。
- **授權是自願且無償的。** 貢獻以雙軌授權給專案——AGPL-3.0-only 與商業授權。當著作權人銷售包含貢獻程式碼的商業授權時，不需要向這些程式碼的貢獻者支付權利金、分潤或任何其他報酬。
- **重大貢獻可以另行討論。** 若貢獻者認為某項貢獻夠重大、應該有不同的表彰或條件，可以在貢獻前後向維護者提出討論。任何超出上述預設的安排，都需要與著作權人明確書面約定。

對商業授權買方而言，這代表只有單一授權人，且授權不附帶任何來自個別貢獻者的付款主張。

## 第三方元件

CodefyUI 會轉散布第三方軟體，包含 Python 相依套件，以及預先建置的前端 bundle 內的編譯資產（React、KaTeX 及其字型等）。它們的著作權聲明與授權條款收錄在 [THIRD_PARTY_NOTICES.md](https://github.com/CodefyUI/CodefyUI/blob/main/THIRD_PARTY_NOTICES.md)，該檔會與 `LICENSE`、`NOTICE` 一起放進 release tarball。這些元件全部是寬鬆授權，沒有任何一個帶有自己的 copyleft 義務。
