---
sidebar_position: 5
title: Graph Copilot
description: Use an AI assistant to generate and edit node graphs through the plugin frontend extension API and LLM streaming proxy.
---

# Graph Copilot

Graph Copilot is an agent workbench plugin for the editor. It builds and edits the graph by conversation: describe what you want in plain language and the AI adds nodes, connects ports, and sets parameters, applying each batch of operations as one undo step. It can also run the current graph after you approve, run bounded experiment studies on in-memory copies of the graph, and keep a browsable conversation history.

:::note Availability
Graph Copilot is built on two CodefyUI features: the [plugin frontend extension API](/advanced/plugin-frontend-extensions) and the unified LLM proxy endpoint (`/api/llm/chat`). Both are in CodefyUI **1.3.0** and later. If `cdui --version` reports an older version, run `cdui update` before installing.
:::

## Installation

```bash
cdui plugin install graph-copilot
```

You can also open the [Plugin Center](/advanced/plugins#plugin-center) in the editor and install `graph-copilot` from the catalog. Installation through the Plugin Center loads the panel immediately. After installing from a terminal, reload the editor by pressing F5. The Graph Copilot panel appears as a floating widget in the editor.

Plugin source and issues: [github.com/CodefyUI/CodefyUI-Plugin-Graph-Copilot](https://github.com/CodefyUI/CodefyUI-Plugin-Graph-Copilot)

## Quick start

1. Install the plugin (above).
2. Click the **Graph Copilot** button (a chat-bubble icon) in the bottom-right corner of the editor to open the panel.
3. Click **Settings** (gear icon), choose a provider, and paste your API key. To use **OpenAI Codex (ChatGPT)**, click **Sign in** and approve access in the tab that opens. Select a model; use **Refresh** to fetch the provider's model list.
4. Type a request such as `Build a small MLP classifier` and press **Enter**.
5. The AI streams its plan while adding and connecting nodes on the canvas. Press **Ctrl+Z** to undo its last batch of operations, or send another message to refine it.

The browser stores the provider and key, so you only need to configure them once. The following sections describe each feature.

## Panel views

The panel has four views, chosen with the buttons at its top:

- **Chat**: the conversation with the agent.
- **Lab**: the experiment studies you asked for in Chat. Variants of the current graph run on in-memory copies and are ranked; the canvas does not change.
- **History**: earlier conversations. Click one to resume it, or start a new one with **+ New chat**.
- **Settings**: provider, model and reasoning effort, API key or sign-in, and run notifications.

## Choosing an LLM provider

Click the **Settings** (gear) icon in the Graph Copilot panel to configure the provider and key.

| Provider | Notes |
|----------|-------|
| **OpenAI API** | Standard `https://api.openai.com/v1` endpoint. Requires an OpenAI API key. Billed per token. |
| **OpenAI Codex (ChatGPT)** | Uses OAuth with your ChatGPT account through the Codex CLI PKCE flow and client ID. It uses ChatGPT subscription quota rather than API credits and remains subject to ChatGPT usage limits and OpenAI's terms. Tokens are stored **on the server** in `llm/codex_auth.json` under the user-data directory. Everyone who uses that server shares one signed-in account, and **Sign out** clears the tokens for everyone. The OAuth callback listens on `localhost:1455` (or `1457`) in the server process. Complete sign-in within 5 minutes in a browser on the machine that runs the server. |
| **OpenRouter** | Aggregates many providers under one key. The proxy sends requests to `https://openrouter.ai/api/v1`; select your preferred model. |
| **Claude API** | Anthropic's API, accessed through CodefyUI's proxy. The proxy translates the OpenAI-compatible request format. Requires an Anthropic API key. |
| **Custom (OpenAI-compatible)** | Any server that implements the OpenAI `/chat/completions` endpoint. For example, you can use a local Ollama instance at `http://localhost:11434/v1`. Set the base URL and, if required, a key. |

The proxy also provides `POST /api/llm/models`, which lists a provider's models for **Refresh**. `POST /api/llm/codex/login`, `GET /api/llm/codex/status`, and `POST /api/llm/codex/logout` support ChatGPT sign-in. The same controls are available under **Settings → LLM Providers**. Only **OpenAI API** and **OpenAI Codex (ChatGPT)** use `reasoning_effort`. The value must be 1–64 characters of lowercase letters, digits, `_` or `-`, starting with a letter; any other value is `422` for every provider. The proxy refuses `ultra` for **OpenAI Codex (ChatGPT)** (`400`) and forwards other values unchanged. The editor supplies the session token required by the `POST` routes.

## Key handling

API keys are stored in `localStorage` under a namespace private to Graph Copilot. Each request sends the selected key to the local CodefyUI backend. `/api/llm/chat` forwards the key and messages to the configured provider and streams the response back. It does not log or persist the key or messages. Each provider has fixed upstream hosts; only **Custom** uses a base URL that you supply. These keys are separate from the environment variables `CODEFYUI_OPENAI_API_KEY` / `OPENAI_API_KEY` and `CODEFYUI_ANTHROPIC_API_KEY` / `ANTHROPIC_API_KEY`, which only the `LLMChat` node reads; the proxy reads none of them.

## Usage

### Sending a request

Type your request in the chat input and press Enter (or click **Send**). Examples:

- "Add a two-layer MLP with ReLU activations"
- "Connect the CrossEntropy node to the output of the last Linear"
- "Set the hidden size on Linear-1 to 512"

The AI returns a plan and then applies operations. Each tool call appears as a step under the reply, for example **Edit graph** with `add_node ×2, connect ×1`, and shows whether it succeeded.

### Conversation history

Conversations are stored in the browser (Graph Copilot's namespaced `localStorage`), newest first, at most 50; past that the oldest is dropped. They are not tied to a graph. Opening the editor starts a new conversation; open **History** to resume an earlier one.

### Stop and retry

Click **Stop** during a stream to cancel the in-flight request; text that already streamed stays in the conversation. When a request fails, the panel shows **Request failed** with a **Retry** button that resends your last message and its attachments.

### Undoing AI edits

Each batch of operations the AI applies is one undo snapshot. Press **Ctrl+Z** (or Cmd+Z on macOS) once to undo the last batch. When the AI corrects itself within one reply, it applies more than one batch, and each needs its own Ctrl+Z.

## Tips

- Give context about what you are building: "I am building a vision classifier with a ResNet backbone" helps the AI make better choices.
- If the AI adds a node type that does not exist in your palette, it will be skipped and reported — use `cdui plugin install` to add the required pack first.
- Graph Copilot reads the current graph state and the full node palette before each request, so it knows what types are available and what is already on the canvas.

## See also

- [Graph Copilot documentation](https://codefyui.github.io/CodefyUI-Plugin-Graph-Copilot/) — experiments, attachments, graph runs, and the agent contract.
- [Plugin Frontend Extensions](/advanced/plugin-frontend-extensions) — the JS API that Graph Copilot is built on.
- [Plugins](/advanced/plugins) — the plugin pack system.
- [API Reference](/advanced/api-reference) — the `/api/llm/chat` streaming endpoint.
