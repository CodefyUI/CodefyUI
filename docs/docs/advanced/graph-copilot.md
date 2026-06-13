---
sidebar_position: 5
title: Graph Copilot
description: Chat with an AI assistant to generate, tune, and improve your node graph — powered by the plugin frontend extension API and a unified LLM streaming proxy.
---

# Graph Copilot

Graph Copilot is a CodefyUI plugin that adds a chat panel to the editor. You describe what you want in plain language and the AI generates a sequence of graph operations (add nodes, connect ports, set parameters) that are applied atomically — one undo step per AI edit. You can stop mid-stream, retry failed requests, and resume a conversation across sessions.

:::note Availability
Graph Copilot is built on two CodefyUI features: the [plugin frontend extension API](/advanced/plugin-frontend-extensions) and the unified LLM proxy endpoint (`/api/llm/chat`). Both are in CodefyUI **1.3.0** and later. If `cdui --version` reports an older version, run `cdui update` before installing.
:::

## Installation

```bash
cdui plugin install treeleaves30760/CodefyUI-Plugin-Graph-Copilot
```

Then reload the editor (press F5 or close and reopen the tab). The Graph Copilot panel appears as a floating widget in the editor.

Plugin source and issues: [github.com/treeleaves30760/CodefyUI-Plugin-Graph-Copilot](https://github.com/treeleaves30760/CodefyUI-Plugin-Graph-Copilot)

## Quick start

1. Install the plugin (above) and reload the editor.
2. Click the round **Graph Copilot** button in the bottom-right corner of the canvas to open the chat panel.
3. Click the **settings** (gear) icon, choose a provider, and paste your API key — or, for **OpenAI Codex**, click **Sign in** and approve in the tab that opens. Pick a model (use **Load list** to fetch the provider's models).
4. Type a request such as `Build a small MLP classifier` and press **Enter**.
5. Watch the nodes appear and wire up on the canvas as the AI streams its plan. Press **Ctrl+Z** once to undo the whole edit, or keep chatting to refine it.

The provider and key only need to be set once — they persist in your browser. The rest of this page covers each part in detail.

## Choosing an LLM provider

Open the settings icon in the Graph Copilot panel to configure your provider and key.

| Provider | Notes |
|----------|-------|
| **OpenAI API** | Standard `https://api.openai.com/v1` endpoint. Requires an OpenAI API key. Billed per token. |
| **OpenAI Codex (ChatGPT sign-in)** | Uses the ChatGPT web session. No separate API key required, but subject to ChatGPT usage quotas and OpenAI's ToS — use of the internal session API for automation is a gray area not officially sanctioned by OpenAI. |
| **OpenRouter** | Aggregates many providers under one key. Set the base URL to `https://openrouter.ai/api/v1` and select your preferred model. |
| **Claude API** | Anthropic's API, accessed through CodefyUI's proxy which translates the OpenAI-compatible request format. Requires an Anthropic API key. |
| **Custom (OpenAI-compatible)** | Any server that implements the OpenAI `/chat/completions` endpoint — for example, a local Ollama instance: `http://localhost:11434/v1`. Set the base URL and optionally a key. |

## Key handling

API keys are stored in `localStorage` under a namespace private to Graph Copilot and never sent to the CodefyUI backend or any third party — only to the provider you have configured. The local CodefyUI backend (`/api/llm/chat`) acts as a streaming proxy, forwarding your request to the configured provider and streaming the response back; it does not log or persist keys or message content.

## Usage

### Sending a request

Type your request in the chat input and press Enter (or click Send). Examples:

- "Add a two-layer MLP with ReLU activations"
- "Connect the CrossEntropy node to the output of the last Linear"
- "Set the hidden size on Linear-1 to 512"

The AI returns a plan followed by a list of operations. You can see each operation as a chip label (e.g., "add Linear", "add ReLU", "connect") as they are applied.

### Conversation history

The chat history for the current graph is saved in `localStorage`. When you reopen the editor or reload the page, Graph Copilot resumes the conversation where you left off.

### Stop and retry

Click **Stop** during a stream to cancel the in-flight request. The partial response is discarded. Click **Retry** on any AI message to resend that turn with the same context.

### Undoing AI edits

Every AI edit is a single undo snapshot. Press **Ctrl+Z** (or Cmd+Z on macOS) once to undo the entire batch of operations from the last AI response.

## Tips

- Give context about what you are building: "I am building a vision classifier with a ResNet backbone" helps the AI make better choices.
- If the AI adds a node type that does not exist in your palette, it will be skipped and reported — use `cdui plugin install` to add the required pack first.
- Graph Copilot reads the current graph state and the full node palette before each request, so it knows what types are available and what is already on the canvas.

## See also

- [Plugin Frontend Extensions](/advanced/plugin-frontend-extensions) — the JS API that Graph Copilot is built on.
- [Plugins](/advanced/plugins) — the plugin pack system.
- [API Reference](/advanced/api-reference) — the `/api/llm/chat` streaming endpoint.
