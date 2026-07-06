---
sidebar_position: 7.9
title: Example Projects
description: Official ready-to-clone example services -- each one a standalone CodefyUI project repository you can run, publish, and fork.
---

# Example Projects

Each official example is a standalone [project directory](./project-directories.md)
hosted as its own repository under the CodefyUI organization. Clone one, start the
server on it, and you have a working service on your canvas -- then publish it as a
real HTTP API and push the repo anywhere you like. They are also honest templates:
fork one and swap in your own graph.

Every repo follows the same three commands:

```bash
git clone https://github.com/CodefyUI/<example-name>
cd <example-name>
cdui start --project .
```

Open the editor at the URL the server prints, load the graph from the toolbar, and
press Run. Each README walks through the full publish flow (commit, `cdui project
publish`, mint an API key, invoke with curl) and lists anything the example needs
beyond CodefyUI itself.

## The examples

| Repository | What the service does | Needs |
| --- | --- | --- |
| [example-word-analogy](https://github.com/CodefyUI/example-word-analogy) | Word-vector analogy lookup: three words in, nearest analogy words out. | Nothing -- fully offline. |
| [example-tabular-predictor](https://github.com/CodefyUI/example-tabular-predictor) | Tabular classifier: feature rows in, class predictions out. | Nothing -- fully offline. |
| [example-llm-document](https://github.com/CodefyUI/example-llm-document) | Document summarizer: document text in, summary out. | A local [Ollama](https://ollama.com) install. |
| [example-mnist-train-serve](https://github.com/CodefyUI/example-mnist-train-serve) | One project, two graphs: train a small CNN on MNIST on the canvas, then serve YOUR trained weights as a digit-recognition API. | MNIST download on first training run (about 60 MB). |

## Why they are separate repositories

A published service is code you own: it deserves its own history, its own remote,
its own CI. Keeping each example as a real repository (instead of files bundled
inside CodefyUI) means the clone-to-published-API path you practice on an example
is exactly the path you will use for your own services. Each example repo runs
`cdui project validate .` cleanly, so any of them also works as a template for
[CI validation](./project-directories.md#4-validate-the-ci-gate) of your own project.
