---
sidebar_position: 5
title: Licensing FAQ
description: What AGPL-3.0 actually requires of CodefyUI users, whether running it internally triggers section 13, how custom nodes and plugins are treated, and what the commercial license covers.
---

# Licensing FAQ

CodefyUI uses a **dual-path licensing model**.

- **Open source path** — [AGPL-3.0-only](https://github.com/CodefyUI/CodefyUI/blob/main/LICENSE) for individual developers, small teams, education, research, community use, **and any other use case that can comply with AGPL-3.0**.
- **Commercial path** — for proprietary, closed-source, SaaS, OEM, enterprise, or other use that needs terms outside AGPL-3.0. [Contact the maintainers](https://github.com/CodefyUI/CodefyUI/issues).

The copyright holder is **CodefyUI** (https://github.com/CodefyUI) and the CodefyUI contributors — see [NOTICE](https://github.com/CodefyUI/CodefyUI/blob/main/NOTICE).

:::note This page is not legal advice
Everything below is the CodefyUI project's own reading of the license it publishes under. It is written so you can see what the project intends and check it against the license text yourself. It is not legal advice, and it does not create or modify any license. Where this page and [LICENSE](https://github.com/CodefyUI/CodefyUI/blob/main/LICENSE) disagree, LICENSE governs. If your situation needs certainty rather than an interpretation, the commercial license exists for exactly that.
:::

## Does running CodefyUI internally, unmodified, trigger AGPL section 13?

**No.**

AGPL-3.0 section 2 (*Basic Permissions*) says:

> This License explicitly affirms your unlimited permission to run the unmodified Program.

And the network clause in section 13 is conditioned on modification:

> Notwithstanding any other provision of this License, **if you modify the Program**, your modified version must prominently offer all users interacting with it remotely through a computer network (if your version supports such interaction) an opportunity to receive the Corresponding Source of your version [...]

So a company that installs CodefyUI as published and runs it on an internal server — for any number of employees, for commercial purposes, behind the firewall or not — has **no obligation under section 13 and needs no commercial license**. That is the ordinary open source path working as intended, not a loophole.

Two things follow, and both are worth stating plainly because evaluators often assume the opposite:

- **"Commercial use" is not what triggers the commercial license.** AGPL-3.0 places no restriction on using the software for profit. What triggers the commercial path is needing terms the AGPL does not give you — chiefly the ability to keep modifications closed.
- **Distributing an unmodified copy is also fine**, provided you pass along the license and the source (section 4/6). You do not need permission from the project to hand a colleague the installer.

## Does writing a custom node or a plugin count as modifying the Program?

**This is the question that actually matters, and the honest answer is: probably yes, and the project treats it as yes.**

The mechanics are not in dispute. A [custom node](/advanced/custom-nodes) is a Python file that subclasses `BaseNode` from `app.core.node_base` and is imported into the running backend. A [plugin pack](/advanced/plugins) is, in the plugin documentation's own words, "Python that runs in the CodefyUI process." Both import CodefyUI's own API, run inside CodefyUI's process, and are meaningless without it. CodefyUI publishes **no linking exception** — nothing in LICENSE carves out separately-authored modules the way the LGPL or a Classpath exception would.

**The project's interpretation:** a custom node or plugin pack that imports CodefyUI's Python API and executes in the CodefyUI process is part of a *modified version* of the Program for AGPL purposes. If you then let users interact with that deployment remotely over a network, section 13's source-offer requirement is engaged, and it reaches your node or plugin code too.

What that does and does not mean in practice:

| Situation | The project's reading |
|---|---|
| You write a custom node and use it yourself on your own machine. | Nothing is triggered. Section 13 is about users interacting **remotely over a network**; private use is not conveying. |
| You write a custom node, deploy CodefyUI on an internal server, and colleagues use it through the browser. | Section 13 is engaged. Offer the Corresponding Source — including your node — to those users. Inside one organization that is usually a link to an internal repository, not a public release. |
| You publish a plugin pack. | Publish it under AGPL-3.0-compatible terms. This is the normal case and the one the [plugin template](https://github.com/CodefyUI/CodefyUI-Plugin-Official) is set up for. |
| You want to ship a proprietary node or plugin, or run a modified CodefyUI as a service without releasing your changes. | This is what the commercial license is for. |

:::caution Where the uncertainty actually sits
Whether a plugin is a derivative work of its host is a genuinely contested question in copyright law, it varies by jurisdiction, and no court has settled it for the AGPL. The project states the reading above so you know what it intends and will not surprise you with a different one. It is not a legal opinion and it is not binding on anyone else's counsel. If you need an answer you can rely on, take the commercial license — it removes the question rather than answering it.
:::

## Do my graphs and my trained models fall under the AGPL?

**No, in the project's reading.** A `graph.json` you build on the canvas, the weights a training run produces, and any charts or exports are output from running the Program. AGPL-3.0 section 2 addresses this directly:

> The output from running a covered work is covered by this License only if the output, given its content, constitutes a covered work.

A graph description and a tensor of learned weights are your data, not CodefyUI's code. Nothing in the AGPL asks you to publish them.

The one thing to keep separate is the training *data* you feed in and any pretrained weights you download — those carry their own licenses from wherever you obtained them, entirely independent of CodefyUI's.

## What does the commercial license cover, and who grants it?

**Who grants it:** CodefyUI (https://github.com/CodefyUI), the copyright holder named in [NOTICE](https://github.com/CodefyUI/CodefyUI/blob/main/NOTICE). Inbound contributions are accepted under the Developer Certificate of Origin 1.1 plus an explicit dual-licensing term — see [CONTRIBUTING.md](https://github.com/CodefyUI/CodefyUI/blob/main/CONTRIBUTING.md) — so that contributed code can be offered on either path.

**What it covers:** terms outside AGPL-3.0, for the cases the open source path cannot serve.

- Proprietary or closed-source modifications you do not want to publish.
- Offering a modified CodefyUI as a hosted or SaaS product.
- OEM redistribution, or embedding CodefyUI in a product shipped under your own license.
- Shipping proprietary custom nodes or plugin packs (see the question above).
- Enterprise deployments whose internal policy forbids copyleft obligations regardless of whether they are actually triggered.

**What it does not do:** it does not take anything away from the open source path, and it is not a support contract. Terms and pricing are negotiated per case.

Start the conversation on the [issue tracker](https://github.com/CodefyUI/CodefyUI/issues). See also [COMMERCIAL-LICENSE.md](https://github.com/CodefyUI/CodefyUI/blob/main/COMMERCIAL-LICENSE.md).

## Third-party components

CodefyUI redistributes third-party software, both as Python dependencies and as compiled assets inside the prebuilt frontend bundle (React, KaTeX and its fonts, and others). Their copyright notices and license terms are collected in [THIRD_PARTY_NOTICES.md](https://github.com/CodefyUI/CodefyUI/blob/main/THIRD_PARTY_NOTICES.md), which ships inside the release tarball alongside `LICENSE` and `NOTICE`. All of them are permissive; none imposes copyleft obligations of its own.
