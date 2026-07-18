# Third-Party Notices

CodefyUI is licensed under the GNU Affero General Public License v3.0 only
(AGPL-3.0-only); see [LICENSE](LICENSE). Commercial licensing is described in
[COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).

This document lists the third-party software and data that CodefyUI
redistributes or depends on, together with their licenses. It is included in
the `frontend-dist.tar.gz` release asset alongside the repository `LICENSE`,
because that asset redistributes compiled third-party code.

License information below was verified on 2026-07-18 against the installed
packages (the `license` field of each package's `package.json` for npm
packages, and installed distribution metadata for Python packages). Versions
listed are the versions verified at that time; the authoritative version
ranges are `frontend/package.json` and `backend/pyproject.toml`.

## 1. Frontend dependencies (redistributed in compiled form)

The release asset `frontend-dist.tar.gz` contains the compiled frontend
bundle (`frontend/dist`). The production dependencies below are compiled into
that bundle and are therefore redistributed with every release.

| Package | Verified version | License | Upstream |
|---------|------------------|---------|----------|
| @dagrejs/dagre | 3.0.0 | MIT | https://github.com/dagrejs/dagre |
| @xyflow/react (React Flow) | 12.10.1 | MIT | https://github.com/xyflow/xyflow |
| katex | 0.16.45 | MIT | https://github.com/KaTeX/KaTeX |
| react | 19.2.4 | MIT | https://github.com/facebook/react |
| react-dom | 19.2.4 | MIT | https://github.com/facebook/react |
| react-katex | 3.1.0 | MIT | https://github.com/talyssonoc/react-katex |
| zustand | 5.0.12 | MIT | https://github.com/pmndrs/zustand |

The full production dependency closure (direct and transitive packages, as
resolved by `frontend/pnpm-lock.yaml`) was checked with
`pnpm licenses list --prod`. Every package in the closure is licensed under
one of:

- MIT (all packages not listed below);
- ISC (the D3 modules d3-color, d3-dispatch, d3-drag, d3-interpolate,
  d3-selection, d3-timer, d3-transition, d3-zoom);
- BSD-3-Clause (d3-ease).

### 1.1 KaTeX and bundled fonts

KaTeX (https://katex.org, https://github.com/KaTeX/KaTeX) is used for math
rendering. The compiled bundle includes KaTeX's JavaScript and CSS, and the
KaTeX web fonts (the `KaTeX_*` font files in woff, woff2, and ttf formats)
are copied into `frontend/dist/assets` by the build. These fonts ship inside
`frontend-dist.tar.gz`. The fonts are part of the KaTeX distribution and are
covered by the same MIT license as KaTeX itself, reproduced here from the
`LICENSE` file of the katex npm package:

```
The MIT License (MIT)

Copyright (c) 2013-2020 Khan Academy and other contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. Backend Python dependencies (not redistributed)

CodefyUI release artifacts do **not** redistribute the backend's Python
dependencies. Users install them from PyPI (or another index of their choice)
at install time. The direct runtime dependencies declared in
`backend/pyproject.toml` `[project.dependencies]` are listed below with their
license names for reference. This is a best-effort summary verified from
installed distribution metadata; consult each project for the authoritative
license text.

| Package | License |
|---------|---------|
| fastapi | MIT |
| starlette | BSD-3-Clause |
| uvicorn | BSD-3-Clause |
| pydantic | MIT |
| pydantic-settings | MIT |
| websockets | BSD-3-Clause |
| python-multipart | Apache-2.0 |
| numpy | BSD-3-Clause (with bundled components under 0BSD, MIT, Zlib, CC0-1.0) |
| matplotlib | Matplotlib License (PSF-based) |
| Pillow | MIT-CMU |
| torch | BSD-3-Clause |
| torchvision | BSD-3-Clause |
| gymnasium | MIT |
| safetensors | Apache-2.0 |
| datasets | Apache-2.0 |
| kagglehub | Apache-2.0 |
| kagglesdk | Apache-2.0 |
| pyarrow | Apache-2.0 |
| pandas | BSD-3-Clause |
| tiktoken | MIT |
| tokenizers | Apache-2.0 |
| huggingface_hub | Apache-2.0 |
| scikit-learn | BSD-3-Clause |
| platformdirs | MIT |
| psutil | BSD-3-Clause |
| httpx | BSD-3-Clause |
| tomli (Python < 3.11 only) | MIT |

Notable transitive dependencies: certifi is licensed under MPL-2.0, and tqdm
under MPL-2.0 AND MIT. MPL-2.0 is a file-level copyleft license; CodefyUI
uses these packages unmodified and does not redistribute them.

The torch and torchvision wheels published on PyPI bundle additional native
components (for example CUDA libraries in GPU builds) under their own terms;
consult those projects' distributions for details.

## 3. MNIST sample data

The source repository contains a copy of the MNIST database of handwritten
digits under `backend/data/MNIST/raw/` (IDX-format image and label files),
used as sample data for the built-in training examples. This data is not part
of `frontend-dist.tar.gz`; it is distributed with the source repository.

The MNIST database was created by Yann LeCun (Courant Institute, NYU),
Corinna Cortes (Google Labs), and Christopher J. C. Burges (Microsoft
Research), derived from NIST Special Databases 1 and 3. Source:
http://yann.lecun.com/exdb/mnist/.
