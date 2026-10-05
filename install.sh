#!/usr/bin/env bash
# CodefyUI 一鍵安裝腳本
# 用法：curl -fsSL https://raw.githubusercontent.com/CodefyUI/CodefyUI/main/install.sh | bash
#
# 環境變數：
#   CODEFYUI_DIR          自訂安裝路徑（預設 $HOME/CodefyUI）
#   CODEFYUI_RELEASE_TAG  指定要安裝的 release tag（預設 latest）；指定版本時，即使 FORCE_BUILD=1 也裝該版本
#   CODEFYUI_FORCE_BUILD  設為 1 強制本地 build（會額外裝 Node + pnpm）；沒指定版本時安裝 main
set -euo pipefail

# ── 顏色 ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

REPO="https://github.com/CodefyUI/CodefyUI.git"
RELEASE_REPO="CodefyUI/CodefyUI"
RELEASE_ASSET="frontend-dist.tar.gz"
INSTALL_DIR="${CODEFYUI_DIR:-$HOME/CodefyUI}"
RELEASE_TAG="${CODEFYUI_RELEASE_TAG:-latest}"
FORCE_BUILD="${CODEFYUI_FORCE_BUILD:-0}"

step() { echo -e "\n${BLUE}==>${NC} ${BOLD}$*${NC}"; }
ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
warn() { echo -e "  ${YELLOW}!${NC} $*"; }
die()  { echo -e "\n${RED}✗ 錯誤：$*${NC}" >&2; exit 1; }

# root 不需要 sudo
SUDO=""
[[ "$(id -u)" != "0" ]] && SUDO="sudo"

# ── OS 偵測 ───────────────────────────────────────────────────────────
if [[ "$OSTYPE" == "darwin"* ]]; then
  OS="macos"
elif [[ -f /etc/debian_version ]]; then
  OS="debian"
elif [[ -f /etc/redhat-release ]]; then
  OS="redhat"
else
  OS="unknown"
fi

# ── 套件安裝 helper ───────────────────────────────────────────────────
pkg_install() {
  case "$OS" in
    macos)
      if ! command -v brew &>/dev/null; then
        warn "安裝 Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      fi
      brew install "$@" ;;
    debian)
      $SUDO apt-get update -qq
      $SUDO apt-get install -y "$@" ;;
    redhat)
      $SUDO yum install -y "$@" ;;
    *)
      die "不支援的作業系統，請手動安裝：$*" ;;
  esac
}

# 用 nvm 安裝 Node + 全域 pnpm。只有需要本地 build 時才呼叫。
install_node_toolchain() {
  step "Node.js（透過 nvm，僅本地 build 路徑使用）"
  local node_min=24
  local node_ok=false
  if command -v node &>/dev/null; then
    local current_major
    current_major="$(node --version | sed 's/^v//' | cut -d. -f1)"
    [[ "$current_major" =~ ^[0-9]+$ ]] && (( current_major >= node_min )) && node_ok=true
  fi
  if [[ "$node_ok" != "true" ]]; then
    warn "未安裝或版本 < ${node_min}，透過 nvm 安裝 Node ${node_min}..."
    export NVM_DIR="$HOME/.nvm"
    if [[ ! -s "$NVM_DIR/nvm.sh" ]]; then
      curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
    fi
    # shellcheck disable=SC1091
    [[ -s "$NVM_DIR/nvm.sh" ]] && source "$NVM_DIR/nvm.sh"
    nvm install "$node_min" --no-progress
    nvm use "$node_min"
  fi
  ok "Node.js $(node --version)"

  step "pnpm"
  if ! command -v pnpm &>/dev/null; then
    warn "未安裝，正在安裝..."
    if command -v npm &>/dev/null; then
      npm install -g pnpm --silent
    else
      curl -fsSL https://get.pnpm.io/install.sh | sh -
      export PNPM_HOME="$HOME/.local/share/pnpm"
      export PATH="$PNPM_HOME:$PATH"
    fi
  fi
  ok "pnpm $(pnpm --version)"
}

# 嘗試下載預編 dist，成功回傳 0、失敗回傳非 0。
fetch_release_dist() {
  local dist_dir="$INSTALL_DIR/frontend/dist"
  local url
  if [[ "$RELEASE_TAG" == "latest" ]]; then
    url="https://github.com/${RELEASE_REPO}/releases/latest/download/${RELEASE_ASSET}"
  else
    url="https://github.com/${RELEASE_REPO}/releases/download/${RELEASE_TAG}/${RELEASE_ASSET}"
  fi

  local tmpdir tarball
  tmpdir="$(mktemp -d)"
  tarball="$tmpdir/$RELEASE_ASSET"
  trap 'rm -rf "$tmpdir"' RETURN

  echo -e "  ${BOLD}下載：${NC}$url"
  if ! curl -fsSL --connect-timeout 10 --retry 2 -o "$tarball" "$url"; then
    warn "下載失敗（網路問題或 release 還沒附這個 asset）"
    return 1
  fi

  rm -rf "$dist_dir"
  mkdir -p "$dist_dir"
  if ! tar -xzf "$tarball" -C "$dist_dir"; then
    warn "Tarball 解壓失敗"
    rm -rf "$dist_dir"
    return 1
  fi
  if [[ ! -f "$dist_dir/index.html" ]]; then
    warn "解壓後找不到 index.html，asset 內容可能有誤"
    rm -rf "$dist_dir"
    return 1
  fi
  ok "Prebuilt dist 解壓至 $dist_dir"
}

# 解析要安裝的 release tag：印出實際 tag（把 "latest" 解析成具體版號），兩種
# 查法都失敗就什麼都不印。先問 GitHub API（不依賴 jq）；API 對未登入的 IP 每
# 小時只給 60 次，同一個 NAT 後面的整間教室很快就用完，所以再看 release 頁面：
# 它不受這個限制，會 302 到 .../releases/tag/<tag>。一律回傳 0。
resolve_release_tag() {
  if [[ "$RELEASE_TAG" != "latest" ]]; then
    echo "$RELEASE_TAG"
    return 0
  fi
  local api="https://api.github.com/repos/${RELEASE_REPO}/releases/latest"
  local page="https://github.com/${RELEASE_REPO}/releases/latest"
  local tag="" final=""
  tag="$(curl -fsSL --connect-timeout 10 --retry 2 "$api" 2>/dev/null \
    | grep -m1 '"tag_name"' \
    | sed -E 's/.*"tag_name"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')" || true
  if [[ -n "$tag" ]]; then
    echo "$tag"
    return 0
  fi
  # 用 HEAD 跟著轉址走，只看最後停在哪個網址；curl 失敗時的輸出不採用。
  if final="$(curl -fsSLI -o /dev/null -w '%{url_effective}' --connect-timeout 10 --retry 2 "$page" 2>/dev/null)"; then
    case "$final" in
      */releases/tag/*) tag="${final##*/releases/tag/}" ;;
    esac
  fi
  # 只接受單純的版號字元：還沒有 release 時頁面轉到別處，URL 編碼過的（含 %）
  # 也當作查不到。
  if [[ "$tag" =~ ^[A-Za-z0-9._+-]+$ ]]; then
    echo "$tag"
  fi
  return 0
}

# ══════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}╔══════════════════════════════════════╗${NC}"
echo -e "${BOLD}║        CodefyUI  安裝程式            ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════╝${NC}"
echo -e "  安裝目錄：${BOLD}$INSTALL_DIR${NC}"
echo -e "  Release tag：${BOLD}$RELEASE_TAG${NC}"
[[ "$FORCE_BUILD" == "1" ]] && echo -e "  ${YELLOW}強制本地 build（CODEFYUI_FORCE_BUILD=1）${NC}"

# ── git ───────────────────────────────────────────────────────────────
step "git"
if ! command -v git &>/dev/null; then
  warn "未安裝，正在安裝..."
  pkg_install git
fi
ok "$(git --version)"

# ── uv ────────────────────────────────────────────────────────────────
step "uv"
if ! command -v uv &>/dev/null; then
  warn "未安裝，正在安裝..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "$(uv --version)"

# ── Python 3（由 uv 提供）─────────────────────────────────────────────
step "Python 3"
uv python install 3.11
PYTHON="$(uv python find 3.11)"
[[ -x "$PYTHON" ]] || die "uv python find 回傳無效路徑：$PYTHON"
ok "$("$PYTHON" --version) ($PYTHON)"

# ── 解析 release tag ──────────────────────────────────────────────────
# 預編 dist 路徑（非 FORCE_BUILD）會把 backend 鎖到與 dist 同一個 release
# tag，避免「main 後端 + 舊 release 前端」版本漂移——那正是 cdui start 後
# 打開 localhost:8000 卻無法執行的根因（舊前端不會跟新後端做 token bootstrap，
# 所有寫入請求被 auth_guard 擋成 403）。
# 查不到 tag 就停止，不會改裝 main（要 main 請明確設 CODEFYUI_FORCE_BUILD=1）。
# 明確指定的 CODEFYUI_RELEASE_TAG 在 FORCE_BUILD 下也照用，前端則在本機建置。
PINNED_TAG=""
if [[ "$FORCE_BUILD" != "1" || "$RELEASE_TAG" != "latest" ]]; then
  # 放在 if 的條件裡：在 set -e 之下，單獨一行的指派一失敗，腳本就在那裡
  # 無聲結束，連下面的訊息都印不出來。
  if PINNED_TAG="$(resolve_release_tag)" && [[ -n "$PINNED_TAG" ]]; then
    # fetch_release_dist 用 RELEASE_TAG 決定下載哪個 dist；鎖成解析後的具體
    # tag，確保前後端來自同一版。
    RELEASE_TAG="$PINNED_TAG"
    echo -e "  ${BOLD}鎖定 release：${NC}${PINNED_TAG}（前後端同版）"
  else
    die "無法查到最新的 release 版本（GitHub 無法連線或達到速率限制），已停止，尚未下載或更新 CodefyUI。
  請稍後再試，或設定 CODEFYUI_RELEASE_TAG=<版本> 安裝指定版本；要改裝開發中的 main，請設定 CODEFYUI_FORCE_BUILD=1（會另外安裝 Node.js 與 pnpm）。
  Could not look up the latest release (GitHub unreachable or rate-limited). Stopped before downloading or updating CodefyUI.
  Try again later, or set CODEFYUI_RELEASE_TAG=<tag> to install that release; to install the development branch main instead, set CODEFYUI_FORCE_BUILD=1 (this also installs Node.js and pnpm).
  版本列表 / Releases: https://github.com/${RELEASE_REPO}/releases"
  fi
fi

# ── Clone / Update ────────────────────────────────────────────────────
# Clone 先做，這樣後面才能把 dist 解壓到 $INSTALL_DIR/frontend/dist
step "下載 CodefyUI"
if [[ -d "$INSTALL_DIR/.git" ]]; then
  warn "目錄已存在，執行更新..."
  git -C "$INSTALL_DIR" fetch --tags --depth 1 origin
  if [[ -n "$PINNED_TAG" ]]; then
    git -C "$INSTALL_DIR" fetch --depth 1 origin "refs/tags/${PINNED_TAG}:refs/tags/${PINNED_TAG}" 2>/dev/null || true
    git -C "$INSTALL_DIR" checkout -f "$PINNED_TAG"
    ok "已更新並鎖定至 $PINNED_TAG"
  else
    git -C "$INSTALL_DIR" pull --ff-only
    ok "已更新至最新版本"
  fi
else
  if [[ -n "$PINNED_TAG" ]]; then
    git clone --depth 1 --branch "$PINNED_TAG" "$REPO" "$INSTALL_DIR"
    ok "Clone 完成（${PINNED_TAG}）"
  else
    git clone --depth 1 "$REPO" "$INSTALL_DIR"
    ok "Clone 完成"
  fi
fi

# ── Frontend dist：先試 release，失敗才裝 Node 本地 build ────────────────
USE_PREBUILT=0
if [[ "$FORCE_BUILD" != "1" ]]; then
  step "Frontend dist (從 release 下載)"
  if fetch_release_dist; then
    USE_PREBUILT=1
  fi
fi

if [[ "$USE_PREBUILT" -eq 0 ]]; then
  warn "改用本地 build 路徑（會額外安裝 Node.js 與 pnpm）"
  install_node_toolchain
fi

# ── 安裝依賴 ──────────────────────────────────────────────────────────
step "安裝專案依賴"
cd "$INSTALL_DIR"
# CODEFYUI_FORCE_BUILD 透傳給 dev.py，避免 dev.py 又跳過 dist 重建
CODEFYUI_FORCE_BUILD="$FORCE_BUILD" \
CODEFYUI_RELEASE_TAG="$RELEASE_TAG" \
"$PYTHON" scripts/dev.py install --yes

# ── 安裝 cdui 到 PATH ─────────────────────────────────────────────────
step "安裝 cdui 到 PATH"
LAUNCHER_DIR="$HOME/.local/bin"
LAUNCHER="$LAUNCHER_DIR/cdui"
mkdir -p "$LAUNCHER_DIR"
cat > "$LAUNCHER" <<STUB
#!/usr/bin/env bash
# CodefyUI launcher stub — forwards to the install at $INSTALL_DIR.
exec "$INSTALL_DIR/cdui" "\$@"
STUB
chmod +x "$LAUNCHER"
ok "cdui → $LAUNCHER"

# ── 確保 ~/.local/bin 在 PATH 上 ─────────────────────────────────────
# Distros vary: Debian's ~/.profile auto-adds ~/.local/bin if it exists, but
# many minimal images / non-login shells don't source ~/.profile. We append
# an export to whichever rc file the user's shell would actually read so
# `cdui` is on PATH after they open a new terminal. Idempotent — guarded by
# a marker so re-running install doesn't duplicate the line.
ensure_path_rc() {
  local rc="$1"
  [[ -n "$rc" ]] || return 0
  # Already on PATH and rc already mentions ~/.local/bin → nothing to do.
  if grep -Fq "# >>> CodefyUI PATH (cdui) >>>" "$rc" 2>/dev/null; then
    return 0
  fi
  mkdir -p "$(dirname "$rc")"
  cat >> "$rc" <<'PATCH'

# >>> CodefyUI PATH (cdui) >>>
# Added by install.sh so the `cdui` launcher in ~/.local/bin is on PATH.
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac
# <<< CodefyUI PATH (cdui) <<<
PATCH
  ok "已將 ~/.local/bin 加入 $rc"
}

if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
  step "把 ~/.local/bin 加入 PATH"
  shell_name="$(basename "${SHELL:-bash}")"
  case "$shell_name" in
    bash)
      # Prefer ~/.bashrc for interactive non-login (most terminals); fall back
      # to ~/.profile so login shells also pick it up.
      ensure_path_rc "$HOME/.bashrc"
      [[ -f "$HOME/.profile" || ! -f "$HOME/.bash_profile" ]] && ensure_path_rc "$HOME/.profile"
      ;;
    zsh)
      ensure_path_rc "${ZDOTDIR:-$HOME}/.zshrc"
      ;;
    fish)
      mkdir -p "$HOME/.config/fish"
      ensure_path_rc "$HOME/.config/fish/config.fish"
      ;;
    *)
      warn "未知的 shell：$shell_name；請手動把 ~/.local/bin 加入 PATH"
      ;;
  esac
fi

# ── 完成 ──────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║          安裝完成！                  ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}重新開啟 terminal${NC} 讓 PATH 生效，然後："
if [[ "$USE_PREBUILT" -eq 1 ]]; then
  echo -e "    ${BOLD}cdui start${NC}      # production 模式（單一 :8000，不需 Node）"
fi
echo -e "    ${BOLD}cdui dev${NC}        # 開發模式（HMR；需 Node）"
echo ""
echo -e "  其他指令：${BOLD}cdui update | build | stop | test | clean | uninstall${NC}"
echo ""
echo -e "  若 ${BOLD}cdui${NC} 仍找不到（例如非互動式 shell），可暫時使用："
echo -e "    ${BOLD}~/.local/bin/cdui start${NC}  或  ${BOLD}$INSTALL_DIR/cdui start${NC}"
echo ""
