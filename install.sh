#!/usr/bin/env bash
# CodefyUI 一鍵安裝腳本
# 用法：curl -fsSL https://raw.githubusercontent.com/latteine1217/CodefyUI/main/install.sh | bash
set -euo pipefail

# ── 顏色 ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

REPO="https://github.com/latteine1217/CodefyUI.git"
INSTALL_DIR="${CODEFYUI_DIR:-$HOME/CodefyUI}"

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

# ══════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}╔══════════════════════════════════════╗${NC}"
echo -e "${BOLD}║        CodefyUI  安裝程式            ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════╝${NC}"
echo -e "  安裝目錄：${BOLD}$INSTALL_DIR${NC}"

# ── git ───────────────────────────────────────────────────────────────
step "git"
if ! command -v git &>/dev/null; then
  warn "未安裝，正在安裝..."
  pkg_install git
fi
ok "$(git --version)"

# ── Python 3 ──────────────────────────────────────────────────────────
step "Python 3"
PYTHON=""
for cmd in python3 python; do
  if command -v "$cmd" &>/dev/null && "$cmd" -c "import sys; assert sys.version_info >= (3,8)" 2>/dev/null; then
    PYTHON="$cmd"; break
  fi
done

if [[ -z "$PYTHON" ]]; then
  warn "未安裝，正在安裝..."
  case "$OS" in
    macos)   pkg_install python@3.11 ;;
    debian)  pkg_install python3 ;;
    redhat)  pkg_install python3 ;;
    *)       die "請手動安裝 Python 3.8+" ;;
  esac
  PYTHON="python3"
fi
ok "$($PYTHON --version)"

# ── Node.js ───────────────────────────────────────────────────────────
step "Node.js"
if ! command -v node &>/dev/null; then
  warn "未安裝，透過 nvm 安裝 LTS..."
  export NVM_DIR="$HOME/.nvm"
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
  # nvm 內部使用未定義變數，需暫時關閉 -u
  set +u
  [[ -s "$NVM_DIR/nvm.sh" ]] && source "$NVM_DIR/nvm.sh"
  nvm install --lts --no-progress
  nvm use --lts
  set -u
fi
ok "Node.js $(node --version)"

# ── pnpm ──────────────────────────────────────────────────────────────
step "pnpm"
if ! command -v pnpm &>/dev/null; then
  warn "未安裝，正在安裝..."
  # 優先用 npm（已隨 Node.js 安裝）
  if command -v npm &>/dev/null; then
    npm install -g pnpm --silent
  else
    curl -fsSL https://get.pnpm.io/install.sh | sh -
    export PNPM_HOME="$HOME/.local/share/pnpm"
    export PATH="$PNPM_HOME:$PATH"
  fi
fi
ok "pnpm $(pnpm --version)"

# ── Clone / Update ────────────────────────────────────────────────────
step "下載 CodefyUI"
if [[ -d "$INSTALL_DIR/.git" ]]; then
  warn "目錄已存在，執行更新..."
  git -C "$INSTALL_DIR" pull --ff-only
  ok "已更新至最新版本"
else
  git clone --depth 1 "$REPO" "$INSTALL_DIR"
  ok "Clone 完成"
fi

# ── 安裝依賴 ──────────────────────────────────────────────────────────
step "安裝專案依賴"
cd "$INSTALL_DIR"
$PYTHON dev.py install

# ── 完成 ──────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║          安裝完成！                  ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════╝${NC}"
echo ""
echo -e "  啟動開發伺服器："
echo -e "    ${BOLD}cd $INSTALL_DIR${NC}"
echo -e "    ${BOLD}python dev.py dev${NC}"
echo ""
