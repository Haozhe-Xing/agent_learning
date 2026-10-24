#!/usr/bin/env bash
# =============================================================================
# serve.sh — 本地开发一键启动脚本
# 用法：
#   ./serve.sh          # 使用默认端口 3000
#   ./serve.sh 8080     # 使用自定义端口 8080
#   ./serve.sh --watch  # 启用文件监听自动重建（需要 fswatch 或 inotifywait）
#   ./serve.sh 8080 --watch
# =============================================================================

set -euo pipefail

# ---------- 颜色输出 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()    { echo -e "\n${BOLD}${CYAN}▶ $*${NC}"; }

# ---------- 参数解析 ----------
PORT=3000
WATCH=false
USER_REQUESTED_WATCH=false

for arg in "$@"; do
  case "$arg" in
    --watch) WATCH=true; USER_REQUESTED_WATCH=true ;;
    [0-9]*) PORT="$arg" ;;
    *)
      error "未知参数：$arg"
      echo "用法：$0 [端口] [--watch]"
      exit 1
      ;;
  esac
done

# ---------- 依赖检查 ----------
step "检查依赖"

check_mdbook() {
  if ! command -v mdbook &>/dev/null; then
    error "未找到 mdbook，请先安装："
    echo ""
    echo "  # macOS (Homebrew)"
    echo "  brew install mdbook"
    echo ""
    echo "  # 或通过 cargo 安装"
    echo "  cargo install mdbook --version 0.5.4 --locked"
    echo ""
    echo "  # 或直接下载二进制（macOS arm64）"
    echo "  curl -sSL https://github.com/rust-lang/mdBook/releases/download/v0.5.4/mdbook-v0.5.4-aarch64-apple-darwin.tar.gz | tar -xz"
    echo "  mv mdbook /usr/local/bin/"
    exit 1
  fi
  if [[ "$(mdbook --version)" != "mdbook v0.5.4" ]]; then
    error "需要 mdBook 0.5.4，与 mdbook-katex 0.10.0 配套；安装方法见 tools/README.md"
    exit 1
  fi
  success "$(mdbook --version)"
}

check_mdbook_katex() {
  if ! command -v mdbook-katex &>/dev/null; then
    error "未找到正式构建要求的 mdbook-katex 0.10.0"
    warn "安装方式："
    warn "  cargo install mdbook-katex --version 0.10.0 --locked"
    warn "  # 或下载二进制：https://github.com/lzanini/mdbook-katex/releases"
    exit 1
  else
    if [[ "$(mdbook-katex --version)" != "mdbook-katex 0.10.0" ]]; then
      error "需要 mdbook-katex 0.10.0；安装方法见 tools/README.md"
      exit 1
    fi
    success "mdbook-katex 0.10.0 已安装"
  fi
}

check_python() {
  if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
  elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
  else
    error "未找到 Python；请安装 Python 3.11+，建议 Python 3.12，见 tools/README.md"
    exit 1
  fi
  if ! "$PYTHON_CMD" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
    error "书稿检查与构建需要 Python 3.11+；请先激活 Python 3.12 虚拟环境，见 tools/README.md"
    exit 1
  fi
  success "$("$PYTHON_CMD" --version)"
}

check_mdbook
check_mdbook_katex
check_python

# ---------- 获取脚本所在目录（项目根目录）----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

step "检查目录顺序与章节交接"
"$PYTHON_CMD" tools/check_learning_path.py

# ---------- 构建函数 ----------
# 与CI共用暂存构建、源资源和实际成品门禁；不替换工作区配置。
build_all() {
  step "构建中英文版并检查成品"
  "$PYTHON_CMD" tools/build_books.py
  cp root-index.html book/index.html
  success "双语成品已生成 → book/"
}

build_language() {
  local language="$1"
  info "重新构建 $language 并检查成品..."
  "$PYTHON_CMD" tools/check_learning_path.py
  "$PYTHON_CMD" tools/build_books.py --language "$language"
  success "$language 成品重建完成"
}

build_zh() { build_language zh; }
build_en() { build_language en; }

# ---------- 首次全量构建 ----------
build_all

# ---------- 启动静态文件服务器 ----------
step "启动静态文件服务器"
info "访问地址：${BOLD}http://localhost:${PORT}${NC}"
info "中文版：  ${BOLD}http://localhost:${PORT}/zh/${NC}"
info "英文版：  ${BOLD}http://localhost:${PORT}/en/${NC}"
info "按 Ctrl+C 停止服务"

# ---------- 文件监听（可选）----------
if [ "$WATCH" = true ]; then
  step "启动文件监听（自动重建）"

  OS="$(uname -s)"

  if [ "$OS" = "Darwin" ]; then
    # macOS：使用 fswatch
    if ! command -v fswatch &>/dev/null; then
      warn "未找到 fswatch，无法启用文件监听"
      warn "安装方式：brew install fswatch"
      warn "降级为仅静态服务模式"
      WATCH=false
    fi
  else
    # Linux：使用 inotifywait
    if ! command -v inotifywait &>/dev/null; then
      warn "未找到 inotifywait，无法启用文件监听"
      warn "安装方式：sudo apt-get install inotify-tools"
      warn "降级为仅静态服务模式"
      WATCH=false
    fi
  fi
fi

if [ "$WATCH" = true ]; then
  info "监听目录：src/zh/ 和 src/en/"

  OS="$(uname -s)"

  if [ "$OS" = "Darwin" ]; then
    # macOS：fswatch 后台监听
    (
      fswatch -r --event Created --event Updated --event Removed \
        --exclude '\.git' \
        src/zh src/en | while read -r changed_file; do
        if echo "$changed_file" | grep -q "/src/zh/"; then
          build_zh
        elif echo "$changed_file" | grep -q "/src/en/"; then
          build_en
        fi
      done
    ) &
    WATCHER_PID=$!
  else
    # Linux：inotifywait 后台监听（--monitor 持续监听模式）
    (
      inotifywait -r -m -e modify,create,delete,move \
          --exclude '\.git' \
          --format '%w%f' \
          src/zh src/en 2>/dev/null | while read -r changed_file; do
        if echo "$changed_file" | grep -q "src/zh"; then
          build_zh
        elif echo "$changed_file" | grep -q "src/en"; then
          build_en
        fi
      done
    ) &
    WATCHER_PID=$!
  fi

  # 退出时清理后台进程
  trap "kill $WATCHER_PID 2>/dev/null; exit 0" INT TERM

  success "文件监听已启动（PID: $WATCHER_PID）"
else
  if [ "$USER_REQUESTED_WATCH" = false ]; then
    # 用户未传 --watch，提示可用
    echo ""
    info "提示：使用 ${BOLD}./serve.sh --watch${NC} 可启用文件变更自动重建"
  fi
fi

# ---------- 启动 HTTP 服务器（前台，阻塞）----------
echo ""
success "服务已启动 → ${BOLD}http://localhost:${PORT}${NC}"
echo ""

cd book
exec "$PYTHON_CMD" -m http.server "$PORT" --bind 127.0.0.1
