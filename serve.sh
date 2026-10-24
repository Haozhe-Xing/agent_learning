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
    echo "  cargo install mdbook"
    echo ""
    echo "  # 或直接下载二进制（macOS arm64）"
    echo "  curl -sSL https://github.com/rust-lang/mdBook/releases/download/v0.4.40/mdbook-v0.4.40-aarch64-apple-darwin.tar.gz | tar -xz"
    echo "  mv mdbook /usr/local/bin/"
    exit 1
  fi
  success "mdbook $(mdbook --version)"
}

check_mdbook_katex() {
  if ! command -v mdbook-katex &>/dev/null; then
    warn "未找到 mdbook-katex，数学公式渲染可能不可用"
    warn "安装方式："
    warn "  cargo install mdbook-katex"
    warn "  # 或下载二进制：https://github.com/lzanini/mdbook-katex/releases"
    # 不退出，katex 在 book.toml 中标记为 optional = true
  else
    success "mdbook-katex 已安装"
  fi
}

check_python() {
  if command -v python3 &>/dev/null; then
    success "python3 $(python3 --version 2>&1 | awk '{print $2}')"
    PYTHON_CMD="python3"
  elif command -v python &>/dev/null && python --version 2>&1 | grep -q "Python 3"; then
    success "python $(python --version 2>&1 | awk '{print $2}')"
    PYTHON_CMD="python"
  else
    error "未找到 python3，请安装 Python 3："
    echo ""
    echo "  # macOS"
    echo "  brew install python3"
    echo ""
    echo "  # Ubuntu/Debian"
    echo "  sudo apt-get install python3"
    exit 1
  fi
}

check_mdbook
check_mdbook_katex
check_python

# ---------- 获取脚本所在目录（项目根目录）----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- 构建函数 ----------
# mdbook v0.4.40+ 移除了 --config 参数，通过临时替换 book.toml 来切换配置
_build_en_impl() {
  # 备份中文版配置，临时替换为英文版配置，构建完成后恢复
  cp book.toml book.toml.zh_bak
  cp book-en.toml book.toml
  # 确保无论成功还是失败都能恢复
  trap 'cp book.toml.zh_bak book.toml; rm -f book.toml.zh_bak' EXIT
  mdbook build "$@"
  cp book.toml.zh_bak book.toml
  rm -f book.toml.zh_bak
  trap - EXIT
}

build_all() {
  step "构建中文版"
  mdbook build
  success "中文版构建完成 → book/zh/"

  step "构建英文版"
  _build_en_impl
  success "英文版构建完成 → book/en/"

  step "复制根页面"
  cp root-index.html book/index.html
  success "根页面已复制 → book/index.html"
}

build_zh() {
  info "重新构建中文版..."
  mdbook build 2>&1 | tail -3
  success "中文版重建完成"
}

build_en() {
  info "重新构建英文版..."
  _build_en_impl 2>&1 | tail -3
  success "英文版重建完成"
}

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
