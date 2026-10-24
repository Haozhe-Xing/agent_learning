# 书稿验证工具

书稿检查、构建工具和配套示例均需要 **Python 3.11+**；建议使用与CI一致的Python 3.12。构建工具用标准库`tomllib`读取原配置，系统Python 3.9不能运行这部分工具。书稿使用固定的工具组合：**mdBook 0.5.4 + mdbook-katex 0.10.0**，本地与两个 GitHub Actions 工作流保持一致。旧的 mdBook 0.4 与新的 KaTeX 0.10 使用不同的数据协议，不能混搭。

先在仓库根目录创建并激活环境，再执行检查。以下激活命令适用于macOS/Linux的bash或zsh；`python3.12`需已安装：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

确认版本为Python 3.12后，继续使用下列命令；这些书稿工具和`tools/tests`不需要额外Python包：

```bash
cargo install mdbook --version 0.5.4 --locked
cargo install mdbook-katex --version 0.10.0 --locked
python tools/check_book.py
python tools/check_learning_path.py
python -m unittest discover -s tools/tests -v
python tools/tests/test_rendered_links.py --require-tools
python tools/build_books.py
```

`build_books.py` 在临时目录中依次使用两份原始配置，保留 KaTeX 预处理，不交换或覆盖仓库中的 `book.toml`。输出默认为 `book/zh`、`book/en`；可用 `--output DIR` 改为临时预览目录。构建结束后还会检查计划发布的源SVG是否全部进入成品、明确排除目录是否泄漏，以及成品中全部SVG的XML语法。mdBook额外生成的favicon也计入成品检查，因此成品与源图计数可能不同。

两语配置使用`readme-links`预处理器：`tools/readme_links.py`在`index`之后、KaTeX之前，按已声明章节的`source_path → path`改写正文章首页目标，保留源码路径与文件。它处理本地实际章节，保护代码示例和外链；不是全文URL替换器。mdBook 0.5.4的最小实构建曾复现正文`README.html`目标与实际`index.html`不一致，不能仅靠源文件存在来验收。顺序配置依据见[官方预处理说明](https://rust-lang.github.io/mdBook/format/configuration/preprocessors.html)。

构建默认再用`check_rendered_book.py`检查两语SUMMARY实际页面的`<main>`本地链接、图片目标和HTML锚点。单独重查可运行`python tools/check_rendered_book.py --output book`。它不检查外链、页面其他区域、视觉或内容正确性。`test_rendered_links.py --require-tools`会强制要求精确构建工具，实际生成双语小书验证路由与KaTeX；CI在安装后明确运行，避免前置结构测试缺工具时跳过成为唯一证据。

`./serve.sh`也调用同一个暂存构建和成品门禁，要求Python3.11+；不再临时交换中文版与英文版配置。可选文件监听依赖与进程生命周期仍需在相应系统环境另行验证。

金额正文使用 `USD`，避免与 `$...$` 数学定界符冲突。代码围栏中的 shell 提示符仍保留 `$`。

`check_book.py` 检查章节链接、静态图片、配套路径和中英文目录顺序，并按`book.toml`与`book-en.toml`的源目录递归解析全部计划发布SVG的XML，包括嵌套目录。代码围栏中的假设文件树和动态图片占位符不作为静态路径检查。

检查和构建共用`build_books.py`中的`EXCLUDED_SOURCE_DIRECTORIES`。目前只明确排除未被发布内容引用的`src/en/svg_backup_broken`，不使用`*backup*`等通配规则。源文件保留原样，只在构建暂存副本中省略；若发布源的已支持Markdown、HTML、CSS或SVG引用指向被排除资源，检查与构建都会拒绝。回归用例见`tools/tests/test_publication_assets.py`。

资源检查证明包装范围与XML语法，不证明图中知识、视觉排版、印刷可读性、版权或HTML交互行为正确；这些仍需独立审校。

`book_structure.py` 定义25章、7部分的阅读顺序，原有目录名作为稳定URL保留。`check_learning_path.py` 依据它检查两语目录、章节归属、导航与页面标题编号、章节结构表、带编号链接，以及每章的不足/增量/验证/下一步四项交接。机器检查不替代教学语义审读，也不推测未链接正文或图像中的任意数字。`tools/tests/` 用错误顺序、错号、缺字段和坏链接验证检查器能拒绝回归。这些检查已接入构建检查与部署工作流。

独立参考项目的运行、难度与阅读顺序见 `reference_code/evidence_assistant/README.md`；可在项目目录执行 `python run.py` 和 `python run_tests.py`。章节示例运行与测试说明见仓库根目录的 `examples/README.md`。所有验证结果及其边界记录在 `examples/VALIDATION.md`。

本版以中英文在线书稿为主要交付。编辑与验收条件见[出版验收约定](PUBLICATION_REVIEW.md)，逐章进展和未完成项见[审校库存](PUBLICATION_INVENTORY.md)，产品案例见[产品事实台账](PRODUCT_FACTS.md)，已核论文的固定文本、修正与核读范围见[论文来源台账](PAPER_FACTS.md)。这些记录与机器检查相互补充，不能用构建成功代替全文审读。
