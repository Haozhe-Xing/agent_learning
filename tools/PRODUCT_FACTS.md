# 产品案例事实核验台账

核验日期：2026-10-22。当前范围仅为第12章 OpenClaw、第13章 Hermes Agent、第15章 DeepSeek Harness 的来源身份、版本锚、许可证与若干明确史实。与[出版库存](PUBLICATION_INVENTORY.md)的 PUB-08/09 对应；不是三章全文审结证明。

## 证据规则

- **已核验**：已读取下列一手页面或固定源码，结论限于所列字段。
- **项目自述**：README、文档或宣传页声称具备的功能，不等于本书已运行或独立验证。
- **教学分析**：作者用于解释边界的分层、伪代码与对照，不冒充官方目录、API、默认策略或性能结论。
- **未核验**：未获得足够一手材料，或尚未运行对应版本。不得用 Star、安装量、第三方转述或版本号推断效果。

以下 tag 是本次材料定位点，不保证永不移动；复查优先使用完整 commit。官方文档网站仍会更新，因此另记其核验日期，不将网站内容自动归到固定源码版本。

## 固定材料与产品身份

| 产品 | 官方身份与文档入口 | 固定版本与源码 | 许可证 | 当前证明范围 |
|---|---|---|---|---|
| OpenClaw | [官方仓库](https://github.com/openclaw/openclaw)、[文档站](https://docs.openclaw.ai/) | [v2026.10.5 release](https://github.com/openclaw/openclaw/releases/tag/v2026.10.5) → [1b58cb9a1b4f4353e1add63c8506db4d890962e8](https://github.com/openclaw/openclaw/commit/1b58cb9a1b4f4353e1add63c8506db4d890962e8) | [该提交 LICENSE](https://github.com/openclaw/openclaw/blob/1b58cb9a1b4f4353e1add63c8506db4d890962e8/LICENSE)：MIT | 仓库、官方文档入口、tag/commit、许可身份；未安装运行 |
| Hermes Agent | [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)、[官方文档](https://hermes-agent.nousresearch.com/docs/) | [v0.22.4 release](https://github.com/NousResearch/hermes-agent/releases/tag/v0.22.4) → [4d04303818b61bf3aa09278af48356315515965b](https://github.com/NousResearch/hermes-agent/commit/4d04303818b61bf3aa09278af48356315515965b) | [该提交 LICENSE](https://github.com/NousResearch/hermes-agent/blob/4d04303818b61bf3aa09278af48356315515965b/LICENSE)：MIT | 项目归属、tag/commit、许可、README 所列入口；未执行迁移 |
| DeepSeek Harness | [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)、[产品页](https://deepseek.com/harness/)、[开发者文档](https://deepseek-harness.github.io/deepseek-harness/guide/quickstart) | [dsh-v0.2.1-rc.2 release](https://github.com/deepseek-ai/deepseek-harness/releases/tag/dsh-v0.2.1-rc.2) → [687f282e8625c6b8d11ff58211a90524be2e4317](https://github.com/deepseek-ai/deepseek-harness/commit/687f282e8625c6b8d11ff58211a90524be2e4317) | [该提交 LICENSE](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/LICENSE)：MIT | 项目归属、开发者预览、tag/commit、许可、README 指向的 Cordis；未安装运行 |

许可证身份不证明插件、模型、数据、第三方依赖的全部许可相同，也不证明生产适用性。本轮不作法律合规结论。

## 已核对并回写正文的声明

| ID | 原书稿问题 | 一手依据 | 处理与边界 |
|---|---|---|---|
| OC-01 | 12.1 将原型起点写为2025-12、OpenClaw 名称写为2026-01-30之后 | Peter Steinberger 的[官方介绍，2026-01-29](https://openclaw.ai/blog/introducing-openclaw) | 作者回顾 Clawd 生于2025-11，并在文章当天宣布 OpenClaw；两语改按该回顾描述。没有补造 Moltbot 的精确日期或未证实原型仓库名 |
| OC-02 | 热度、多语言改写被直接用来证明核心抽象正确 | 上述公告只提供作者经历；不能证明架构优越性 | 删除直接质量推论，第三方派生关系和语言清单仍需另核。未进行跨产品质量排名 |
| HE-01 | 将 Hermes 称为 OpenClaw 的同源继承者 | 固定提交的 [README](https://github.com/NousResearch/hermes-agent/blob/4d04303818b61bf3aa09278af48356315515965b/README.md)记录迁移入口 | 迁移功能不能证明谱系。两语13.1移除同源、继承、改名关系推论，保留迁移为项目自述 |
| HE-02 | 13.1 写 Python 核心 + TypeScript Gateway，与13.3自身冲突 | 固定源码 [gateway/run.py](https://github.com/NousResearch/hermes-agent/blob/4d04303818b61bf3aa09278af48356315515965b/gateway/run.py) 是 Python Gateway runner | 改为可直接核对的 Python 文件事实，不推断整个仓库或所有界面的语言 |
| HE-03 | 首发2025-07、v1.x运行近一年、“当前最深”等精确或排他声明 | 本轮读取的 [v0.22.4 release](https://github.com/NousResearch/hermes-agent/releases/tag/v0.22.4)仅证明所选版本，不能证明首发史或比较排名 | 删除这些无充分来源的肯定语；首次公开发布时间仍未核验，不能由 LICENSE 的版权年推导 |
| HE-04 | 将 DSPy/GEPA、独立离线进程、固定30分钟周期、`self_evolving.on_tool_use` 写成官方实现 | [官方架构文档](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)可用于继续定位源码；本轮未逐项找到以上实现证据 | 13.1/13.3相关断言改为待验证设计或教学钩子。不能以 README 的学习目标证明某个优化算法、进程模型或默认周期 |
| HE-05 | “每个 Skill 都由自进化产生”、作者身份决定默认信任 | 固定 README 同时记录技能与迁移入口；并不支持“每个”量词 | 删除排他来源与默认信任推论，要求分别检查导入、生成、审查与启用边界 |
| DS-01 | Cordis 上游写为 `koishijs/koishi` 同名子项目 | 固定提交的 [README](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/README.md)及[官方产品页](https://deepseek.com/harness/)均指向 [cordiverse/cordis](https://github.com/cordiverse/cordis) | 两语15.1更正上游链接，不据项目渊源推断当前代码目录 |
| DS-02 | 把发布口号、全部能力可换和完整功能表写成已验证能力 | 固定 README 标为 developer preview；所选 release 标为 Pre-release | 将官方自述、分析清单和待核接口分开，删除无来源引语及数量背书 |
| DS-03 | 首发2026-08-13及个人负责人等精确来源声明 | 本轮一手材料未建立首次公开发布事件；仓库创建与产品发布不能混同 | 15.1不保留未核验日期和个人职责。此处不是判定原日期为假，而是撤回未经证实的精确表述 |

## 编辑与引用修复

PUB-08 的目标按语义逐处确认：OpenClaw 架构12.3、渠道12.4、技能12.5；Hermes 技能13.4、记忆13.5、主动任务13.6；Cordis 架构15.3、插件15.4、模式15.2。修正正文、表格和代码注释中旧14.x/15.x/17.x，并在可点击正文处补本地链接。不能指向原来承诺的内容时，删除错误承诺并说明真实验证责任；没有全局机械减号。

PUB-09 的两处“working memory 中约定”已从两语相关来源段落清除，改成面向读者的证据范围；同时移除虚构的作者审读过程和未指向具体记录的发布节奏断言。

## 明确未完成的出版项

| 范围 | 下一步所需证据 | 当前状态 |
|---|---|---|
| [12.3架构中文](../src/zh/chapter_openclaw/03_architecture.md)与[英文](../src/en/chapter_openclaw/03_architecture.md) | 两语已按同一七节架构案例重写，核对固定Telegram manifest、会话、循环、沙箱与skills文档；责任图与教学伪代码不冒充产品API | 原不安全代码/假成功输出/默认策略断言已移除，补英文图；参考Harness练习已跑。未安装产品、执行渠道或验证沙箱/并发恢复 |
| 12.2/12.4–12.6 安装、身份合并、渠道配置、Skills/插件命令 | 固定版本的配置 schema、目录与实际拒绝/隔离用例 | 未执行产品安装、渠道登录、消息发送；具体配置与伪输出未全面核验 |
| 12.1/12.7 第三方生态 | 各独立项目自己的固定仓库、语言及派生关系 | 未核验，不能作为谱系或质量证明 |
| 13.2–13.6 CLI、存储、身份、技能草稿与迭代 | 固定代码路径、命令帮助、数据格式、授权与回退反例 | 部分明显断言已降为教学设计；并未证明其余命令和伪代码真实可运行 |
| 13.4中文与英文 | 两语已按同一八节重写，五阶段明确为本书候选技能设计；相同JSON Schema、草稿隔离、独立回归与晋升/回退条件 | 旧Python错误/越界路径/假draft门禁/伪优化日志与官方CLI已移除；候选与14个拒绝变体验证，两语图一致。没有实现Hermes加载器或真实优化服务 |
| 15.2–15.4 四种模式、插件键、Python SDK、轨迹回放与沙箱 | 固定源码/文档、组件生命周期、错误和卸载测试 | 未逐项核验；15.1的能力表不是官方 API 清单 |
| 三产品的效果、兼容和安全 | 同一任务、权限、预算与验收，真实执行记录 | 未做统一适配或同题比较；没有生产或兼容背书 |

后续每次新增事实，应记录“原声明、固定来源、核验日、结论范围、正文回写位置、未验证项”。发现上游版本变化时新增记录，不把旧结论自动延展到新版本。
