# Chapter 12: OpenClaw — Cross-Platform Personal AI Assistant

**Reading level**: Optional product and architecture case; skip without a chat-entry requirement.

Use OpenClaw as a case for chat entry points and personal assistants. Examine identity, sessions, tools, and permissions after a message arrives. The aim is to connect an entry point to existing capabilities and inspect the boundaries through task results.

## This chapter's iteration

- **Previous limitation**: The assistant receives scripts or API requests but has no defined mapping from chat identities and sessions to tool permissions.
- **This chapter adds**: A product case for message gateways, routing, sessions, and extensions; moving the entire assistant into the product is optional.
- **How to verify**: No uniform product adapter is supplied. Pin a version and one channel, then use controlled tasks to inspect identity, sources, permissions, failures, and outputs. Identify executed and unexecuted parts explicitly.
- **Next step**: For cross-session accumulation, select [chapter 13, Hermes](../chapter_hermes/README.md). Otherwise return to the [Part III choice guide](../part3.md) and continue the existing implementation.

## Chapter structure

| Section | Focus |
|---|---|
| [12.1 OpenClaw Panorama: From Clawdbot to OpenClaw](./01_history_and_positioning.md) | Understand the case and its limits |
| [12.2 Installation & 4 Deployment Modes](./02_install_and_deploy.md) | Record version, environment, and installation path |
| [12.3 Architecture: Gateway / Agent Loop / Skills](./03_architecture.md) | Map messages, state, and tool execution |
| [12.4 Multi-Channel Routing: WhatsApp / Telegram / Discord / Slack / Signal](./04_channels.md) | Verify identity and permissions in one channel |
| [12.5 Skills & Plugin Ecosystem: ClawHub and Community Contributions](./05_skills_and_plugins.md) | Review extension inputs, capabilities, and trust |
| [12.6 Practice: Build a Personal Assistant](./06_practice.md) | Run one controlled entry-point experiment |
| [12.7 Lessons for Engineers](./07_lessons_for_engineers.md) | Map a verified mechanism onto the existing assistant |

## Reading path

Review execution controls in [chapter 8, Harness](../chapter_harness/README.md) and capability packaging in [chapter 9, Skills](../chapter_skill/README.md). For routing, focus on 12.3–12.4. To borrow architecture alone, select 12.7; chapters 10–11 are not required first.

## Verification and material boundaries

The chapter outlines product and architecture experiments; it does not promise all channels will connect within a fixed time. Verify installation, channel, and extension interfaces in a selected version. Without execution conditions, complete design analysis and mark the experiment “not run.”

The offline distance-and-calculation fixture does not call OpenClaw or validate its data handling and permissions. Preserve the task contract in a product experiment and add identity and session cases for the new entry point.

---

*Next section: [12.1 OpenClaw Panorama: From Clawdbot to OpenClaw](./01_history_and_positioning.md)*
