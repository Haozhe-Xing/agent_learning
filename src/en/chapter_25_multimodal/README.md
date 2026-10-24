# Chapter 23: Project Practice: Multimodal Agent

**Reading level**: Capstone; choose at least one domain from chapters 21–23.

## This chapter's iteration

- **Previous limitation**: Citing text does not establish reliable details from screenshots, speech or video, nor authorize actions in an interface.
- **This chapter adds**: Retain image regions, audio intervals and video-frame locations; define separate acceptance for content understanding and external actions.
- **How to verify**: This is a component-integration and domain experiment requiring real media/model adapters. Check objects, quantities and locations on fixed media. Blurred, silent or conflicting evidence must yield uncertainty; dangerous GUI actions require host permission before execution.
- **Next step**: Finish after one modality task passes. For automatic improvement, bring the frozen baseline and failure traces to [Part VII](../part7.md); completing every modality is unnecessary.

> 🎨 *"The real world speaks more than words — images, audio, and video are all languages an Agent needs to understand."*

---

## Chapter Overview

In previous chapters, all the Agents we built only processed text. But in the real world, information exists in many forms — users might send a screenshot asking "how do I fix this error," or say "help me analyze this chart." Multimodal Agents can understand and generate multiple types of content, greatly expanding application scenarios. This chapter provides component designs and integration exercises. Choose one modality and supply its actual adapter, evidence and acceptance records before expanding scope.

## Chapter Goals

After completing this chapter, you will be able to:

- ✅ Understand the core capabilities and application scenarios of multimodal models
- ✅ Implement image understanding and analysis with GPT-4.1
- ✅ Implement image generation with DALL-E
- ✅ Integrate speech recognition (STT) and text-to-speech (TTS)
- ✅ Define acceptance for one multimodal task and assemble its required components
- ✅ Understand Computer Use / GUI Agent architecture and safety practices
- ✅ Implement video understanding and multimodal RAG

## Chapter Structure

| Section |
|---|
| [23.1 Multimodal Capabilities Overview](./01_multimodal_overview.md) |
| [23.2 Image Understanding and Generation](./02_image_understanding.md) |
| [23.3 Voice Interaction Integration](./03_voice_interaction.md) |
| [23.4 Practice: Multimodal Personal Assistant](./04_practice_multimodal_assistant.md) |
| [23.5 Computer Use and GUI Agents](./05_computer_use_agent.md) |
| [23.6 Video Understanding and Multimodal RAG](./06_video_and_multimodal_rag.md) |

## 🔗 Learning Path

Bring task, permission and acceptance records from [tools](../chapter_tools/README.md) and the [Harness](../chapter_harness/README.md). Production and domain delivery require [evaluation](../chapter_20_evaluation/README.md) and [security](../chapter_21_security/README.md) gates.

Finish after one modality task passes. For automatic improvement, bring the frozen baseline and failure traces to [Part VII](../part7.md); completing every modality is unnecessary.

---

*Next: [23.1 Multimodal Capabilities Overview](./01_multimodal_overview.md)*
