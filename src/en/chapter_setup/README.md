# Chapter 2: Development Environment Setup

> A well-sharpened tool makes the work easier. A well-configured development environment will make your Agent development twice as efficient.

---

## Chapter Overview

This chapter walks you through setting up a professional Agent development environment from scratch. From Python virtual environment management, to secure API Key storage, to your first runnable Agent — follow along and you'll have a workbench ready to start developing at any time.

## Chapter Goals

After completing this chapter, you will be able to:

- ✅ Manage Python environments and dependencies using `uv` or `conda`
- ✅ Install and verify core libraries like LangChain and the OpenAI SDK
- ✅ Securely manage API Keys to prevent key leakage
- ✅ Run your first real Agent!

## Chapter Structure

| Section | Content | Estimated Time |
|---------|---------|---------------|
| 2.1 Python Environment & Dependency Management | venv, conda, uv comparison and usage | 20 min |
| 2.2 Key Library Installation | LangChain, OpenAI SDK, etc. | 15 min |
| 2.3 API Key Management | .env files, environment variable best practices | 10 min |
| 2.4 Hello Agent! | Your first complete Agent | 30 min |

## Prerequisites

- Python 3.10 or higher installed
- Basic command-line experience
- An OpenAI account registered (or another LLM service)

## Tool Selection Guide

| Tool | Purpose | Recommendation |
|------|---------|---------------|
| `uv` | Python package management (newest, fastest) | ⭐⭐⭐⭐⭐ Highly recommended |
| `conda` | Environment management (common in data science) | ⭐⭐⭐⭐ Recommended |
| `venv` | Built-in virtual environment | ⭐⭐⭐ Sufficient |
| VS Code | Code editor | ⭐⭐⭐⭐⭐ Recommended |
| Jupyter | Interactive development and experimentation | ⭐⭐⭐⭐ Recommended |

## 🔗 Learning Path

> **Prerequisites**: [Chapter 1: What is an Agent?](../chapter_intro/README.md)
>
> **Recommended next steps**:
> - 👉 [Chapter 2: LLM Fundamentals](../chapter_llm/README.md) — Understand the Agent's core "brain"
> - 👉 [Chapter 3: Tool Calling](../chapter_tools/README.md) — Let Agents "get things done"

---

*Next section: [2.1 Python Environment & Dependency Management](./01_python_setup.md)*
