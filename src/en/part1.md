# Part 1: Getting Started

## Getting Started with Large Models: How Should Non-Programmers Begin Using AI?

> 🌱 *"The first step to using LLMs well is not understanding every technical detail, but knowing what they can do, what they can't, and how to articulate your tasks clearly."*

This chapter does not require programming skills or an understanding of neural networks. We approach LLMs purely from a user's perspective, treating them as everyday tools.

![A roadmap for absolute beginners entering the world of large models](./svg/chapter_llm_00_beginner_map.svg)

You can think of a large model as an assistant who is exceptionally good with language: it can read materials, write drafts, revise expressions, summarize, outline plans, explain concepts, generate code, and brainstorm with you.

But it is not an infallible expert, nor is it a database that inherently knows the truth. It is more like a collaborator who is highly capable and responsive, but whose output you need to verify.

The first step to using LLMs well is not learning complex theory — it is learning to articulate your tasks clearly.

### How Should You Read This Chapter?

This chapter is written for absolute beginners. You do not need to memorize model names, calculate tokens precisely, or understand how models are trained internally.

We suggest focusing on just four things:

| Intuition to Build | One-Sentence Understanding |
|---|---|
| **What a large model is** | A language-processing collaborator |
| **What it can do** | Read, write, revise, summarize, explain, and plan for you |
| **How to ask effectively** | Provide background, goal, materials, requirements, and output format |
| **When to be cautious** | Verify important facts, numbers, legal, medical, and financial advice |

If, after reading this chapter, you can start using LLMs for learning, writing, office work, or programming assistance, then this chapter has served its purpose.

## LLMs Are Not Search Engines — They Are Language Collaborators

Many people treat LLMs like search engines when they first use them:

```text
What is the tallest mountain in the world?
What was Company X's revenue this year?
Is this news story true?
```

These are certainly askable, but they are not where LLMs provide the most value.

A search engine is better at helping you **find source materials**; a calculator is better at **precise computation**; what an LLM truly excels at is **processing language and tasks**.

![LLMs are not search engines — they are language collaborators](./svg/chapter_llm_00_llm_vs_search.svg)

For example, given the same long article:

- A search engine can help you find it.
- A browser can help you open it.
- An LLM can help you summarize it, explain it, rewrite it, extract its key points, and even turn it into a presentation script.

So a more accurate understanding is:

> **A search engine helps you find materials; an LLM helps you process materials.**

If you only ask an LLM factual questions, it may be nothing more than an unreliable Q&A machine. But if you give it a piece of material, a task, or a goal, it becomes a very useful collaborator.

## Think of an LLM as "a Highly Capable Intern Who Needs Checking"

For absolute beginners, the easiest analogy for an LLM is: **an intern**.

This intern has read a lot of material, writes quickly, is great at imitation, and can help organize your thoughts. But it has a few obvious shortcomings:

- It may not know your real background.
- It may misunderstand your goals.
- It may fabricate details to produce a complete-sounding answer.
- It may treat outdated information as current.
- What it writes may look convincing but is not necessarily correct.

Therefore, the right way to use it is not "let it give the final answer in one shot," but rather to collaborate with it iteratively.

![Think of an LLM as a highly capable intern who needs checking](./svg/chapter_llm_00_ai_intern_loop.svg)

A common usage loop is:

1. You tell it the task and background.
2. It produces a first draft.
3. You check what's wrong or insufficient.
4. You add requirements.
5. It revises further.
6. You make the final call on whether to use it.

For example, you could say:

```text
You are an expert in AI popular science.
I am preparing to write an AI popular science article for high school students.
Please first outline it for me — do not write the full article yet.
Requirements: use relaxed language, no complex formulas.
Give a real-life analogy for each section.
```

This is far better than simply saying "write me an AI article," because it gives the model a clear identity, audience, task, constraints, and output target.

## What Can LLMs Do for Regular People?

The most fundamental capability of LLMs is not "thinking like a human," but rather **performing various transformations and organizations around language**.

Give it a paragraph, it can summarize. Give it a topic, it can draft. Give it a jumble of ideas, it can organize. Give it a concept, it can explain. Give it code, it can analyze.

![What LLMs can do for regular people](./svg/chapter_llm_00_capability_map.svg)

From an ordinary user's perspective, its uses can be roughly divided into several categories:

| Scenario | What the LLM Can Do | How You Might Ask |
|---|---|---|
| **Learning** | Explain concepts, create study plans, generate questions, grade answers | "Explain what inflation is in terms a middle-schooler can understand" |
| **Writing** | Draft, polish, change style, expand, condense | "Rewrite this passage to be clearer and more suitable for a WeChat public account opening" |
| **Office Work** | Write emails, organize meeting notes, extract to-do items | "Organize this meeting transcript into three columns: owner, task, and deadline" |
| **Reading** | Summarize long articles, extract key points, compare materials | "Summarize the core argument of this article and list the author's three pieces of evidence" |
| **Programming** | Explain code, generate scripts, troubleshoot errors | "What does this Python error mean? How should I fix it?" |
| **Creativity** | Name things, write scripts, draft proposals, brainstorm | "Generate 10 titles for an AI introductory course — keep the style relaxed" |
| **Decision Support** | List options, do pros/cons analysis, identify blind spots | "Should I buy a tablet or a laptop? Please create a decision table for me" |
| **Multimodal** | Read images, interpret screenshots, explain charts | "What issue does this report chart reveal?" |

Here's an important principle:

> **LLMs are best at helping you produce first drafts, not making final judgments for you.**

For example, it can help you draft a resume, but you should verify whether the experiences are accurate. It can help you summarize a contract, but important clauses still need professional review. It can help you interpret a medical report, but it cannot replace a doctor's diagnosis.

## How to Ask LLMs Effective Questions

Many people find models unhelpful — not necessarily because the model is bad, but because the question is too vague.

For instance, questions like this usually produce poor results:

```text
Write me a proposal.
```

The model does not know who you are, who the proposal is for, what the goal is, how long it should be, or what style you prefer. It can only guess.

For beginners, the simplest prompt formula is:

> **Background + Goal + Materials + Requirements + Output Format**

![The most practical prompt formula for beginners](./svg/chapter_llm_00_prompt_formula.svg)

Let's rewrite that vague question:

```text
I am a university lecturer preparing a 90-minute AI introductory class for students with no prior background.
The goal is for them to understand what LLMs can do and complete one effective prompt.
Please help me design a course outline.
Requirements:
1. No formulas;
2. One interactive activity every 20 minutes;
3. Include an in-class exercise at the end;
4. Output in table format.
```

This question is much clearer now.

It tells the model:

- **Background**: University lecturer, students with no background.
- **Goal**: Understand LLM capabilities and complete one effective prompt.
- **Task**: Design a course outline.
- **Requirements**: No formulas, interactive, with an exercise.
- **Format**: Output as a table.

You can also memorize it as a template:

```text
You are playing the role of [role].
I want to accomplish [task].
The background is [background information].
The materials I have are [materials].
Please complete this according to [requirements].
Output format: [format].
If you lack sufficient information, please ask me questions first.
```

The last line — "If you lack sufficient information, please ask me questions first" — is extremely useful. It prevents the model from rushing to fabricate and instead prompts it to fill in critical missing information first.

## What Are Tokens?

When using LLMs, you will often encounter a term: **token**.

You can roughly think of a token as: **the small granules that a model uses when processing text**.

![Tokens are the small text granules in a model's perspective](./svg/chapter_llm_00_token_particles.svg)

For example, take this sentence:

```text
I want the large model to help me summarize this article.
```

Internally, the model does not read this sentence holistically like a human. It breaks the text into smaller units. In Chinese, a token might be roughly one character, one word, or one punctuation mark. In English, a token might be a whole word or part of a word.

Regular users do not need to calculate tokens precisely. Just remember one thing:

> **The longer the input, the longer the output — the more tokens are consumed.**

"Input" here includes more than just the question you just typed. It can also include:

- The history of your previous conversation turns.
- Documents you upload.
- Code you paste.
- System instructions given to the model.
- Content returned by tool calls.

The model's response to you becomes the output tokens.

![Where the token cost of a single call comes from](./svg/chapter_llm_00_token_cost.svg)

Thus, the cost of a single LLM call can be roughly understood as:

```text
Total cost ≈ input token cost + output token cost
```

For many models, output tokens are more expensive than input tokens because generating content requires more computation.

## How Much Does Using an LLM Cost?

If you are using consumer-facing products like ChatGPT, Claude, Gemini, Kimi, Doubao, Tongyi, or DeepSeek, you typically do not need to calculate per-token costs yourself. What you are more likely to encounter are free quotas, membership subscriptions, usage limits, or plan caps.

If you are a developer calling models via API, you need to pay closer attention to token costs.

As of this book's writing, LLM APIs have become much cheaper compared to earlier days. Tasks like casual chat, short text polishing, and simple summaries typically incur very low per-call costs. But if you ask the model to process long documents, entire codebases, or let an Agent execute many steps autonomously, the cost can rise significantly.

![The token consumption ladder for different tasks](./svg/chapter_llm_00_cost_pyramid.svg)

Use the following table to build intuition:

| Usage Scenario | Token Consumption Feel | Cost Reminder |
|---|---|---|
| **Ask a simple question** | Very low | Usually nothing to worry about |
| **Polish a short passage** | Very low | Suitable for frequent daily use |
| **Write a short article** | Low | The longer the output, the higher the cost |
| **Summarize a long article** | Medium | The longer the article, the more input |
| **Analyze a multi-page PDF** | Higher | Watch context length and fees |
| **Analyze an entire codebase** | Higher | May need batch processing |
| **Agent autonomously executes dozens of steps** | Potentially very high | Set budgets and stopping conditions |

Pay special attention to multi-turn conversations.

Many chat products send the conversation history to the model each time so the model can "remember what was said earlier." This is convenient, but it also means the longer the conversation, the more input tokens may be used.

If you find a conversation has grown very long, consider starting a new one, first asking the model to summarize the key points from before, and then bringing that summary into the new conversation.

## Which AI Should a Beginner Actually Use? How to Download? Should You Pay?

Many beginners' biggest concern is not model leaderboards, but three very practical questions: **Which one should I use right now? Where do I download it? Do I need to pay?**

Here's the direct takeaway:

> **For absolute beginners, don't research all models at once. First pick one chat product that's easy to access, has good Chinese-language experience, and offers sufficient free quota. Use it for a week before deciding whether to pay.**

![How to choose a model: look at the task first, not the leaderboard](./svg/chapter_llm_00_model_choice_tree.svg)

### If You Just Want to Start Right Now, Here's How to Choose

| Your Situation | Recommended First Choice | Why |
|---|---|---|
| **Just want to experience AI, write, summarize, learn** | `Doubao`, `Kimi`, `Tongyi Qianwen`, `Wenxin Yiyan`, `Zhipu Qingyan` | Good Chinese experience, low registration barrier, usually free quota available |
| **Often read long documents, papers, reports** | `Kimi`, `Claude`, `Gemini` | Better suited for long-text reading, summarization, and point extraction |
| **Want strong all-around ability — writing and coding** | `ChatGPT`, `Claude`, `Gemini` | Strong general capabilities for Q&A, writing, coding, and complex tasks |
| **Mainly write code, modify projects** | `Cursor`, `Trae`, `GitHub Copilot`, `Claude Code` | Can integrate with code editors or project files, not just chat |
| **Want low-cost API access or local deployment** | `DeepSeek`, `Qwen`, `Llama` | More developer-oriented; ordinary beginners can skip for now |

If you still don't know how to choose, follow this order:

1. **Step 1**: Start with a domestic chat product, such as `Doubao` or `Kimi`.
2. **Step 2**: If you can normally access overseas services, try `ChatGPT`, `Claude`, or `Gemini`.
3. **Step 3**: If you start writing code or modifying projects, consider programming tools like `Cursor`, `Trae`, or `GitHub Copilot`.

For most absolute beginners, **one tool for the first week is enough**. Don't switch from one to another every day — otherwise you'll struggle to tell whether the tool is the problem or your prompting approach is still unclear.

### Where to Download or Access?

The safest approach is: **Prioritize official websites and official app stores. Do not casually download installation packages from unknown sources.**

| Access Method | How to Start | Who It's For |
|---|---|---|
| **Web** | Search the product name in your browser, go to the official site, and log in | Computer users, people who don't want to install software |
| **Mobile App** | Search the product name in the App Store or Android app store | Daily chat, photo recognition, on-the-go use |
| **Desktop Client** | Download from the product's official website, e.g., `Cursor`, `Trae` | People who need to read/write code projects or do extended office work |
| **Editor Plugin** | Install from the `VS Code`, `JetBrains`, or other editor plugin marketplace | People who are already writing code |

Beginners can remember these principles:

- **Use the web version first if possible**: No installation, lowest risk.
- **Mobile is suitable for daily use**: For example, photo recognition, voice questions, quick summaries.
- **Don't install programming tools too early**: If you're not coding yet, a chat product is enough.
- **Verify the official source**: Before downloading, confirm the product name, developer, and official website to avoid counterfeit apps.

### Quick Reference: Official Entry Points for Common AI / Agent Tools

The table below is not a leaderboard. It is a **safe entry-point checklist** for beginners. Links and product forms may change — always rely on official websites, official app stores, and project READMEs when downloading.

| Tool / Product | Type | Official Entry | How Beginners Can Use It |
|---|---|---|---|
| `ChatGPT` | Chat product / general AI assistant | [ChatGPT](https://chatgpt.com/) / [OpenAI](https://openai.com/chatgpt/) | General Q&A, writing, learning, code explanation |
| `Claude` | Chat product / long-document & code assistant | [Claude](https://claude.ai/) | Long-document reading, rigorous writing, code comprehension |
| `Gemini` | Chat product / multimodal assistant | [Gemini](https://gemini.google.com/) | Images, video, long materials, and Google ecosystem |
| `DeepSeek` | Chat product / model API | [DeepSeek Chat](https://chat.deepseek.com/) / [DeepSeek API Platform](https://platform.deepseek.com/) | Regular users use the chat web; developers explore the API |
| `Doubao` | Chinese chat product / multimodal assistant | [Doubao](https://www.doubao.com/) | Chinese daily use, writing, search, image understanding |
| `Kimi` | Chinese chat product / long-document reader | [Kimi](https://kimi.moonshot.cn/) | Uploading materials, summarizing long documents, reading papers and reports |
| `Qwen / Tongyi Qianwen` | Chinese chat product / open-source model ecosystem | [Qwen Chat](https://chat.qwen.ai/) / [Tongyi Qianwen](https://tongyi.aliyun.com/qianwen/) / [Qwen GitHub](https://github.com/QwenLM) | Regular users use the chat entry; developers explore open-source models and API |
| `Cursor` | AI IDE / coding Agent | [Cursor](https://www.cursor.com/) | Writing code and modifying projects in a VS Code-like editor |
| `Trae` | AI IDE / coding Agent | [Trae International](https://www.trae.ai/) / [Trae China](https://www.trae.com.cn/) | Chinese developers using AI coding in an IDE |
| `GitHub Copilot` | Coding assistant / IDE plugin | [GitHub Copilot](https://github.com/features/copilot) | Code completion and explanation in VS Code, JetBrains, Visual Studio, etc. |
| `Claude Code` | Terminal coding Agent | [Claude Code](https://www.anthropic.com/claude-code) / [Official Docs](https://docs.anthropic.com/en/docs/claude-code/overview) | Reading projects, editing files, running tests in the terminal; beginners should start with small tasks |
| `Codex` | OpenAI coding Agent / CLI | [OpenAI Codex GitHub](https://github.com/openai/codex) / [OpenAI](https://openai.com/) | Developers handling code tasks in terminal or ChatGPT ecosystem |
| `WorkBuddy` | Desktop Agent / office Agent | [WorkBuddy](https://www.codebuddy.cn/work/) | Processing files, spreadsheets, PPTs, meeting notes, and multi-step office tasks |
| `OpenClaw` | Open-source local Agent / automation platform | [OpenClaw GitHub](https://github.com/openclaw/openclaw) | More suited for users willing to tinker with deployment; beginners should read the README and permission notes first |
| `Hermes Agent` | Open-source self-learning Agent | [Hermes Agent](https://hermes-agent.nousresearch.com/) / [GitHub](https://github.com/NousResearch/hermes-agent) | Developers interested in researching open-source Agent memory and skill-learning mechanisms |

If you just want to start using AI, you don't need to install all of these. A more sensible order is: **Start with chat products like `Doubao`, `Kimi`, `ChatGPT`. When you start coding, use `Cursor`, `Trae`, `Copilot`. When you need AI to truly operate on files and workflows, consider Agent tools like `WorkBuddy`, `OpenClaw`, or `Hermes Agent`.**

### Is the Free Version Enough? When Should You Pay?

Beginners generally **do not need to pay right away**. Start with the free version for these tasks:

- Explain a concept.
- Summarize an article.
- Polish a passage.
- Create a study plan.
- Help revise an email.
- Analyze a screenshot or a short document.

If after a week of continuous use you find it genuinely saves you time, then consider paying.

| Situation | Suggestion |
|---|---|
| **Just occasional questions** | Free version is usually enough |
| **Daily use for learning, writing, office work** | Consider trying a one-month membership |
| **Frequently upload long documents, images, spreadsheets** | Paid versions usually have fewer limits and more stable experience |
| **Frequently write code, debug projects** | Consider `Cursor`, `Copilot`, or a stronger chat model subscription |
| **Just want to try it out** | Not recommended to pay annually at first — start monthly or with a free trial |
| **Want to call model APIs in programs** | This is a developer scenario, typically billed by token or call volume; beginners can skip for now |

Two things to note when paying:

1. **Paying does not mean answers are necessarily correct**: Important facts, numbers, legal, medical, and financial advice still need verification.
2. **Don't subscribe to too many tools at once**: Regular users should stick with one primary tool first and switch only when they hit a clear bottleneck.

A simple selection guideline:

> **Don't want to tinker: Start with `Doubao` or `Kimi`. Want stronger all-around capability: Then try `ChatGPT`, `Claude`, or `Gemini`. Starting to code seriously: Then use `Cursor`, `WorkBuddy`, or `GitHub Copilot`.**

Most importantly:

> **Don't ask "which model is always the best." Ask "for this task right now, which one makes it easiest."**

## From Chat Window to AI Tools: How Should Regular People Choose?

Earlier we discussed "which model to choose." But when you actually start using AI, you will quickly run into another question:

- Should I use chat products like `ChatGPT`, `Claude`, `Gemini` directly, or specialized AI programming tools?
- `Claude Code`, `Codex`, `Cursor`, `Trae` all seem to write code — what's the difference?
- Are desktop Agents like `WorkBuddy`, `OpenClaw` more advanced?
- Do I need to care about engineering concepts like `Harness` and `Trace` right now?

Start by remembering a very practical distinction:

> **The model is the "brain"; the tool is the "workbench, hands and feet, permissions, and process record."**

The same model, placed in a chat window, mainly answers you. Placed in a coding Agent, it can read projects, edit files, and run tests. Placed in an office Agent, it might also organize documents, manipulate spreadsheets, and generate PPTs. Placed in an enterprise Agent system, it needs permissions, logging, evaluation, and auditing to ensure safety and reliability.

![From chat models to AI tools: which to choose](./svg/chapter_llm_00_agent_tool_choice.svg)

So beginners don't need to research all tools at once. Just ask yourself three questions first:

| Ask Yourself First | If the Answer Is | Better Choice |
|---|---|---|
| **Does it need to operate on my files or code?** | No — just Q&A, writing, summarizing | Chat product |
| **Does it need to understand and modify a project?** | Needs to read code, edit files, run tests | Coding Agent / AI IDE |
| **Does it need long-running execution, tool calls, or sensitive data handling?** | Needs automation, permissions, auditing, rollback | Agent Harness / Trace / enterprise workflow |

In one sentence:

> **First learn to prompt with chat products, then handle real tasks with Agents, and only then consider engineering systems like Harness and Trace.**

### First, Distinguish: Models, Chat Products, Agents, Harness, Trace Are Not the Same Thing

These terms are often mixed together, but they sit at different layers.

| Term | More Like | Typical Examples | How Beginners Can Understand It |
|---|---|---|---|
| **Model** | Brain | `GPT`, `Claude`, `Gemini`, `DeepSeek`, `Qwen` | Responsible for understanding, reasoning, and generating content |
| **Chat Product** | Conversation window | `ChatGPT`, `Claude`, `Kimi`, `Doubao`, `Tongyi` | Best for beginners — Q&A, writing, summarizing, explaining |
| **AI IDE / Coding Agent** | An AI colleague that can modify code | `Claude Code`, `Codex`, `Cursor`, `Trae`, `Copilot` | Can read projects, edit files, run commands, fix bugs |
| **Office / Desktop Agent** | An AI assistant that operates office tools | `WorkBuddy`, `OpenClaw` | Can process files, spreadsheets, PPTs, messages, and multi-step workflows |
| **Harness** | The shell and console for an Agent | Tool orchestration, permissions, memory, evaluation, logging | Makes Agents execute tasks safely, stably, and controllably |
| **Trace / Agent Trace** | Process record | Execution logs, code attribution, audit trails | Records what the AI did, enabling inspection, accountability, and rollback |

Think of them as a progressively enhanced system:

```text
Model → Chat Product → Agent Tool → Harness → Trace / Audit / Evaluation
```

The further right you go, the more the AI can do, but the higher the risk and management cost. For beginners, the right order is not "jump straight to the strongest tool," but rather **start from low-risk scenarios and gradually delegate more authority**.

### 1. Chat Products: The Best First Stop for Beginners

If your tasks are mainly reading, writing, editing, summarizing, explaining, or planning, chat products are usually sufficient.

Common choices include:

- `ChatGPT`: Mature all-around capability and tool ecosystem, suitable for general Q&A, writing, and code assistance.
- `Claude`: Typically excels at long-document reading, precise expression, and rigorous summarization.
- `Gemini`: Strong multimodal and long-context capabilities, suitable for images, video, long materials, and Google ecosystem.
- `Kimi`: Friendly Chinese long-document reading experience, suitable for material summarization, paper reading, and report reading.
- `Doubao`, `Tongyi`, `Wenxin`, `GLM`: Suitable for Chinese office work, content generation, and domestic application ecosystems.
- `DeepSeek`, `Qwen`: Often used by developers for coding, reasoning, low-cost API, or local deployment.

For example, you could ask:

```text
I have a 20-page industry report.
Please first summarize it into 10 key points, then list 3 risks I should focus on.
Requirement: Do not fabricate information not present in the report.
```

The advantages of chat products are low barrier to entry, low risk, and fast feedback. They don't directly operate on your computer or automatically modify files, making them perfect for building your basic feel for using AI.

**Tasks they're suited for**:

- Learning a new concept.
- Polishing a passage.
- Summarizing an article or report.
- Creating a study plan, work plan, or activity proposal.
- Explaining code or an error message.

If you're still unsure what tool to use, start with a chat product — you generally can't go wrong.

### 2. AI IDE / Coding Agent: For Development Tasks in Real Projects

When the task shifts from "answer a question" to "please modify this project for me," a chat window becomes inconvenient.

This is where AI coding tools like `Claude Code`, `Codex`, `Cursor`, `Trae`, `Copilot` come in. What they share: they don't just answer questions — they can work around a codebase.

They can typically help you:

- Read project structure.
- Search for functions and call relationships.
- Explain unfamiliar code.
- Modify files.
- Continue fixing based on error messages.
- Generate test cases.
- Run tests or commands and iterate based on results.

Different tools can be roughly understood as follows:

| Tool | Better Suited For | Beginner Usage Advice |
|---|---|---|
| `Claude Code` | Multi-step code tasks in projects — reading code, editing files, running tests | Suitable for medium-to-large projects, but start with small changes the first time |
| `Codex` | Code generation, scripting, automated development in the OpenAI ecosystem | Writing scripts, supplementing tests, fixing small bugs |
| `Cursor` | Writing and asking questions in an AI editor, generating diffs, understanding projects | Suitable for those used to a VS Code-style workflow |
| `Trae` | AI-native IDE / Agent coding experience | Suitable for those wanting Agent capabilities inside an IDE |
| `Copilot` | Daily completion, code explanation, generating local functions or tests | Suitable for low-friction embedding into existing IDEs |

A safer usage approach: let it read first, then let it modify.

```text
Please first read the login flow of this project — do not modify code yet.
First tell me:
1. Where the login entry point is;
2. Where tokens are generated and stored;
3. If I want to add SMS verification, which files need to change.
```

Once it explains clearly, then let it execute:

```text
Modify according to the plan just discussed.
Requirements:
1. Do not change the existing password login logic;
2. Add SMS verification code validation;
3. Run relevant tests after modification;
4. Finally summarize which files were changed.
```

**Beginner reminder**: Don't start by saying "refactor my entire project." A better approach is to let the AI first explain the flow, locate issues, modify a small feature, and then gradually expand the task scope.

### 3. Office / Desktop Agent: For Multi-File, Multi-Tool, Multi-Step Tasks

If your tasks are not coding but processing documents, spreadsheets, PPTs, folders, messages, and workflows, then office or desktop Agents like `WorkBuddy`, `OpenClaw` are more appropriate.

The difference from chat products: chat products mainly "answer," while desktop Agents emphasize "execution."

For example, you can ask it to:

- Organize materials in a folder.
- Summarize multiple customer interview transcripts.
- Extract key information from a spreadsheet.
- Generate a presentation outline or PPT first draft.
- Convert daily reports, weekly reports, meeting notes into to-do items.
- Call external tools and send results to team groups or workflows.

`WorkBuddy` is more like an office assistant for workplace users, suitable for documents, spreadsheets, PPTs, meeting notes, and workflow organization.

`OpenClaw` is more like an extensible Agent workbench, focused on letting AI connect tools, execute skills, call interfaces, and even integrate with Feishu, Telegram, Discord, and other workflows.

You could describe a task like this:

```text
Please organize the 20 customer interview transcripts in this folder.
Output:
1. Each customer's core needs;
2. Top 5 frequent issues;
3. A three-page summary outline suitable for a PPT.
```

For automated workflows, you could say:

```text
Every morning at 9 AM, read the daily reports in the designated folder,
summarize new issues from yesterday,
rank by severity,
then send to the team group.
```

These tools are highly efficient but require more attention to permissions. Any Agent that can read/write local files, access enterprise data, send messages, or call interfaces should have its scope limited upon first use — test with a sandbox folder or non-sensitive materials first.

### 4. Harness: The "Shell" That Enables Agents to Work Safely and Stably

`Harness` is not a product that ordinary users open daily to chat with. It is more like the engineering shell of an Agent system.

For an Agent to do real work reliably, a model alone is not enough. It needs a whole set of supporting capabilities:

- **Tool List**: What tools can it invoke?
- **Permission Control**: Which files can it read? Which commands can it execute? Which operations require human confirmation?
- **Memory System**: What information needs long-term storage? What must not be saved?
- **Execution Environment**: Where does the code run? How does it stop on failure? Can it roll back?
- **Evaluation Mechanism**: How do you judge whether it did the right thing?
- **Logging and Auditing**: What did it do at each step? Who approved it?

Together, these capabilities can be understood as the Agent's `Harness`.

In one sentence:

> **The model determines how smart the Agent is; the Harness determines whether the Agent can do work safely, stably, and controllably.**

If you're just an individual user, knowing the concept is enough for now. If you're deploying Agents in a team or enterprise, `Harness` becomes very important.

### 5. Trace: Leaving an Inspectable Trail of the AI's Execution Process

`Trace` or `Agent Trace` refers to the mechanism for recording the Agent's execution process.

When AI is only helping you rewrite a single sentence, complex recording isn't needed. But when it starts modifying code, processing customer data, calling interfaces, and sending messages, you need to know:

- Which files did it read?
- What content did it modify?
- Why did it make that decision?
- Which step introduced the problem?
- Did a human review it?
- Can accountability be traced and rollback performed if something goes wrong?

This is the value of `Trace`: **not making the AI better at answering, but making what the AI did inspectable, reviewable, and accountable.**

Also note a similarly named tool: `Trae`.

| Name | More Like | Focus |
|---|---|---|
| `Trace / Agent Trace` | Process recording and auditing mechanism | Records what the AI did, enabling inspection, attribution, and rollback |
| `Trae` | AI coding IDE / Agent tool | Helps developers write code, read projects, edit files |

Simply put: `Trace` is "recording the process"; `Trae` is "helping you code."

### Which Should You Actually Use? A Choice Roadmap for Beginners

If you still don't know how to choose, follow the roadmap below.

| Who You Are / What You Want to Do | Priority Choice | Don't Rush Into |
|---|---|---|
| **Absolute beginner, just want to experience AI** | `ChatGPT`, `Claude`, `Kimi`, `Doubao`, `Tongyi` | `OpenClaw`, `Harness` |
| **Student / writer / office user** | `Claude`, `ChatGPT`, `Kimi`, `WorkBuddy` | Complex Agent frameworks |
| **Just started learning to code** | `ChatGPT`, `Claude`, `Copilot`, `Cursor` | Letting an Agent auto-refactor large projects |
| **Everyday developer** | `Claude Code`, `Codex`, `Cursor`, `Trae` | Auto-execution scripts without permission control |
| **Team tech lead** | Coding Agent + `Trace` + code review process | Only looking at model leaderboards without building processes |
| **Want to build automated workflows** | `WorkBuddy`, `OpenClaw`, `MCP`, `Harness` | Giving an Agent full-disk permissions directly |
| **Enterprise internal deployment** | `Harness`, permission system, auditing, evaluation, private models | Processing sensitive data with only a personal chat account |

A more sensible onboarding order is:

1. **Start with chat products**: Learn to articulate tasks clearly.
2. **Then use AI IDE / coding Agent**: Let AI read code, modify small features, run tests.
3. **Then use office / desktop Agent**: Let AI process files, spreadsheets, PPTs, and multi-step tasks.
4. **Finally consider Harness and Trace**: Bring these in when you need long-running execution, team collaboration, permission control, and auditing.

Most importantly, don't pursue "full automation" from the start.

> **The more automated the AI tool, the more gradually you should delegate authority: first let it suggest, then let it modify, and only finally let it execute.**

## A 7-Day Beginner Practice Plan

If you're just starting with LLMs, spend a week on the following exercises.

These exercises require no programming and no professional background. The key is to integrate them into your real life and work.

| Day | Exercise | Example Prompt |
|---|---|---|
| **Day 1** | Explain a concept you recently didn't understand | "Use a real-life analogy to explain what a large model is" |
| **Day 2** | Revise something you wrote yourself | "Rewrite this passage to be clearer and more natural" |
| **Day 3** | Summarize an article | "Summarize this article into 5 key points and list the author's views" |
| **Day 4** | Have it act as a teacher | "Create 5 practice problems around this knowledge point, with answers" |
| **Day 5** | Create a plan | "Help me design a 14-day spoken English practice plan" |
| **Day 6** | Do a decision analysis | "Should I buy a tablet or a laptop? List pros and cons by scenario" |
| **Day 7** | Complete a first draft of a real task | "Help me write an email requesting a homework deadline extension — keep the tone polite" |

As you practice, consciously observe three things:

1. When you provide more information, do the answers become more aligned with your needs?
2. When you specify an output format, are the results easier to use?
3. When you point out issues and ask for revisions, do the results noticeably improve?

If the answer is yes, you have already grasped the most important part of using LLMs: **they are not one-shot answer machines, but iterable collaborators.**

## What Are LLMs' Most Common Mistakes?

LLM responses are often fluent, confident, and even beautifully formatted. But that doesn't mean they're necessarily correct.

![The five most common types of LLM mistakes](./svg/chapter_llm_00_risk_cards.svg)

There are five main categories of risk.

### 1. Confidently Fabricating

LLMs may invent non-existent papers, authors, links, legal provisions, statistics, or even give citations that look real.

So when you ask factual questions, it's best to require sources and verify them yourself.

### 2. Treating Old Information as Current

Some models lack internet access; some models' knowledge is not updated promptly. They may not know about the latest policies, latest product prices, latest model names, or latest company developments.

If the question involves "current time," it's better to use a tool with web search capability, or look up official sources yourself.

### 3. Getting Math and Precise Calculations Wrong

LLMs are good at explaining mathematical reasoning, but not always good at precise calculations. Simple arithmetic can also be wrong; complex calculations should definitely be handed off to calculators, spreadsheets, or code.

A practical approach: have the model write out the calculation process, then verify the result with a reliable tool.

### 4. Misunderstanding Your Real Intent

If you just say "write me a proposal," it doesn't know whether you want a business proposal, course proposal, event proposal, or project proposal.

The less background you provide, the more it has to guess; the more it guesses, the higher the chance of going off track.

### 5. Correct Format but Incorrect Content

This is the most deceptive category.

LLMs can present wrong content in a very tidy way: with headings, tables, numbering, and conclusions — looking very professional. But good formatting does not equal reliable content.

Therefore, the more important something is, the less you should rely on just one query.

You can have the model perform these self-checks:

```text
Please check your previous answer: which parts are certain, and which parts need further verification.
```

```text
Please critique this proposal from an opposing perspective and point out the three most likely reasons it could fail.
```

```text
Please list the key assumptions your answer depends on. If these assumptions don't hold, how would the conclusions change?
```

This kind of follow-up questioning can significantly improve result quality.

## From "Using" to "Mastering": What to Learn Next?

When you start consciously providing background, setting requirements, checking results, and asking follow-up questions, you have in fact entered the world of Prompt Engineering.

![The loop for using LLMs correctly](./svg/chapter_llm_00_use_loop.svg)

From here, you can continue learning several more advanced topics to build a holistic understanding of Agents and the LLM fundamentals that drive them. If you only wanted to learn how to use LLMs, reading this far is enough.

| Chapter | Content | What You'll Gain |
|---------|---------|-----------------|
| **Chapter 1** What is an Agent? | Agent definition, architecture, history, and use cases | A complete conceptual framework |
| **Chapter 2** LLM Fundamentals | LLM principles, Prompt Engineering, API calls | Confident mastery of the Agent's "brain" |
---

*Start learning: [Chapter 1: What is an Agent?](./chapter_intro/README.md)*
