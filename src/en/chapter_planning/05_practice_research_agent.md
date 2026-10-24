# 5.5 Hands-On: Automated Research Assistant Agent

Integrating the planning, reasoning, and reflection skills learned in this chapter to build an Agent capable of conducting autonomous research.

> **Design Note**: This project adopts a "Plan-then-Execute" multi-stage Pipeline architecture rather than a pure ReAct loop. This is because in a research task, each stage (planning → searching → analysis → quality check) follows a clear sequential order, and the Pipeline pattern makes it easier to control the flow and debug. Within the Pipeline, each stage still applies ReAct thinking — the Agent "thinks" about the next action based on the current stage's output, and "reflects" during the quality check stage. This embodies the principle discussed in Section 5.1: "apply the right reasoning framework to the right scenario."

> **Cutting-Edge Positioning**: The research assistant in this section is an introductory form of the **Deep Research Agent**. A true Deep Research Agent does not just "search a few times and then summarize"; it is capable of continuously raising sub-questions around open-ended problems, cross-validating evidence across sources, managing citations, identifying contradictions, and progressively converging on conclusions over multiple rounds of research.

## What Is Auto Research?

**Auto Research** refers to having an Agent act like a junior researcher, autonomously completing the full research workflow around an open-ended topic: "raising questions → formulating a plan → searching for materials → reading and excerpting → cross-validating → synthesizing into a report → self-reviewing."

The biggest difference between Auto Research and ordinary search is: ordinary search focuses on "finding an answer," while Auto Research focuses on "forming a credible conclusion." Therefore, Auto Research is not about summarizing search engine results, but about decomposing the research process into a trackable, verifiable, and iterable workflow.

| Capability | Ordinary Search | Auto Research |
|------------|----------------|---------------|
| **Input** | A specific question | An open-ended topic or vague objective |
| **Process** | Search → Summarize | Plan → Multi-round retrieval → Read → Verify → Synthesize → Review |
| **Evidence** | Often relies on top few results | Maintains sources, timestamps, credibility, and citation chains |
| **Quality Control** | User judges manually | Agent proactively checks for omissions, contradictions, and opposing viewpoints |
| **Output** | Short answer | Structured report with conclusions, uncertainty, and follow-up questions |

A mature Auto Research Agent typically contains five core modules:

1. **Research Planner**: Decomposes the topic into research questions, search queries, and report outlines.
2. **Data Collector**: Invokes tools such as search, web reading, paper retrieval, and database queries.
3. **Evidence Manager**: Converts materials into "evidence cards," recording source, summary, credibility, and related conclusions.
4. **Synthesis Writer**: Organizes materials by outline, generating structured analysis rather than simply splicing together summaries.
5. **Quality Reviewer**: Checks coverage, citation support, contradictory information, timeliness, and potential bias.

## Representative Frontier Work: The Research Lineage of Auto Research

Auto Research did not emerge as a product concept out of nowhere; it evolved gradually from directions such as **long-form question answering, browser-assisted QA, retrieval-augmented writing, Web Agents, and automated scientific discovery**. The following works represent key capabilities in this evolutionary chain.

### WebGPT: Turning "Web Browsing" into a Learnable QA Behavior

- **Paper Link**: [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332)
- **Core Contribution at the Time**: In 2021, this paper transformed "searching the web—reading webpages—citing evidence—generating long answers" into a task that can be optimized through imitation learning and human feedback, laying an early paradigm for subsequent browser Agents, Deep Research Agents, and QA systems with citations.

OpenAI's **WebGPT: Browser-assisted question-answering with human feedback** is a critically important early work. The question it addresses is: when a model faces open-domain long-form questions, can it use a browser like a human to look up materials, excerpt evidence, and generate answers with citations?

The core of WebGPT is not simply accessing a search API, but modeling the browsing process as a sequence of learnable actions: search, open webpages, scroll, cite snippets, and compose answers. Researchers then train the model using human preference feedback, teaching it "which browsing trajectories and answers are more credible." The inspiration for Auto Research is: **research capability comes not only from final generation, but also from a supervised and evaluatable material-gathering process.**

From an engineering perspective, WebGPT corresponds to three modules in this section:

- **Data Collector**: Searching and browsing are not one-shot calls but trackable action sequences.
- **Evidence Manager**: Key claims in the answer should be traceable back to specific webpage snippets.
- **Quality Reviewer**: Human preferences or automated evaluation can be used to train "more credible research trajectories."

### STORM: Generating High-Quality Long-Form Article Outlines Through Multi-Perspective Questioning

- **Paper Link**: [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207)
- **Core Contribution at the Time**: In 2024, this work shifted the focus of automated research writing from "directly generating text" forward to the "pre-writing stage," proposing to synthesize topic outlines through retrieval and multi-perspective questioning, enabling LLMs to more systematically cover key dimensions of open-ended topics.

Stanford OVAL's **STORM: Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking** further advances the problem to "writing a Wikipedia-like long article from scratch." It addresses not a single Q&A, but the "pre-writing stage" in long-form knowledge organization: before writing, how to decide which angles to cover, which questions to ask, and which materials to retrieve.

STORM's key mechanism is **multi-perspective question asking**. The system simulates questioners from different backgrounds, probing the same topic from multiple angles, and then synthesizes an article outline based on retrieval results. The significance is that the hardest part of an open-ended topic is not generating text, but knowing "which dimensions are still missing."

This provides very direct inspiration for Auto Research:

- Do not just have the Agent generate a single search query; have it generate a set of complementary perspectives.
- The report outline should come from "problem space exploration," not from the model intuitively listing sections.
- Quality evaluation should check coverage: whether historical context, core mechanisms, controversial viewpoints, application cases, and limitations have been overlooked.

### MindSearch: Constructing Deep Search Graphs with WebPlanner + WebSearcher

- **Paper Link**: [MindSearch: Mimicking Human Minds Elicits Deep AI Searcher](https://arxiv.org/abs/2407.20183)
- **Core Contribution at the Time**: In 2024, this work explicitly modeled complex web search as a multi-Agent collaborative process of "planner building a problem graph + searcher node-by-node retrieval," upgrading search from one-shot queries to scalable, backtrackable deep information exploration.

**MindSearch: Mimicking Human Minds Elicits Deep AI Searcher** models deep search as a multi-agent collaborative framework. Its typical architecture includes `WebPlanner` and `WebSearcher`: the former is responsible for decomposing complex problems into a dynamically expanding sub-problem graph, and the latter is responsible for executing searches and reading for each sub-problem.

The key value of MindSearch is that it does not treat the research plan as a one-time list, but as a graph structure that continuously expands with search results. Whenever a new entity, relationship, or gap is discovered, the planner can continue adding nodes. This brings it closer to the human research process: first having a rough framework, then continuously discovering new branches through reading.

For the implementation in this section, MindSearch reminds us:

- `research_questions` should not just be a static array; it can be upgraded to a "research problem graph."
- Each search result should feed back to the planner to decide whether to add new sub-problems.
- For complex topics, the stopping condition should not just be search count, but whether key nodes have been adequately covered.

### WebSailor: A Post-Training Paradigm for High-Uncertainty Web Tasks

- **Paper Link**: [WebSailor: Navigating Super-human Reasoning for Web Agent](https://arxiv.org/abs/2507.02592)
- **Core Contribution at the Time**: In 2025, this work targeted high-uncertainty, multi-hop, strongly distracting Web tasks like BrowseComp, proposing to use complex task generation, reasoning trajectory reconstruction, and post-training methods to enable open-source Web Agents to learn to systematically reduce uncertainty.

The 2025 work **WebSailor: Navigating Super-human Reasoning for Web Agent** represents a more cutting-edge direction: not just relying on prompts to make models search-capable, but using specialized data construction and post-training to enable Web Agents to handle tasks with high uncertainty, unclear paths, and multi-hop reasoning requirements.

The tasks it focuses on are similar to BrowseComp: answers are often hidden among multiple webpages, multiple entities, and indirect clues. Ordinary search models tend to stop at surface results, while strong Web Agents need to systematically reduce uncertainty: first locate clues, then eliminate noise, then validate across pages, and finally provide an answer.

WebSailor's inspiration for Auto Research:

- High-difficulty research tasks require "proactively eliminating uncertainty," not passively summarizing search results.
- Training data must include not only answers but also effective search trajectories and reasoning trajectories.
- Complex Web Agent capability comes from the joint training of "retrieval + reasoning + verification," not from enhancing a single module in isolation.

### The AI Scientist: From Research Reports to Automated Scientific Discovery

- **Paper Link**: [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292)
- **Core Contribution at the Time**: In 2024, this work advanced the Research Agent from "writing surveys/reports" to an end-to-end automated scientific discovery pipeline of "proposing research ideas, writing code experiments, analyzing results, writing papers, and performing simulated peer review," demonstrating a prototype of open-ended research automation.

Sakana AI's **The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery** pushes Auto Research in a more radical direction: not just writing research surveys, but attempting to automatically propose research ideas, retrieve related work, design experiments, run code, analyze results, write papers, and even perform automated peer review.

Such systems are still far from replacing real scientists, but they reveal the upper-bound form of Auto Research: a Research Agent is no longer just a "material organizer" but may become a composite system of "hypothesis generator + experiment executor + paper writer + reviewer."

The inspiration for engineering practice is: if the research object involves experimentally verifiable questions, it cannot stop at web retrieval and must also include:

- **Experiment Plan Generation**: Translating research questions into executable experiments.
- **Code Execution Environment**: Running experiments and recording results.
- **Result Interpreter**: Distinguishing real discoveries, random fluctuations, and experimental errors.
- **Automated Reviewer**: Reviewing conclusions from the perspectives of novelty, validity, and reproducibility.

### Summary of the Research Lineage

The works above form more of a "capability foundation" for Auto Research: WebGPT addresses browsing and citation, STORM addresses multi-perspective outlines, MindSearch addresses dynamic search graphs, WebSailor addresses high-uncertainty Web reasoning, and The AI Scientist addresses the extension from research reports to experimental closed loops. After truly entering 2025, the frontier focus has shifted further from "can it do research" to three harder questions: **can it productively complete long-term research tasks, can it be stably evaluated on high-difficulty benchmarks, and can open-source models and frameworks replicate closed-source Deep Research capabilities.**

| Representative Work | Research Question | Core Mechanism | Inspiration for Auto Research |
|---------------------|-------------------|----------------|-------------------------------|
| **WebGPT** | How to enable models to use browsers to answer open-domain long-form questions | Browsing action modeling + evidence citation + human feedback | Research trajectories should be supervised, citable, and evaluable |
| **STORM** | How to generate structured long-form knowledge articles from scratch | Multi-perspective questioning + retrieval + outline synthesis | Explore the problem space before writing the report |
| **MindSearch** | How to progressively unfold complex searches like a human | `WebPlanner` + `WebSearcher` + sub-problem graph | Research plans should expand dynamically, not be static lists |
| **WebSailor** | How to handle high-uncertainty, multi-hop Web reasoning tasks | Complex task generation + trajectory reconstruction + post-training | Train Agents to proactively reduce uncertainty |
| **The AI Scientist** | Can we automatically complete an open-ended scientific discovery pipeline | Idea generation + literature retrieval + experimentation + writing + review | Auto Research can be further extended into an automated experiment system |

### Frontier Progress 2025–2026: From Research Prototypes to the Deep Research Competition

If we only introduced the papers above, it would indeed seem "not cutting-edge enough." Because after 2025, Auto Research is no longer just a research prototype in papers; it has become a direction where closed-source products, open-source frameworks, specialized models, and evaluation benchmarks all compete.

#### OpenAI Deep Research: Productizing Long-Term Research Workflows

- **Official Link**: [Introducing deep research](https://openai.com/index/introducing-deep-research/)
- **Core Contribution**: In 2025, OpenAI launched Deep Research as an Agent capability for real users, packaging "research plan → multi-round browsing → note-taking → evidence integration → generating a cited report" into an interactive product workflow.

Its importance lies not in proposing a specific algorithmic breakthrough, but in advancing Auto Research from "can the model search the web" to "can it complete, in a few dozen minutes, an information synthesis task that would take a human researcher several hours." This means the evaluation criteria for Deep Research are no longer just answer accuracy, but also report structure, evidence coverage, citation quality, task persistence, and uncertainty expression.

#### BrowseComp: Evaluating Browsing Agents with "Hard to Find but Easy to Verify" Questions

- **Project Link**: [OpenAI simple-evals / BrowseComp](https://github.com/openai/simple-evals)
- **Core Contribution**: In 2025, OpenAI released BrowseComp, using 1,266 high-difficulty web retrieval questions to evaluate Agents' ability to find hidden information on the real Web, with questions typically requiring cross-webpage, multi-hop clues, and long browsing trajectories to answer.

BrowseComp's value lies in upgrading Web Agent evaluation from "can it answer simple facts" to "can it locate information on the open internet that is difficult to search for directly." This is critical for Auto Research: a true Research Agent cannot only read the first page of search results but must trace clues through noise, eliminate distractions, and verify unique answers.

#### BrowseComp-ZH: The Chinese Internet Makes Deep Research Harder

- **Paper Link**: [BrowseComp-ZH: Benchmarking Web Browsing Ability of Large Language Models in Chinese](https://arxiv.org/abs/2504.19314)
- **Core Contribution**: In 2025, BrowseComp-ZH extended high-difficulty browsing evaluation to the Chinese internet, highlighting challenges in the Chinese context such as information fragmentation, platform dispersion, abbreviated expressions, differences in search entry points, and multi-hop clue problems.

This work reminds us: Deep Research is not something that can be solved by simply translating an English Web Agent to Chinese. The Chinese internet has vast amounts of information distributed across encyclopedias, news, forums, government websites, social platforms, and video platforms, where keywords often do not directly match answers. Therefore, Chinese Auto Research Agents require stronger query rewriting, entity disambiguation, cross-platform retrieval, and source credibility assessment capabilities.

#### Open Deep Research: Open-Source Frameworks Begin to Replicate Deep Research Workflows

- **Project Link**: [LangChain Open Deep Research](https://github.com/langchain-ai/open_deep_research)
- **Core Contribution**: LangChain's Open Deep Research open-sourced the engineering patterns of Deep Research, using components like `LangGraph` to implement research planning, searching, content reading, report generation, and human-feedback adjustment, enabling developers to replicate and customize research Agent workflows.

Its significance lies in pushing Deep Research from a closed-source capability toward a composable engineering framework. For developers, the focus is not just "which model to call" but how to design state machines, manage intermediate notes, control concurrent searches, and feed user feedback back into report planning.

#### Tongyi DeepResearch: Open-Source Specialized Deep Research Model and System

- **Project Link**: [Alibaba-NLP DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)
- **Model Link**: [Tongyi-DeepResearch-30B-A3B](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B)
- **Core Contribution**: In 2025, Tongyi DeepResearch advanced Deep Research from "general-purpose model + Agent framework" to "specialized open-source models and systems trained for long-cycle deep information search tasks," forming a complete engineering stack around data generation, continual training, reinforcement learning, reasoning, and evaluation.

This represents an important trend: future Deep Research may not entirely rely on general-purpose conversational models with prompts, but may see the emergence of Agent models specifically optimized for search trajectories, evidence integration, long-chain reasoning, and report generation.

#### DeepResearch Bench / BrowseComp-Plus: Evaluation Moving from Single-Question Answers to End-to-End Report Quality

- **Project Example**: [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus)
- **Core Contribution**: Around 2026, evaluation began expanding from "providing a verifiable answer" to "assessing complete research reports." New evaluations focus more on comprehensiveness, insight, instruction following, readability, citation quality, and end-to-end research process performance.

This shows that frontier evaluation for Auto Research is diverging into two categories: one like BrowseComp, examining whether the Agent can find hidden facts; the other like DeepResearch Bench, examining whether the Agent can produce high-quality research reports. The former is more like a "search reasoning ability test," while the latter is more like a "researcher work quality test."

### Summary of the Frontier Lineage

| Stage | Representative Progress | Focus Question | Frontier Significance |
|-------|------------------------|----------------|----------------------|
| **Early Browsing QA** | **WebGPT** | How models browse webpages and cite sources | Make search trajectories and evidence citations learnable objects |
| **Retrieval-Augmented Writing** | **STORM** | How to generate high-coverage outlines for open-ended topics | Shift research writing forward to problem space exploration |
| **Deep Search Graphs** | **MindSearch** | How to dynamically decompose and expand complex searches | Turn research plans from static lists into problem graphs |
| **High-Uncertainty Web Reasoning** | **WebSailor** | How to train Agents to solve extremely difficult multi-hop retrieval | Post-training begins targeting search trajectories and uncertainty elimination |
| **Productized Deep Research** | **OpenAI Deep Research** | How to enable users to truly complete long-term research tasks | Deep Research becomes a universal knowledge work entry point |
| **High-Difficulty Browsing Evaluation** | **BrowseComp / BrowseComp-ZH** | How to evaluate deep retrieval capability on the real Web | Force Agents to handle multi-hop, noisy, and cross-lingual Web environments |
| **Open-Source Replication & Specialized Models** | **Open Deep Research / Tongyi DeepResearch** | How to replicate closed-source Deep Research capabilities | Move from prompt engineering to frameworkization, model specialization, and training |
| **End-to-End Research Evaluation** | **DeepResearch Bench / BrowseComp-Plus** | How to evaluate complete report quality | Evaluation moves from answer accuracy to research quality |

Therefore, a more cutting-edge Auto Research Agent should not stop at "search + summarize." It needs to synthesize these advances: retain evidence like WebGPT, ask multi-perspective questions like STORM, maintain a problem graph like MindSearch, proactively reduce uncertainty like WebSailor, productize long-term workflows like OpenAI Deep Research, accept high-difficulty retrieval evaluation like BrowseComp/BrowseComp-ZH, and further absorb the engineering, open-source, and training experiences of Open Deep Research and Tongyi DeepResearch.

## Typical Workflow of Auto Research

```text
User Topic
  ↓
Research Question Decomposition
  ↓
Generate Search Plan & Source List
  ↓
Multi-Round Search & Reading
  ↓
Extract Evidence Cards
  ↓
Cross-Validation & Conflict Detection
  ↓
Generate Report Draft
  ↓
Self-Review: Any missing key dimensions? Do citations support conclusions?
  ↓
Supplemental Retrieval or Output Final Report
```

In engineering, the most important thing is to always make the Agent know "why it is searching this time." A good search should not just be keyword matching, but should correspond to a specific research gap:

- **Concept Gap**: Core concepts have not been clearly defined yet.
- **Fact Gap**: Missing key data, timelines, or cases.
- **Evidence Gap**: Existing conclusions lack reliable source support.
- **Counter-Example Gap**: Only supportive views have been seen, not opposing views.
- **Timeliness Gap**: Materials may be outdated and need the latest information for verification.

## Engineering Challenges of Auto Research

Auto Research may seem like "calling the search tool a few more times," but the real difficulty lies in long-term task control.

| Challenge | Common Problem | Engineering Solution |
|-----------|---------------|---------------------|
| **Search Drift** | Deviating from the original topic during search | Bind each search round to research questions and expected evidence |
| **Evidence Contamination** | Excerpted low-quality, duplicate, or outdated materials | Label sources with credibility, timeliness, and independence tags |
| **Citation Hallucination** | Citations in the report do not support the conclusions | Perform claim-to-source checks before generation |
| **Premature Summarization** | Outputting conclusions with insufficient information | Set coverage thresholds and enforce reverse searches |
| **Context Inflation** | Too many search results causing window explosion | Use evidence cards, hierarchical summaries, and retrieval-based memory |
| **Infinite Research** | Agent keeps searching and never outputs | Set budgets: max rounds, max sources, max cost |

> **Practical Advice**: The goal of Auto Research is not to "exhaust all materials," but to maximize the credibility, coverage, and traceability of conclusions within a given time and cost budget.

## From Search Agent to Deep Research Agent

The goal of a traditional Search Agent is to "find an answer"; the goal of a Deep Research Agent is to "form a credible conclusion." The difference between the two lies not in whether they are connected to the internet, but in whether they possess a **long-term research process**.

| Capability Dimension | Search Agent | Deep Research Agent |
|---------------------|-------------|---------------------|
| **Task Objective** | Answer a specific question | Research an open-ended topic and form a report |
| **Planning Approach** | Generate search queries once | Dynamically decompose research questions, continuously supplement sub-problems |
| **Information Processing** | Summarize top few results | Multi-source cross-validation, deduplication, conflict detection |
| **Context Management** | Save search results | Manage research notes, evidence cards, citation chains |
| **Quality Control** | Simple completeness check | Check coverage, credibility, timeliness, opposing viewpoints |
| **Output Form** | Short answer | Report with citations, structured arguments, and uncertainty statements |

A Deep Research Agent can be understood as a research pipeline composed of multiple sub-capabilities:

![Deep Research Agent Research Pipeline](../svg/chapter_planning_05_research_pipeline.svg)

For teaching simplicity, the code in this section implements only the core skeleton: planning, searching, synthesis, and quality checking. You can gradually expand this into a complete Deep Research Agent.

## Research Assistant Feature Design

![Research Assistant Agent Feature Design](../svg/chapter_planning_05_research_arch.svg)

## Complete Implementation

```python
import json
import datetime
from openai import OpenAI
import requests

client = OpenAI()

class ResearchAssistant:
    """Automated Research Assistant"""
    
    def __init__(self):
        self.research_notes = []
        self.sources = []
    
    def _search(self, query: str) -> str:
        """Search tool (using DuckDuckGo)"""
        try:
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": 1}
            response = requests.get(url, params=params, timeout=8)
            data = response.json()
            
            results = []
            if data.get("AbstractText"):
                results.append(data["AbstractText"])
                if data.get("AbstractURL"):
                    self.sources.append(data["AbstractURL"])
            
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(topic["Text"][:300])
            
            return "\n".join(results) if results else "No relevant results found"
        except Exception as e:
            return f"Search failed: {e}"
    
    def _take_notes(self, content: str, source: str = ""):
        """Record research notes"""
        self.research_notes.append({
            "content": content,
            "source": source,
            "time": datetime.datetime.now().isoformat()
        })
    
    def research(self, topic: str, depth: str = "standard") -> str:
        """
        Execute research
        
        Args:
            topic: Research topic
            depth: "quick"=quick overview, "standard"=standard research, "deep"=deep research
        """
        
        depth_config = {
            "quick": {"max_searches": 2, "sections": 3},
            "standard": {"max_searches": 4, "sections": 5},
            "deep": {"max_searches": 8, "sections": 7}
        }
        config = depth_config.get(depth, depth_config["standard"])
        
        print(f"\nResearch started: {topic}")
        print(f"Research depth: {depth}\n")
        
        # ===== Stage 1: Plan Research =====
        print("Stage 1: Formulating research plan...")
        plan_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a research analyst. Develop a research plan for the following topic:

Topic: {topic}
Research Objective: Comprehensively understand this topic and generate a report with {config['sections']} core sections

Please generate the research plan in JSON format:
{{
  "research_questions": ["core question 1", "core question 2", ...],
  "search_queries": ["search query 1", "search query 2", ... (max {config['max_searches']})],
  "report_outline": ["Section 1 Title", "Section 2 Title", ...]
}}"""
                }
            ],
            response_format={"type": "json_object"}
        )
        
        plan = json.loads(plan_response.choices[0].message.content)
        search_queries = plan.get("search_queries", [topic])[:config["max_searches"]]
        report_outline = plan.get("report_outline", [f"{topic} Overview"])
        
        print(f"  Search plan: {len(search_queries)} queries")
        print(f"  Report structure: {len(report_outline)} sections")
        
        # ===== Stage 2: Search for Information =====
        print("\nStage 2: Searching for information...")
        all_findings = []
        
        for i, query in enumerate(search_queries, 1):
            print(f"  Search [{i}/{len(search_queries)}]: {query}")
            result = self._search(query)
            
            self._take_notes(result, source=f"Search: {query}")
            all_findings.append(f"[Query: {query}]\n{result}")
        
        findings_text = "\n\n".join(all_findings)
        
        # ===== Stage 3: Analyze and Synthesize =====
        print("\nStage 3: Analyzing and synthesizing...")
        
        analysis_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": f"""Based on the following research materials, conduct a deep analysis of the topic "{topic}".

Research Materials:
{findings_text[:4000]}

Report Outline: {report_outline}

Please generate a complete research report following the outline, with requirements:
1. Each section should have substantive content (200-400 words)
2. Include specific data, cases, or viewpoints
3. Provide conclusions and recommendations at the end
4. Use Markdown format"""
                }
            ]
        )
        
        report = analysis_response.choices[0].message.content
        
        # ===== Stage 4: Quality Check =====
        print("\nStage 4: Quality check...")
        
        review_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"""Briefly evaluate the quality of the following research report (JSON format):

Topic: {topic}
Report (first 1000 chars): {report[:1000]}

Evaluation:
{{
  "completeness_score": 1-10,
  "accuracy_indicators": "high/medium/low",
  "missing_aspects": ["missing point 1"],
  "overall_quality": "excellent/good/fair"
}}"""
                }
            ],
            response_format={"type": "json_object"}
        )
        
        review = json.loads(review_response.choices[0].message.content)
        
        # Generate final report
        final_report = f"""# Research Report: {topic}

> Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
> Research Depth: {depth}
> Quality Score: {review.get('completeness_score', 'N/A')}/10
> Information Sources: {len(self.research_notes)} entries

---

{report}

---

## Research Notes

- This report is based on {len(search_queries)} web searches
- Information cutoff date: {datetime.datetime.now().strftime('%Y-%m-%d')}
- It is recommended to verify against the latest materials
"""
        
        print(f"\nReport generation complete!")
        print(f"Quality: {review.get('overall_quality', 'N/A')} | "
              f"Completeness: {review.get('completeness_score', 'N/A')}/10")
        
        return final_report


# Usage Example
assistant = ResearchAssistant()

report = assistant.research(
    topic="Applications of Large Language Models in Software Development",
    depth="standard"
)

# Save report
filename = f"research_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md"
with open(filename, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\nReport saved to: {filename}")
```

## Key Code Analysis

Although this research assistant has relatively short code, it already possesses the prototype of a Deep Research Agent:

- **Planning Stage**: First generates `research_questions` and `search_queries` to avoid getting lost while searching.
- **Information Collection Stage**: Each search is written to `research_notes`, forming a traceable research trajectory.
- **Synthesis Stage**: Not simply splicing together search results, but reorganizing information according to `report_outline`.
- **Quality Check Stage**: Introduces a second model call to evaluate coverage, simulating the researcher's self-review process.

To upgrade it to a production-grade Deep Research Agent, it is recommended to add four modules:

| Module | Purpose | Implementation Highlights |
|--------|---------|--------------------------|
| **Evidence Cards** | Save facts, source URLs, publication dates, credibility | Every conclusion can be traced back to its source |
| **Reverse Search** | Proactively find counterexamples and different viewpoints | Avoid only adopting supporting evidence |
| **Citation Check** | Verify whether citations in the report genuinely support conclusions | Prevent "citation hallucinations" |
| **Research State Machine** | Control research stage transitions | Prevent infinite searching or premature summarization |

The key to a Deep Research Agent is not "searching more," but **ensuring every search serves a specific research gap**. This is also a classic landing point for long-term planning capability in real Agent applications.

## Running the Research Assistant

```bash
pip install openai python-dotenv requests rich
python research_agent.py
```

Example output:
```markdown
# Research Report: Applications of Large Language Models in Software Development

> Generated at: 2024-03-15 14:30
> Research Depth: standard
> Quality Score: 8/10

## 1. Overview
...

## 2. Code Generation and Completion
...

## 3. Code Review and Bug Detection
...
```

## References

1. Nakano et al. [**WebGPT: Browser-assisted question-answering with human feedback**](https://arxiv.org/abs/2112.09332). OpenAI, 2021.
2. Shao et al. [**Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models**](https://arxiv.org/abs/2402.14207). Stanford OVAL, 2024.
3. Chen et al. [**MindSearch: Mimicking Human Minds Elicits Deep AI Searcher**](https://arxiv.org/abs/2407.20183). 2024.
4. Li et al. [**WebSailor: Navigating Super-human Reasoning for Web Agent**](https://arxiv.org/abs/2507.02592). Alibaba Tongyi Lab, 2025.
5. Lu et al. [**The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery**](https://arxiv.org/abs/2408.06292). Sakana AI, 2024.
6. OpenAI. [**Introducing Deep Research**](https://openai.com/index/introducing-deep-research/). 2025.
7. OpenAI. [**BrowseComp in simple-evals**](https://github.com/openai/simple-evals). 2025.
8. Li et al. [**BrowseComp-ZH: Benchmarking Web Browsing Ability of Large Language Models in Chinese**](https://arxiv.org/abs/2504.19314). 2025.
9. LangChain AI. [**Open Deep Research**](https://github.com/langchain-ai/open_deep_research). 2025.
10. Alibaba-NLP. [**Tongyi DeepResearch**](https://github.com/Alibaba-NLP/DeepResearch). 2025.
11. Alibaba-NLP. [**Tongyi-DeepResearch-30B-A3B**](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B). 2025.
12. Texttron. [**BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent**](https://github.com/texttron/BrowseComp-Plus). 2026.

## Summary

This section completed an automated research assistant and demonstrated the foundational architecture of a Deep Research Agent:

- Research plan generation: decomposing an open-ended topic into questions and search queries
- Multi-round information collection: recording search results and sources
- Analysis and synthesis: generating structured reports following an outline
- Quality check: self-reviewing completeness and missing aspects
- Extensible directions: evidence cards, reverse search, citation check, research state machine

A true Deep Research Agent is a composite of "long-term planning + Web/document tools + evidence governance + quality evaluation." It will continue to unfold in subsequent chapters on Web Agents, context engineering, evaluation, and safety.

---

*Next Section: [5.6 Plan-and-Execute and Test-time Compute Scaling](./07_plan_and_execute.md)*
