# Chapter 6: Retrieval-Augmented Generation (RAG)

**Reading level**: Core; advanced RAG is optional.

## This chapter's iteration

- **Previous limitation**: Controlled steps may still use a distance recalled by the model. A search hit does not establish valid citations, and an empty answer may be mistaken for completed research.
- **This chapter adds**: Preserve retrieved documents, source labels, and quotes for the same task, and make extraction consume that evidence. Record retrieval, generation, and verification separately; `sources` contains only citations actually used by an accepted answer.
- **How to verify**: Run `python -m reference_code.evidence_assistant.demo --stage rag` and `python -m unittest reference_code.evidence_assistant.tests.test_answer_acceptance reference_code.evidence_assistant.tests.test_project_contracts`. These cover empty output, unknown citations, rejected acceptance, changed evidence, and conflicting documents. Deliver the evidence record, answer, and verdict rationale. Default QA checks citation labels only; semantic support requires an independent task verifier or human review.
- **Next step**: As documents and memories accumulate, the key distance may be crowded out. Budget the same evidence chain in [chapter 7](../chapter_context_engineering/README.md), before adding more elaborate retrieval backends.

RAG passes retrieved external material to an answerer. It can supply evidence in the model's input, but does not automatically establish source truth, answer correctness, or authorization. Start with small local tasks, then select vector and advanced retrieval as needed.

## Chapter Goals

After the core exercises, you should be able to:

- Separate candidate retrieval, answering, citation checks, source support, and requirement coverage.
- Load local UTF-8 text/Markdown, create exact character chunks, and retain source versions and offsets.
- Inspect distances, source filters, reopening, and deletion with real Chroma and fixed vectors.
- Compare BM25, same-ID RRF, and reranking proposals without treating scores as correctness probabilities.
- Run lexical QA with independent excerpt/coverage checks and preserve normal, unknown, and failure states.

## Chapter Structure

| Section | Content | Difficulty |
|---|---|---|
| 6.1 RAG Concepts | Separate retrieval, answering, and acceptance | ⭐⭐ |
| 6.2 Document Loading and Splitting | Local txt/md, exact character chunks, source positions | ⭐⭐ |
| 6.3 Embeddings and Vector Databases | Explicit embedding contracts, real Chroma, distances | ⭐⭐⭐ |
| 6.4 Retrieval Strategies and Reranking | BM25/RRF/score proposals and counterexamples | ⭐⭐⭐ |
| 6.5 Document QA Practice | Lexical QA, source excerpts, independent coverage | ⭐⭐⭐ |
| 6.6 Paper Readings | Self-RAG/CRAG/GraphRAG and related research; optional | ⭐⭐⭐ |
| 6.7 Advanced RAG | Graph retrieval and agentic orchestration designs; optional | ⭐⭐⭐⭐⭐ |

## Reference Code and Completion

Loading/chunking mechanisms live in `reference_code/memory_rag/`, vector/hybrid retrieval in `reference_code/vector_retrieval/`, and local QA/research applications in `reference_code/evidence_assistant/`. These exercises have different responsibilities. No adapter currently connects vector candidates directly to QA; 6.5 defines the source mapping, version, hit-snapshot, and budget contracts for that extension.

Citation labels do not establish support, and supported excerpts may omit required points. Deliver independent expectations, observed sources, stage states, and a final verdict. Passing the local exercises does not accept real semantic models, PDF/Word/Web loaders, online services, GraphRAG, or a production agentic system. You need not install every advanced dependency first.

## Learning Path

Prerequisites: [Chapter 4: Memory](../chapter_memory/README.md) and Python file/list/dictionary operations. Core practice needs no model key or OpenAI embedding API experience. Chroma exercises use the pinned dependencies.

Continue to [Chapter 7: Context Engineering](../chapter_context_engineering/README.md) to budget retrieved evidence. Read [Chapter 11: LangGraph](../chapter_langgraph/README.md) when state orchestration warrants it.

*Next: [6.1 RAG Concepts](01_rag_concepts.md)*
