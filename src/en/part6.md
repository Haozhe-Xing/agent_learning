# Part VI: Capstone Projects

Choose one domain and combine task contracts, tools, state, evaluation and permissions. These are parallel delivery branches, not three features every assistant must acquire.

| Chapter | Principle transferred | Artifact | Minimum failure case |
|---|---|---|---|
| [21 Coding Assistant](./chapter_23_coding_agent/README.md) | Actions are traceable and acceptance is independent | Reviewable patch, test command and results | Deleting tests cannot make a repair successful; failed changes can be reverted |
| [22 Data Analysis](./chapter_24_data_agent/README.md) | Evidence and calculations are traceable; tools have least privilege | Read-only query, source reconciliation and report definitions | Writes are denied; empty data does not become a confident conclusion |
| [23 Multimodal Agent](./chapter_25_multimodal/README.md) | Evidence has a location and low-quality input has a refusal path | Frame/region/time references, outputs and acceptance records | Unreadable or inaudible input is acknowledged, not invented |

The chapters provide component designs and assembly exercises. Domain tools, indexes, data connections and models require implementation and integration. The small `reference-agent/` baseline is not a complete implementation of these domains. Write the acceptance card before assembling components and judge delivery from actual evidence.

A passing domain project completes the application route. Continue to [Part VII](./part7.md) only if error analysis motivates training or automatic optimization; independent held-out evaluation must establish improvement.
