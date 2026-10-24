# 向量检索机制练习 / Vector Retrieval Lab

对应两语6.3–6.4。一个本地Chroma集合、三份虚构文本、同一批稳定ID，依次观察向量距离、BM25、RRF与重排提议。实现与fixture在本目录，回归测试在仓库的`examples/integration_tests/test_vector_retrieval.py`；不生成或验收问答报告。

在仓库根目录、Python 3.11+环境中运行，依赖锁定在`examples/requirements.txt`（实际验证ChromaDB 1.5.9）：

```bash
python -m reference_code.vector_retrieval
python -m pytest examples/integration_tests/test_vector_retrieval.py -q
```

本目录也可单独复制，从目录内运行`python __main__.py`；需要先安装同版本ChromaDB。默认fixture写入临时目录，结束后删除；没有模型下载或服务商调用。测试运行真实数据库，模型相关机制使用明确的固定向量和预设评分。

| 阅读顺序 | 内容 |
|---|---|
| `fixture.py` | 404/500/预算三条来源，故意产生错误的向量首位候选 |
| `store.py` | 明确cosine、嵌入修订与维度、唯一ID、有限非零向量、空库、重开、删除 |
| `retrieval.py` | ASCII/CJK二元组教学分词、BM25、同ID排名融合、完整评分提议校验 |
| `__main__.py` | 输出各阶段ID，保持`answer_accepted: false` |

`VectorStore`是最多2000块、每批100块、单写入者的教学存储。来源过滤不是租户授权；版本标签由调用方声明；同维度不等于同嵌入空间。多写入者、跨调用事务、迁移/回滚、物理擦除、线上负载与模型质量未实现或验收。更换嵌入器/预处理需新索引并回归。

入库/查询向量在有限float64检查后，先按最大绝对分量缩放再做L2归一化；集合记录`normalization=stable_l2_v1`。这样避免极小/极大值直接转float32而损坏方向。仍有浮点误差，归一化不产生语义质量保证。

本例BM25以`k1=1.2,b=0.75`和正值IDF形式实现，查询词去重；分词器不是通用中文分词器。RRF的`k=60`、候选数量及重排阈值都是练习配置，不是最优参数或相关概率。评分器必须覆盖全部候选，单候选也不能跳过；格式正确不证明评分正确。

扩展练习：用独立问题集比较向量/BM25/融合，随后注入一个按输入顺序返回的真实嵌入器。记录来源版本、模型修订、召回、输入规模与答案验收；不要直接与固定fixture的顺序比较模型质量。

## English

This self-contained lab supports Sections 6.3–6.4. Real Chroma stores three fictional records with manually assigned vectors that deliberately rank a wrong source first. BM25 and RRF preserve shared record IDs; validated score proposals may still be wrong. No answer is generated or accepted.

Run the commands above from the repository, or copy this directory and run `python __main__.py` with ChromaDB 1.5.9 installed. Tests remain in the repository path listed above. The default uses a temporary collection, no model downloads, and no provider calls.

The store supports one writer, up to 2000 records, and batches up to 100. It checks declared embedding provenance, dimensions, finite nonzero vectors, and duplicate IDs. Source filters are not authentication. Model quality, production concurrency/migration, multi-call transactions, and physical erasure are outside the implemented scope. A new embedder or preprocessing pipeline requires a new index and evaluation.

Read `fixture.py`, `store.py`, `retrieval.py`, then `__main__.py`. Treat BM25/tokenizer choices, RRF constants, and reranking thresholds as teaching settings. Complete proposal validation does not establish semantic relevance or answer support.
