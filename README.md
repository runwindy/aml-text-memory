# AML Text Memory

一个面向 **Agent Memory Challenge 文本赛道 / 开源方法榜** 的可扩展基座框架。

核心原则来自官方赛制：

> 参赛方只负责 `Add` 和 `Search`；平台统一完成 `Answer`、`Eval`、复核和公榜。
> `Search` 不能返回最终答案，只能返回按相关性排序的记忆证据。

当前版本提供：

- 严格对齐官方字段的 `Add` / `Search` / `Health` FastAPI 接口
- SQLite 持久化 + `request_id` 幂等
- `user_id` 严格检索隔离
- 默认离线 `hashing` embedding，方便本地跑通
- 可切换到 OpenAI-compatible `text-embedding-v4`
- Dense + BM25 混合检索
- `Connector` 风格的扩展点：Extractor / Embedding / Store / Retriever / Reranker / Packer
- Docker 部署和基础契约测试

> 这是基座，不是最终高分系统。下一步应根据官方公开 pipeline 和数据能力，加入原子事实抽取、时间/冲突治理、实体图和多跳检索。

---

## 1. 快速开始

### 本地运行

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 本地 Smoke

```bash
python scripts/local_smoke.py --base-url http://127.0.0.1:8000
```

### 测试

```bash
pytest -q
```

### Docker

```bash
docker compose up --build
```

---

## 2. 官方接口契约

### Health

```http
GET /health
```

无需鉴权，返回任意 2xx 即为健康。

### Add

```http
POST /add
```

```json
{
  "request_id": "eval:run_abc123:locomo_refined:conv-0:chunk-0",
  "messages": [
    {
      "role": "user",
      "timestamp": 1704067200000,
      "content": "memory text"
    }
  ],
  "user_id": "eval:run_abc123:locomo:conv-0",
  "session_id": "eval:run_abc123:sample:0"
}
```

成功响应：

```json
{
  "success": true,
  "request_id": "eval:run_abc123:locomo_refined:conv-0:chunk-0",
  "user_id": "eval:run_abc123:locomo:conv-0",
  "session_id": "eval:run_abc123:sample:0"
}
```

要求：

- 同步写入：持久化且可立即检索后才能返回 `HTTP 200`。
- 重试时必须使用同一个 `request_id`，并保证幂等。
- `user_id` 是唯一检索隔离范围。
- `session_id` 只用于组织来源，不参与 Search 过滤。

### Search

```http
POST /search
```

```json
{
  "query": "Which answer best matches the memory?",
  "options": ["A. First answer", "B. Second answer"],
  "user_id": "eval:run_abc123:locomo:conv-0",
  "top_k": 100
}
```

成功响应：

```json
{
  "data": [
    {
      "id": "mem_1",
      "content": "remembered fact text",
      "score": 0.87,
      "created_at": "2026-07-01T12:00:00Z"
    }
  ]
}
```

要求：

- 必须是对象，且包含 `data` 数组。
- 按相关性从高到低排序。
- 没有结果返回 `{"data": []}`。
- 返回数量不能超过 `top_k`。
- `content` 会按顺序进入平台统一 Answer。
- 不能把最终答案直接写进 `content`。

---

## 3. 架构

```text
Platform
  |-- POST /add
  |-- POST /search
  |
FastAPI app
  |-- auth.py            鉴权
  |-- schemas.py         官方契约模型
  |
  |-- AddService
  |     extractor -> embedding -> SQLiteMemoryStore
  |
  |-- SearchService
        query_analyzer
        HybridRetriever (Dense + BM25 + RRF)
        Reranker
        EvidencePacker
```

目录：

```text
app/
├── api/routes.py              # Add / Search / Health
├── core/
│   ├── auth.py                # Token / Bearer / X-Api-Key
│   ├── errors.py
│   └── logging.py
├── memory/
│   ├── models.py              # MemoryRecord
│   └── extractor.py           # PassThroughExtractor，可替换
├── retrieval/
│   ├── embedding.py           # hashing / OpenAI-compatible
│   ├── query_analyzer.py      # 查询分类
│   ├── hybrid.py              # Dense + BM25 + RRF
│   ├── reranker.py            # IdentityReranker，可替换
│   └── packer.py              # 转成官方 Search 响应
├── services/
│   ├── add_service.py
│   ├── search_service.py
│   └── container.py
├── storage/
│   ├── base.py                # MemoryStore 协议
│   └── sqlite.py              # SQLite baseline
├── config.py
├── schemas.py
└── main.py
```

---

## 4. 配置

所有配置使用 `AML_` 前缀。

```env
AML_AUTH_MODE=none
AML_MEMORY_SYSTEM_KEY=change-me
AML_DATABASE_PATH=./data/aml.db

AML_EMBEDDING_PROVIDER=hashing
AML_EMBEDDING_MODEL=text-embedding-v4
AML_EMBEDDING_API_BASE=
AML_EMBEDDING_API_KEY=
AML_EMBEDDING_DIM=256

AML_RETRIEVAL_CANDIDATE_K=500
AML_MAX_RETURN_ITEMS=100
AML_MAX_CONTENT_CHARS=4000
```

### 切换到 text-embedding-v4

```env
AML_EMBEDDING_PROVIDER=openai
AML_EMBEDDING_MODEL=text-embedding-v4
AML_EMBEDDING_API_BASE=https://your-openai-compatible-endpoint/v1
AML_EMBEDDING_API_KEY=...
```

注意：开源/学术榜的模型限制以官方审核反馈为准。官方 FAQ 提到 embedding 使用 `text-embedding-v4`，LLM 组件使用 `gpt-4o-mini`。

---

## 5. 当前扩展点

### Extractor

当前：`PassThroughExtractor`，每条消息转成一条 `MemoryRecord`。

建议升级为：

- 原子事实
- 事件与时间
- 偏好/角色
- 关系与实体
- 会话摘要
- 冲突更新

### Embedding

当前：`HashingEmbeddingProvider`，离线、确定性，只用于跑通链路。

生产建议：

- `text-embedding-v4`
- 批量 embedding
- 失败重试和缓存

### Store

当前：`SQLiteMemoryStore`。

生产建议：

- PostgreSQL + pgvector
- Qdrant / Weaviate
- 原始 chunk、事实、事件、实体图分开存
- `user_id` 索引和过滤必须强制

### Retriever

当前：Dense + BM25 + RRF。

建议升级：

- Query 分类
- 多跳查询分解
- 实体图扩展
- 时间有效区间过滤
- 冲突消解
- 个性化检索

### Reranker

当前：`IdentityReranker`。

建议升级：

- `bge-reranker-v2-m3` 等跨编码器
- 细粒度事实 rerank

### Packer

当前：简单排序、去重、截断。

关键点：

- 平台 Answer 输入预算约 117,760 tokens。
- 超出后只保留你返回顺序的前缀。
- 所以必须保证前几条又短又准。

---

## 6. 竞赛实施建议

1. 先把 Add/Search/Health 跑通，再用公开 Smoke 验证契约。
2. 尽早跑第一次 Full。第二次 Full 要在第一次完成后 30 天才能发起。
3. 文本赛道优先做：事实抽取、时间治理、多跳关系、个性化、规则和安全拒答。
4. 10 月 31 日前提交完整材料。
5. 11 月 4 日评测停止，注意 Full 需要 0.5—2 天。

---

## 7. 重要边界

- Search 不得生成最终答案。
- 不得跨 `user_id` 检索。
- 不得硬编码、数据泄漏、提示词注入、人工答题。
- 评测数据只能用于本次评测，30 天内删除。
- 提交后 API 至少保持 30 天公网稳定可访问。
- Full 每赛道最多 2 次，配额非常宝贵。

---

## 8. 本地检索指标评测

新增了本地评测模块：

```text
app/eval/metrics.py      # Hit@k / Recall@k / MRR / nDCG / EM / Token-F1
app/eval/harness.py      # 添加会话 -> Search -> 计算指标
scripts/eval_retrieval.py
examples/sample_eval.jsonl
```

运行：

```bash
python scripts/eval_retrieval.py \
  --base-url http://127.0.0.1:8000 \
  --dataset examples/sample_eval.jsonl \
  --top-k 20
```

公网带鉴权：

```bash
python scripts/eval_retrieval.py \
  --base-url https://你的域名 \
  --key 你的MemorySystemKey \
  --dataset examples/sample_eval.jsonl
```

输出示例：

```json
{
  "hit@1": 0.666,
  "hit@5": 1.0,
  "recall@5": 0.888,
  "mrr": 0.75
}
```

数据集格式：

```json
{
  "id": "case-alice",
  "user_id": "eval-alice",
  "sessions": [
    {
      "session_id": "alice-s1",
      "messages": [
        {"role": "user", "content": "Alice lives in Shanghai and loves hiking."}
      ]
    }
  ],
  "question": "Where does Alice live and what does she like?",
  "expected_keywords": ["Shanghai", "hiking"]
}
```

用途：

- 比较 chunk 策略
- 比较 embedding 模型
- 比较 BM25 / dense / hybrid
- 比较 reranker
- 比较事实抽取前后
- 比较时间治理和冲突更新效果

### 挑战性评测集

额外提供：

```text
examples/challenge_eval.jsonl
```

覆盖：

- 当前状态 vs 历史信息
- 多跳关系
- 时间顺序
- 偏好与冲突
- 列表召回
- 否定条件

运行：

```powershell
python scripts\eval_retrieval.py --base-url http://127.0.0.1:8000 --dataset examples/challenge_eval.jsonl --top-k 20
```

输出中新增：

```text
forbidden@1
```

表示 Top-1 中是否误命中了禁止出现的旧信息或错误信息。

---

## 9. 企业级 Ingestion 流程

Add 链路已重构为：

```text
AddService
→ IngestionPipeline
   → request_id 原子占位（pending）
   → Bronze raw 先落盘
   → Normalizer
   → Deduplicator
   → TimeProcessor
   → SafetyProcessor
   → Silver messages
   → Extractor
   → Gold memory_items
   → Embedding
   → Silver/Gold 事务提交并标记 succeeded
```

失败时保持 Bronze 并标记 `failed`，同一 `request_id` 可重试；只有 Silver/Gold 已提交且可立即 Search 时才返回成功。

新增模块：

```text
app/ingestion/
├── models.py          # RawAddRequest / CanonicalMessage
├── normalizer.py      # 非破坏性文本规范化
├── deduplicator.py    # 精确去重标记
├── time_processor.py  # 时间粒度识别
├── safety.py          # PII / prompt injection 标记
├── validator.py       # 契约校验
└── pipeline.py        # Bronze -> Silver -> Gold 编排
```

当前 SQLite 表：

```text
add_requests        # 幂等与响应缓存
raw_add_requests    # Bronze 原始 Add 请求
messages            # Silver 规范消息
memory_items        # Gold 记忆单元 + embedding
```

关键原则：

- `raw_content` 原样保留，用于最终 Answer 证据
- `normalized_content` 用于检索、去重和 BM25
- `raw_content_hash` 与 `content_hash` 分离，清洗规则升级不会改变来源 `message_id`
- `message_id` 基于 `request_id + sequence_no`，不依赖规范化结果
- 时间信息拆分为 `timestamp_ms` / `time_granularity` / `time_mentions`，保留原文时间表达
- 重复消息保留在 Bronze/Silver；只有确认传输层重放时才由 `request_id` 幂等消除
- 同 request_id 不同 payload 返回 409 Conflict
- `user_id` 隔离贯穿所有存储和检索

---

## 10. 结构化 Gold / Index / Multi-Index Retrieval

已完成第一阶段到第五阶段的第一版：

```text
Stage 1 Gold 结构化记忆
  memory_items 增加 subject / predicate / object_value / entities /
  source_message_ids / valid_from / valid_to / confidence / importance / status

Stage 2 Memory Organizer
  app/organizer/rule_based.py
  app/organizer/composite.py
  抽取事实、偏好、画像字段、事件

Stage 3 Index Layer
  app/indexing/entity_index.py
  app/indexing/time_index.py
  app/indexing/type_index.py

Stage 4 Multi-Index Retrieval
  app/retrieval/multi_index.py
  Dense + BM25 + Entity + Time + Memory-Type + RRF

Stage 5 Profile / Event / Conflict
  memory_type = raw / fact / event / preference / profile / rule / summary
  app/retrieval/conflict.py 提供 current-state 冲突降权
```

当前是 deterministic 版本，还没有接入 LLM Organizer。启用 `gpt-4o-mini`
做结构化抽取时，必须遵守开源/学术榜模型限制。

## 11. 本地 DeepSeek 实验模式

DeepSeek V4.1 Flash 仅用于**本地预研和消融实验**，不适合直接用于开源/学术榜正式提交。开源/学术榜要求：

- Embedding：`text-embedding-v4`
- LLM 相关组件：`gpt-4o-mini`

DeepSeek 实验建议：

1. 保持 `text-embedding-v4`，保证向量与正式配置可比；
2. 使用新的 SQLite 数据库，例如 `longmemeval_s_deepseek.db`；
3. 只在本地使用 DeepSeek 做 Organizer / Decomposer；
4. 记录 `decomposer_provider` / `decomposer_model` / `organizer` 元数据，便于区分实验结果。

环境变量示例见：

```text
.env.deepseek.example
deploy/.env.deepseek.example
```

使用独立配置文件：

```powershell
Copy-Item .env.deepseek.example .env.deepseek
# 编辑 .env.deepseek，填入你的 embedding 和 DeepSeek 配置
$env:AML_ENV_FILE = "D:\workspace\memory\aml-text-memory\.env.deepseek"
uvicorn app.main:app --host 0.0.0.0 --port 8031
```

如果不设置 `AML_ENV_FILE`，程序仍然按原逻辑加载 `.env`。

关键配置：

```env
AML_EMBEDDING_PROVIDER=openai
AML_EMBEDDING_MODEL=text-embedding-v4

# 当 decomposer 开启时，organizer 不会被实际调用。
# 保持 rule 可以避免重复配置 LLM，也能避免误解为两个模型同时工作。
AML_ORGANIZER_PROVIDER=rule

AML_DECOMPOSER_PROVIDER=deepseek
AML_DECOMPOSER_MODEL=deepseek-v4.1-flash
AML_DECOMPOSER_API_BASE=https://your-deepseek-compatible/v1
AML_DECOMPOSER_API_KEY=replace-me

AML_DATABASE_PATH=D:\workspace\memory\amc\longmemeval_s_deepseek.db
```

当前代码会把 `deepseek`、`openai`、`openai-compatible` 都视为 OpenAI-compatible `/chat/completions` 提供方。

## 12. JEPA-inspired Predictive Memory Layer

项目加入了轻量版 JEPA 风格预测记忆层，作为本地实验的辅助信号，不替代
`text-embedding-v4`。

核心思想：

- `user_context_text` 作为上下文；
- `assistant_target_text` 作为观测到的未来目标；
- 计算上下文 latent 与 assistant latent 的距离；
- 距离作为 surprise / prediction error；
- 高 surprise 代表 assistant 可能引入更新、冲突或新信息；
- 结果写入 `memory_items.metadata_json`，Search 时对结果做小幅重排。

配置：

```env
AML_PREDICTIVE_MEMORY_ENABLED=true
AML_PREDICTION_SURPRISE_THRESHOLD=0.35
AML_PREDICTIVE_RERANK_WEIGHT=0.15
```

说明：

- 默认关闭，避免额外 embedding 成本；
- 开启后会对有 assistant 目标的记录额外调用一次 embedding；
- 主检索仍然是 text-embedding-v4 / BM25 / BGE；
- JEPA 层只负责 surprise、grounding status 和轻量 rerank bias。

## 13. Graph-JEPA 与 Dialogue-JEPA

在 Section 12 的 assistant-context surprise 之上，项目继续加入了两个可选阶段。

### Graph-JEPA 式链接预测

代码：

```text
app/retrieval/graph_jepa.py
```

作用：

- 在候选记忆之间预测隐式边；
- 分数由三部分组成：
  - latent embedding similarity
  - entity overlap
  - temporal adjacency
- 预测边会合并进 `graph_expander` 的 beam search；
- 让多 session / 跨 session 证据可以通过隐式图边连接，而不只依赖显式 relation。

配置：

```env
AML_GRAPH_LINK_PREDICTION_ENABLED=true
AML_GRAPH_LINK_PREDICTION_TOP_K=8
AML_GRAPH_LINK_PREDICTION_THRESHOLD=0.55
AML_GRAPH_LINK_PREDICTION_WEIGHT=0.5
AML_GRAPH_LINK_PREDICTION_MAX_NODES=200
```

### Dialogue-JEPA 预测器

代码：

```text
app/memory/dialogue_jepa.py
scripts/train_dialogue_jepa.py
```

训练样本：

```text
context = 当前 session 中 assistant 之前的对话
target  = assistant 回复的 latent embedding
```

训练：

```powershell
python scripts/train_dialogue_jepa.py `
  --dataset examples/longmemeval_s_eval.jsonl `
  --output data/dialogue_jepa.pt `
  --max-cases 50 `
  --epochs 20
```

默认使用 `hashing` embedding，所以本地训练不需要外部 API。

### JEPA latent query expansion

Search 时：

```text
query vector
  → DialogueJepaPredictor
  → predicted evidence latent
  → 和原 query 向量融合
  → dense retrieval
```

配置：

```env
AML_JEPA_QUERY_EXPANSION_ENABLED=true
AML_JEPA_MODEL_PATH=./data/dialogue_jepa.pt
AML_JEPA_BLEND_WEIGHT=0.5
```

如果模型文件不存在或未启用，系统会自动使用一个训练无关的
pseudo-relevance 回退：

- 先做一轮普通检索；
- 取 Top seeds 的 embedding 质心作为 predicted evidence latent；
- 与原 query vector 融合；
- 作为额外 dense 分支加入 RRF。

配置：

```env
AML_JEPA_QUERY_EXPANSION_FALLBACK=true
AML_JEPA_PSEUDO_RELEVANCE_SEED_K=20
AML_JEPA_PSEUDO_RELEVANCE_WEIGHT=0.5
```

这样即使没有训练数据，也可以使用 Phase 4 的 latent query expansion 思想。

## 14. 持久化会话树与知识图谱骨架

Add 事务现在会额外写入两个持久化结构。

### dialogue_nodes

```text
Session
  └── Turn
        ├── User message
        └── Assistant message
```

每个节点保存：

```text
node_id
user_id
session_id
request_id
node_type
role
sequence_no
turn_index
parent_id
prev_id
next_id
content
timestamp
metadata_json
```

用途：

- 保留会话内部层级；
- 区分 user / assistant；
- 支持父节点、兄弟节点、前后节点扩展；
- 为后续 tree-aware retrieval 提供基础。

### memory_edges

```text
edge_id
user_id
source_id
target_id
edge_type
confidence
valid_from
valid_to
evidence_ids_json
metadata_json
```

当前写入：

- Decomposer 抽出的 relation 记录；
- assistant responds_to 边；
- assistant confirms / supports 边。

Graph-JEPA 预测边仍然在 Search 时临时计算，不落库。

SQLite 表：

```text
dialogue_nodes
memory_edges
```

这样我们已经从运行时窗口 + 临时图扩展推进到：

```text
会话内部：持久化有序树骨架
会话之间：持久化图边骨架
Search 时：Graph-JEPA 预测边做动态补充
```

## 15. 会话树与知识图谱接入检索和更新闭环

### 检索闭环

Search 现在可以在候选记忆加载后读取：

```text
dialogue_nodes
memory_edges
```

新增模块：

```text
app/retrieval/dialogue_graph.py
    build_tree_graph_scores()
```

流程：

```text
初始多路召回
  → 取 Top seeds
  → 映射 source_message_ids 到 dialogue_nodes
  → 扩展父节点 / 兄弟节点 / prev / next
  → 沿 memory_edges 扩展 relation / responds_to / supports
  → 扩展 Graph-JEPA 预测边
  → RRF 融合
  → result window
  → rerank
```

配置：

```env
AML_DIALOGUE_TREE_EXPANSION_ENABLED=true
AML_DIALOGUE_TREE_WEIGHT=0.7
AML_DIALOGUE_TREE_DECAY=0.7
AML_DIALOGUE_TREE_MAX_HOPS=2
```

### 更新闭环

时间治理现在不仅更新：

```text
valid_from / valid_to
status = active / superseded
```

还会写入：

```text
memory_edges.edge_type = supersedes
```

例如：

```text
user live_in Beijing   -> superseded
user live_in Shanghai  -> active
edge: Shanghai --supersedes--> Beijing
```

同时 Add 事务会写入：

```text
dialogue_nodes      会话树
memory_edges        relation / responds_to / confirms / supports
```

这样树、图、Gold 记忆和检索链路已经形成闭环：

```text
Add -> 写树/图/记忆
Search -> 读树/图/记忆并扩展
Update -> 更新状态并写 supersedes 边
```

## 16. 实体图、持久化 Graph-JEPA 与异步 Worker

### entity_nodes / entity_edges

新增：

```text
app/memory/entities.py
```

Add 事务会构建：

```text
entity_nodes
entity_edges
memory_entity_links
```

当前实体消歧是确定性的轻量规则：

```text
lowercase
去冠词
去所有格
统一标点
按 canonical_name 合并
```

后续可以替换为 embedding-based entity resolution。

### 持久化 Graph-JEPA 预测边

Add 完成时会计算新记忆之间的预测边，并写入：

```text
memory_edges.edge_type = predicted_related
metadata_json.source = graph_jepa
```

Search 时仍然会动态补充预测边，但现在已经有持久化版本。

### 异步 Add 与 dirty set

新增：

```text
app/services/async_worker.py
```

配置：

```env
AML_ASYNC_PROCESSING_ENABLED=true
AML_ASYNC_WORKER_ENABLED=true
AML_ASYNC_WORKER_POLL_INTERVAL=1.0
```

流程：

```text
/add
  → 同步写 Bronze / Silver / raw memory_items
  → status = raw_ready
  → 返回 200

AsyncMemoryWorker
  → claim async_jobs
  → LLM 抽取 / embedding / tree / graph / entities
  → status = succeeded
  → complete job
```

新增 SQLite 表：

```text
async_jobs
dirty_entities
```

dirty set 会在新实体写入时产生，并在异步任务完成后标记 processed。

