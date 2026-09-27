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
   → Bronze raw
   → Normalizer
   → Deduplicator
   → TimeProcessor
   → SafetyProcessor
   → Silver messages
   → Extractor
   → Gold memory_items
   → Embedding
   → Index
```

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
- 时间戳只统一为 UTC，不把相对时间破坏性改写
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
