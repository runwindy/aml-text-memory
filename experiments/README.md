# 本地评测基线

## LoCoMo-Refined 基线

配置：

- dataset: `examples/locomo_refined_eval.jsonl`
- questions: `1382`
- top_k: `20`
- candidate_k: `10000`
- embedding: `hashing` 256 维
- organizer: `rule-based-v1`

指标：

| metric | value |
|---|---:|
| hit@1 | 0.1751 |
| hit@5 | 0.3698 |
| hit@10 | 0.4660 |
| hit@20 | 0.5803 |
| recall@1 | 0.1596 |
| recall@5 | 0.3348 |
| recall@10 | 0.4198 |
| recall@20 | 0.5221 |
| mrr | 0.2681 |
| forbidden@1 | 0.0000 |

完整基线保存在：

```text
experiments/locomo_baseline.json
```

## 每次改代码怎么对比

1. 启动本地服务：

```powershell
$env:AML_EMBEDDING_PROVIDER="hashing"
$env:AML_EMBEDDING_DIM="256"
$env:AML_DATABASE_PATH="D:\workspace\memory\amc\locomo_compare.db"
$env:AML_AUTH_MODE="none"
$env:AML_RETRIEVAL_CANDIDATE_K="10000"

uvicorn app.main:app --host 127.0.0.1 --port 8011
```

2. 跑评测并保存报告：

```powershell
python scripts\eval_retrieval.py `
  --base-url http://127.0.0.1:8011 `
  --dataset examples\locomo_refined_eval.jsonl `
  --top-k 20 `
  --output D:\workspace\memory\amc\locomo_current.json
```

3. 和基线对比：

```powershell
python scripts\compare_eval.py `
  --baseline experiments\locomo_baseline.json `
  --current D:\workspace\memory\amc\locomo_current.json
```

输出示例：

```text
metric             baseline    current      delta
--------------------------------------------------
hit@1                0.1751     0.1900    +0.0149
hit@5                0.3698     0.4100    +0.0402
mrr                  0.2681     0.2900    +0.0219
```

> 注意：这是本地公开数据回归基线，不是官方 Full 成绩。

## 第二轮本地基线：加入 Organizer / Governance / Reranker

配置：

- embedding: `hashing` 256 维
- candidate_k: `10000`
- top_k: `20`
- organizer: `rule-based-v1`
- governance: `valid_from / valid_to`
- reranker: `lexical`

结果：

| metric | baseline | current | delta |
|---|---:|---:|---:|
| hit@1 | 0.1751 | 0.2337 | +0.0586 |
| hit@5 | 0.3698 | 0.4045 | +0.0347 |
| hit@10 | 0.4660 | 0.5043 | +0.0384 |
| recall@1 | 0.1596 | 0.2137 | +0.0541 |
| recall@5 | 0.3348 | 0.3685 | +0.0338 |
| recall@10 | 0.4198 | 0.4525 | +0.0327 |
| mrr | 0.2681 | 0.3157 | +0.0476 |
| forbidden@1 | 0.0000 | 0.0000 | +0.0000 |

完整报告：

```text
experiments/locomo_after_upgrade.json
```

### 切换 text-embedding-v4

本地或服务器上设置：

```powershell
$env:AML_EMBEDDING_PROVIDER="openai"
$env:AML_EMBEDDING_MODEL="text-embedding-v4"
$env:AML_EMBEDDING_API_BASE="你的 OpenAI-compatible /v1 地址"
$env:AML_EMBEDDING_API_KEY="你的 key"
$env:AML_EMBEDDING_BATCH_SIZE="10"
$env:AML_RETRIEVAL_CANDIDATE_K="10000"
$env:AML_DATABASE_PATH="D:\workspace\memory\amc\locomo_v4.db"
$env:AML_AUTH_MODE="none"
uvicorn app.main:app --host 127.0.0.1 --port 8013
```

然后：

```powershell
python scripts\eval_retrieval.py `
  --base-url http://127.0.0.1:8013 `
  --dataset examples\locomo_refined_eval.jsonl `
  --top-k 20 `
  --output D:\workspace\memory\amc\locomo_v4_report.json

python scripts\compare_eval.py `
  --baseline experiments\locomo_after_upgrade.json `
  --current D:\workspace\memory\amc\locomo_v4_report.json
```

### 切换 LLM Organizer

```powershell
$env:AML_ORGANIZER_PROVIDER="openai"
$env:AML_ORGANIZER_MODEL="gpt-4o-mini"
$env:AML_ORGANIZER_API_BASE="你的 OpenAI-compatible /v1 地址"
$env:AML_ORGANIZER_API_KEY="你的 key"
```

开源/学术榜正式 Add/Search 使用 LLM 时，应遵守官方模型限制。
