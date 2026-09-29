from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "BAAI/bge-reranker-v2-m3"
print("loading tokenizer", flush=True)
AutoTokenizer.from_pretrained(model_name)
print("loading model", flush=True)
AutoModelForSequenceClassification.from_pretrained(model_name)
print("prefetch complete", flush=True)
