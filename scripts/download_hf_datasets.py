from pathlib import Path
from huggingface_hub import snapshot_download

base = Path(r"D:\workspace\memory\datasets\hf")
base.mkdir(parents=True, exist_ok=True)

repos = {
    "longmemeval": "xiaowu0162/longmemeval-cleaned",
    "clbench": "tencent/CL-bench",
    "personamem-v2": "bowen-upenn/PersonaMem-v2",
}

for name, repo_id in repos.items():
    target = base / name
    print(f"DOWNLOAD {name} {repo_id}", flush=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=target,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        size = sum(item.stat().st_size for item in target.rglob("*") if item.is_file())
        print(f"DONE {name} size_mb={round(size / 1024 / 1024, 2)}", flush=True)
    except Exception as exc:
        print(f"FAILED {name}: {type(exc).__name__}: {exc}", flush=True)

print("ALL_DONE", flush=True)
