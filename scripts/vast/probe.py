"""Check a freshly rented machine before running a job on it.

Measures GPU name and power limit, bf16 matmul throughput, download speed from the Hugging Face
hub (by fetching the base model, which the job needs anyway) and disk write speed. Prints
`PROBE_RESULT <json>` and then `PROBE_PASS` or `PROBE_FAIL <reasons>`; exits 0 only on pass.

Thresholds come from the environment: PROBE_MIN_TFLOPS, PROBE_MIN_POWER_W, PROBE_MIN_DOWNLOAD_MBS,
PROBE_MIN_DISK_MBS, PROBE_GPU_SUBSTR (expected substring of the GPU name).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time


def gpu_info() -> dict:
    fields = "name,power.limit,power.max_limit,memory.total,clocks.max.sm,temperature.gpu"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    name, limit, max_limit, mem, clock, temp = [x.strip() for x in out.split(",")]
    return {"gpu": name, "power_limit_w": float(limit), "power_max_w": float(max_limit),
            "memory_mib": float(mem), "max_sm_clock_mhz": float(clock), "temp_c": float(temp)}


def matmul_tflops(seconds: float = 15.0) -> float:
    import torch

    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    n, start = 0, time.time()
    while time.time() - start < seconds:
        for _ in range(10):
            a @ b
        n += 10
        torch.cuda.synchronize()
    return 2 * 8192**3 * n / (time.time() - start) / 1e12


def download_mbs(repo: str = "Qwen/Qwen3-VL-2B-Instruct") -> tuple[float, float]:
    from huggingface_hub import snapshot_download

    cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    start = time.time()
    path = snapshot_download(repo)
    elapsed = time.time() - start
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(os.path.realpath(path)) for f in fs)
    blobs = os.path.join(cache, "hub")
    size = max(size, sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(blobs) for f in fs))
    return size / 1e6 / max(elapsed, 1e-6), size / 1e9


def disk_mbs(path: str = "/root/probe.bin", size_mb: int = 2048) -> float:
    start = time.time()
    subprocess.run(["dd", "if=/dev/zero", f"of={path}", "bs=1M", f"count={size_mb}", "oflag=direct"],
                   capture_output=True, check=True)
    elapsed = time.time() - start
    os.remove(path)
    return size_mb / elapsed


def main() -> None:
    result, reasons = {}, []
    try:
        result.update(gpu_info())
        result["tflops_bf16"] = round(matmul_tflops(), 1)
        result["download_mbs"], result["download_gb"] = (round(v, 1) for v in download_mbs())
        result["disk_mbs"] = round(disk_mbs(), 1)
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"error: {exc!r}")

    expected = os.environ.get("PROBE_GPU_SUBSTR")
    if expected and expected.lower() not in result.get("gpu", "").lower():
        reasons.append(f"gpu {result.get('gpu')!r} is not {expected!r}")
    limits = {"tflops_bf16": "PROBE_MIN_TFLOPS", "power_limit_w": "PROBE_MIN_POWER_W",
              "download_mbs": "PROBE_MIN_DOWNLOAD_MBS", "disk_mbs": "PROBE_MIN_DISK_MBS"}
    for key, env in limits.items():
        floor = os.environ.get(env)
        if floor is not None and result.get(key, 0) < float(floor):
            reasons.append(f"{key}={result.get(key)} < {floor}")

    print("PROBE_RESULT " + json.dumps(result), flush=True)
    if reasons:
        print("PROBE_FAIL " + "; ".join(reasons), flush=True)
        sys.exit(1)
    print("PROBE_PASS", flush=True)


if __name__ == "__main__":
    main()
