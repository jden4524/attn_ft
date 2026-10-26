"""Search vast.ai offers and drop the ones that tend to waste money.

Filters (all from the public offer fields):
  verified host, VM not deverified, reliability >= 0.98;
  dlperf >= 90% of the median for that GPU model (broken or throttled cards score far lower);
  power limit >= 90% of the model's rated board power (power-capped cards);
  measured download >= 500 Mbps; traffic <= $0.005/GB down and $0.01/GB up;
  disk >= 100 GB, enough CPU cores and RAM for the job, rentable for >= 1 day.
Offers are ranked by hourly price plus the traffic cost of the expected download.

Usage: python scripts/vast/offers.py "RTX PRO 6000 WS" [--top 10]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import urllib.parse

API = "https://console.vast.ai/api/v0/bundles/?q="
DOWNLOAD_GB = 20.0  # docker layers + pip + model + datasets, rough upper bound
RATED_POWER_W = {  # reference board power per model
    "RTX PRO 6000 WS": 600, "RTX PRO 6000 S": 600, "RTX 5090": 575, "RTX 4090": 450, "RTX 3090": 350,
    "RTX 6000Ada": 300, "L40S": 350, "A100 SXM4": 400, "A100 PCIE": 250, "H100 SXM": 700, "H100 NVL": 400,
}


def fetch(gpu_name: str) -> list[dict]:
    # coarse filters server-side so the 64-offer page is not filled with hosts we reject anyway
    query = {"rentable": {"eq": True}, "num_gpus": {"eq": 1}, "gpu_name": {"eq": gpu_name},
             "verified": {"eq": True}, "reliability2": {"gte": 0.98}, "inet_down": {"gte": 500},
             "type": "on-demand", "order": [["dph_total", "asc"]]}
    out = subprocess.run(["curl", "-sS", "-m", "60", API + urllib.parse.quote(json.dumps(query))],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["offers"]


def filter_offers(offers: list[dict], min_cores: int = 8, min_ram_gb: int = 64) -> tuple[list[dict], dict[str, int]]:
    if not offers:
        return [], {}
    dl_median = statistics.median(o["dlperf"] for o in offers if o.get("dlperf"))
    rated = RATED_POWER_W.get(offers[0]["gpu_name"]) or statistics.median(o.get("gpu_max_power") or 0 for o in offers)
    checks = {
        "verified": lambda o: o.get("verification") == "verified" and not o.get("is_vm_deverified"),
        "reliability": lambda o: (o.get("reliability2") or 0) >= 0.98,
        "dlperf": lambda o: (o.get("dlperf") or 0) >= 0.9 * dl_median,
        "power_limit": lambda o: (o.get("gpu_max_power") or 0) >= 0.9 * rated,
        "bandwidth": lambda o: (o.get("inet_down") or 0) >= 500,
        "traffic_cost": lambda o: (o.get("inet_down_cost") or 1) <= 0.005 and (o.get("inet_up_cost") or 1) <= 0.01,
        "disk": lambda o: (o.get("disk_space") or 0) >= 100,
        "cpu": lambda o: (o.get("cpu_cores_effective") or 0) >= min_cores and (o.get("cpu_ram") or 0) >= min_ram_gb * 1000,
        "duration": lambda o: (o.get("duration") or 0) >= 86_400,
    }
    rejected = {name: 0 for name in checks}
    kept = []
    for o in offers:
        failed = [name for name, ok in checks.items() if not ok(o)]
        for name in failed:
            rejected[name] += 1
        if not failed:
            o["effective_dph"] = o["dph_total"]
            o["traffic_usd"] = DOWNLOAD_GB * (o.get("inet_down_cost") or 0)
            kept.append(o)
    kept.sort(key=lambda o: o["dph_total"] + o["traffic_usd"] / 3.0)  # traffic amortised over ~3 h
    return kept, rejected


def summary(o: dict) -> str:
    return (f"offer {o['id']} machine {o['machine_id']} ${o['dph_total']:.3f}/h  dlperf {o['dlperf']:.0f}  "
            f"power {o.get('gpu_max_power')}W  down {o.get('inet_down', 0):.0f}Mbps  "
            f"traffic ${o.get('inet_down_cost', 0):.3f}/GB  {o.get('geolocation')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu_name")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--min-cores", type=int, default=8)
    parser.add_argument("--min-ram-gb", type=int, default=64)
    args = parser.parse_args()
    offers = fetch(args.gpu_name)
    kept, rejected = filter_offers(offers, args.min_cores, args.min_ram_gb)
    print(f"{len(offers)} offers, {len(kept)} pass; rejections per check: {rejected}")
    for o in kept[: args.top]:
        print(summary(o))


if __name__ == "__main__":
    main()
