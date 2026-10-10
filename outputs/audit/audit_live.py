"""Audit the running local API using an existing test image and its VOC boxes."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import statistics
import sys
import time
import urllib.request
import uuid

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LABELS = ["without_mask", "with_mask", "mask_weared_incorrect"]
manifest = json.loads((ROOT / "data/processed/manifest.json").read_text(encoding="utf-8"))
hash_splits = defaultdict(set)
sources = defaultdict(list)
for item in manifest:
    hash_splits[item["source_hash"]].add(item["split"])
    if item["split"] == "test":
        sources[item["source"]].append(item)
leakage = [key for key, splits in hash_splits.items() if len(splits) > 1]
source = next(name for name in sorted(sources) if set(LABELS) <= {x["label"] for x in sources[name]})
selected = [next(x for x in sources[source] if x["label"] == label) for label in LABELS]
image = ROOT / "data/raw/images" / source
faces = []
for i, item in enumerate(selected):
    x1, y1, x2, y2 = item["box"]
    faces.append({"id": f"audit-{i}", "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1})

def predict(boxes):
    boundary = "Audit" + uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="faces"\r\n\r\n'.encode()
        + json.dumps(boxes).encode() + b"\r\n"
        + f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{image.name}"\r\nContent-Type: image/png\r\n\r\n'.encode()
        + image.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    )
    request = urllib.request.Request("http://127.0.0.1:5500/api/v1/predict", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=15) as response:
        result = json.load(response)
    elapsed = (time.perf_counter() - start) * 1000
    assert [p["id"] for p in result["predictions"]] == [f["id"] for f in boxes]
    assert all(p["label"] in LABELS and 0 <= p["confidence"] <= 1 for p in result["predictions"])
    return result, elapsed

result, elapsed = predict(faces)
comparison = [{"expected": label, "predicted": p["label"], "confidence": p["confidence"],
               "correct": label == p["label"]}
              for label, p in zip(LABELS, result["predictions"])]
samples = []
for count in (1, 2, 3):
    for iteration in range(5):
        response, rtt = predict(faces[:count])
        samples.append({"faces": count, "iteration": iteration + 1,
                        "rtt_ms": round(rtt, 3), "inference_ms": response["total_inference_ms"]})
with (OUT / "local_benchmark.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(samples[0]))
    writer.writeheader()
    writer.writerows(samples)
summary = []
for count in (1, 2, 3):
    rows = [s for s in samples if s["faces"] == count]
    summary.append({"faces": count, "requests": len(rows),
                    "rtt_mean_ms": round(statistics.mean(s["rtt_ms"] for s in rows), 3),
                    "inference_mean_ms": round(statistics.mean(s["inference_ms"] for s in rows), 3)})
audit = {"scope": "Local laptop only; VOC boxes supplied, not browser face detection; 5 requests per face count",
         "source": source, "ground_truth": comparison, "response": result,
         "source_hashes_crossing_splits": len(leakage), "latency": summary}
(OUT / "local_api_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in audit.items() if k != "response"}, indent=2, ensure_ascii=False))
