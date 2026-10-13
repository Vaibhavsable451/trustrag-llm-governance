"""Run the test set through the pipeline and report governance/hallucination stats."""
import json
from app import graph

tests = json.load(open("eval/testset.json"))
rows = []
for t in tests:
    r = graph.run(t["question"])
    ok = (r["verdict"] == "grounded") if t["answerable"] else (r["verdict"] in {"blocked", "no_context", "ungrounded"})
    rows.append({"q": t["question"], "verdict": r["verdict"], "groundedness": r["groundedness"], "pass": ok})
    print(f"{'PASS' if ok else 'FAIL'} | {r['verdict']:<10} | {r['groundedness']} | {t['question']}")
print(f"\nAccuracy: {sum(r['pass'] for r in rows)}/{len(rows)}")
