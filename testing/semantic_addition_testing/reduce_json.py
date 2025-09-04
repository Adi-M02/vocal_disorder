import json
from pathlib import Path

IN_PATH = Path("testing/semantic_addition_testing/compare_09_03_23_03/per_seed_diffs.json")
OUT_PATH = Path("testing/semantic_addition_testing/compare_09_03_23_03/only_full.json")

# If True, entries without a base_terms key will also be dropped.
DROP_IF_MISSING = False

data = json.loads(IN_PATH.read_text())

def keep_entry(entry: dict) -> bool:
    if "base_terms" not in entry:
        return not DROP_IF_MISSING  # keep if missing, unless you choose to drop
    base = entry["base_terms"]
    # keep only if base_terms is a non-empty list
    return isinstance(base, list) and len(base) > 0

filtered = {k: v for k, v in data.items() if keep_entry(v)}

OUT_PATH.write_text(json.dumps(filtered, indent=2, ensure_ascii=False))
print(f"Kept {len(filtered)} / {len(data)} entries (removed {len(data) - len(filtered)})")
