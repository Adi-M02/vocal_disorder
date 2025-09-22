import json

def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def expansion_to_base_json(path: str, outdir: str) -> dict:
    data = load_json(path)
    unique_terms = set()
    for terms in data.values():
        unique_terms.update(terms)
    result = {"seed_terms": sorted(unique_terms)}
    output_path = f"{outdir}/output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result 