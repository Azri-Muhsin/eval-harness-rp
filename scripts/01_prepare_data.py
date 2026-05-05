import argparse
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

SEED = 3407
random.seed(SEED)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_record(r):
    return {
        "language": str(r.get("language", "")).strip(),
        "intent": str(r.get("intent", "")).strip(),
        "user_query": str(r.get("user_query", "")).strip(),
        "keywords": r.get("keywords", []),
        "rationale": str(r.get("rationale", "")).strip(),
        "english_translation_or_summary": str(r.get("english_translation_or_summary", "")).strip(),
    }


def deduplicate(records):
    seen = set()
    kept = []
    duplicates = []

    for r in records:
        key = (r["language"].lower(), r["user_query"].strip().lower())
        if key in seen:
            duplicates.append(r)
            continue
        seen.add(key)
        kept.append(r)

    return kept, duplicates


def stratified_split(records, label_key="intent", val_ratio=0.10, test_ratio=0.10):
    by_label = defaultdict(list)
    for r in records:
        by_label[r[label_key]].append(r)

    train, val, test = [], [], []

    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)

        if n >= 10:
            n_test = max(1, int(n * test_ratio))
            n_val = max(1, int(n * val_ratio))
        else:
            n_test = 0
            n_val = 0

        test.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def label_sft(r):
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a healthcare intent classifier. Return only one intent label from the allowed label set.",
            },
            {
                "role": "user",
                "content": f"Query: {r['user_query']}",
            },
            {
                "role": "assistant",
                "content": r["intent"],
            },
        ],
        "language": r["language"],
        "intent": r["intent"],
        "user_query": r["user_query"],
    }


def reasoning_sft(r):
    rationale = r.get("rationale", "").strip()
    if not rationale:
        rationale = f"The query best matches the healthcare intent label {r['intent']}."

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a healthcare intent classifier. First give the final label, then give a brief reason.",
            },
            {
                "role": "user",
                "content": f"Query: {r['user_query']}",
            },
            {
                "role": "assistant",
                "content": f"Final label: {r['intent']}\nReason: {rationale}",
            },
        ],
        "language": r["language"],
        "intent": r["intent"],
        "user_query": r["user_query"],
    }


def sample_by_language(records, include_langs):
    include = {x.lower() for x in include_langs}
    return [r for r in records if r["language"].lower() in include]


def balanced_by_key(records, key):
    groups = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)

    min_count = min(len(v) for v in groups.values())
    balanced = []

    for _, items in groups.items():
        random.shuffle(items)
        balanced.extend(items[:min_count])

    random.shuffle(balanced)
    return balanced


def make_report(records, deduped, duplicates, train, val, test):
    lines = []
    lines.append("DATASET AUDIT REPORT")
    lines.append("=" * 80)
    lines.append(f"Raw records: {len(records)}")
    lines.append(f"After deduplication: {len(deduped)}")
    lines.append(f"Duplicate records removed: {len(duplicates)}")
    lines.append("")
    lines.append(f"Train: {len(train)}")
    lines.append(f"Val:   {len(val)}")
    lines.append(f"Test:  {len(test)}")
    lines.append("")

    lines.append("Intent distribution:")
    for k, v in Counter(r["intent"] for r in deduped).most_common():
        lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("Language distribution:")
    for k, v in Counter(r["language"] for r in deduped).most_common():
        lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("Train intent distribution:")
    for k, v in Counter(r["intent"] for r in train).most_common():
        lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("Train language distribution:")
    for k, v in Counter(r["language"] for r in train).most_common():
        lines.append(f"  {k}: {v}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--report_file", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_json(args.raw_file)
    records = [normalize_record(r) for r in raw]

    records = [
        r for r in records
        if r["language"] and r["intent"] and r["user_query"]
    ]

    deduped, duplicates = deduplicate(records)
    train, val, test = stratified_split(deduped)

    report = make_report(records, deduped, duplicates, train, val, test)
    Path(args.report_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_file).write_text(report, encoding="utf-8")
    print(report)

    write_jsonl(train, out_dir / "train_raw.jsonl")
    write_jsonl(val, out_dir / "val_raw.jsonl")
    write_jsonl(test, out_dir / "test_raw.jsonl")

    write_jsonl([label_sft(r) for r in val], out_dir / "val.jsonl")
    write_jsonl([label_sft(r) for r in test], out_dir / "test.jsonl")

    # Dataset-size ablations
    for size in [1400, 2800, 7000]:
        subset = train[: min(size, len(train))]
        write_jsonl([label_sft(r) for r in subset], out_dir / f"train_{size}_label.jsonl")
        write_jsonl([reasoning_sft(r) for r in subset], out_dir / f"train_{size}_reasoning.jsonl")

    # Language-type ablations
    english = sample_by_language(train, ["English"])
    native = sample_by_language(train, ["Sinhala", "Tamil"])
    romanized_code = sample_by_language(
        train,
        [
            "Singlish",
            "Tamilish",
            "Tamil English Code Mixed",
            "Sinhala English Code Mixed",
        ],
    )

    write_jsonl([label_sft(r) for r in english], out_dir / "train_english_only_label.jsonl")
    write_jsonl([label_sft(r) for r in native], out_dir / "train_native_si_ta_label.jsonl")
    write_jsonl([label_sft(r) for r in romanized_code], out_dir / "train_romanized_code_mixed_label.jsonl")

    # Balance ablations
    intent_balanced = balanced_by_key(train, "intent")
    language_balanced = balanced_by_key(train, "language")

    write_jsonl([label_sft(r) for r in intent_balanced], out_dir / "train_intent_balanced_label.jsonl")
    write_jsonl([label_sft(r) for r in language_balanced], out_dir / "train_language_balanced_label.jsonl")

    print("\nSaved processed datasets to:", out_dir)


if __name__ == "__main__":
    main()
