import json
from pathlib import Path

from datasets.export_cleaned_corpus import clean_document


SOURCE = Path(
    "datasets/processed/carolina/university_domains/UNIa.jsonl"
)


def main():
    with SOURCE.open(encoding="utf-8") as file:
        original = json.loads(next(file))

    cleaned = clean_document(original)

    print("===== DOCUMENTO =====")
    print("ID:", cleaned["document_id"])

    print("\n===== TEXTO =====")
    print("Original:", len(original["text"]))
    print("Limpo:", len(cleaned["text"]))

    print("\n===== PARÁGRAFOS =====")
    print("Original:", len(original["paragraphs"]))
    print("Limpo:", len(cleaned["paragraphs"]))

    print("\n===== ESTATÍSTICAS =====")
    print(json.dumps(
        cleaned["statistics"],
        ensure_ascii=False,
        indent=2
    ))

    print("\n===== PROCESSAMENTO =====")
    print(json.dumps(
        cleaned["processing"]["cleaning"],
        ensure_ascii=False,
        indent=2
    ))

    print("\n===== PRESERVAÇÃO =====")
    print(
        "Licenças preservadas:",
        cleaned["licenses"] == original["licenses"]
    )

    print(
        "Metadados preservados:",
        cleaned["metadata"] == original["metadata"]
    )

    print(
        "Original inalterado:",
        original["text"] == cleaned["text"]
        if not cleaned["processing"]["cleaning"]["changed"]
        else original["text"] != cleaned["text"]
    )


if __name__ == "__main__":
    main()
