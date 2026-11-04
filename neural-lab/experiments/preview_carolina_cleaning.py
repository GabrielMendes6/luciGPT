
import json

from collections import Counter
from pathlib import Path

from datasets.text_cleaner import clean_text


SOURCE_DIR = Path(
    "datasets/processed/carolina/university_domains"
)

OUTPUT_PATH = Path(
    "datasets/processed/carolina/cleaning_preview.jsonl"
)

MAX_EXAMPLES = 100


def preview_cleaning():
    files = sorted(SOURCE_DIR.glob("UNI*.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"Nenhum JSONL encontrado em {SOURCE_DIR}"
        )

    total_documents = 0
    changed_documents = 0
    unchanged_documents = 0
    empty_after_cleaning = 0

    change_counts = Counter()
    examples_written = 0

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = OUTPUT_PATH.with_suffix(".tmp")

    print("===== PRÉVIA DE LIMPEZA DO CORPUS =====")

    with temporary_path.open("w", encoding="utf-8") as output:

        for file_path in files:
            print(f"\nAnalisando: {file_path.name}")

            with file_path.open("r", encoding="utf-8") as source:

                for line in source:
                    document = json.loads(line)

                    original_text = document["text"]
                    result = clean_text(original_text)

                    total_documents += 1

                    if result.changed:
                        changed_documents += 1

                        for change, count in result.changes.items():
                            change_counts[change] += count

                        if not result.cleaned_text:
                            empty_after_cleaning += 1

                        # Salva apenas documentos alterados.
                        if examples_written < MAX_EXAMPLES:
                            preview = {
                                "document_id": document["document_id"],
                                "title": document["metadata"].get("title"),
                                "typology": document["source"].get("typology", []),
                                "source_file": file_path.name,
                                "changes": result.changes,
                                "original_text": original_text,
                                "cleaned_text": result.cleaned_text,
                            }

                            output.write(
                                json.dumps(
                                    preview,
                                    ensure_ascii=False,
                                ) + "\n"
                            )

                            examples_written += 1

                    else:
                        unchanged_documents += 1

    temporary_path.replace(OUTPUT_PATH)

    print("\n===== RESULTADO =====")
    print("Documentos analisados:", total_documents)
    print("Documentos alterados:", changed_documents)
    print("Documentos inalterados:", unchanged_documents)
    print("Documentos vazios após limpeza:", empty_after_cleaning)

    print("\n===== TRANSFORMAÇÕES =====")

    for change, count in change_counts.most_common():
        print(f"{change}: {count}")

    print("\n===== ARQUIVO DE PRÉVIA =====")
    print("Exemplos gravados:", examples_written)
    print("Arquivo:", OUTPUT_PATH)


if __name__ == "__main__":
    preview_cleaning()
