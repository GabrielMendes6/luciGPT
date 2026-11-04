
import hashlib
import json
import re

from collections import defaultdict
from pathlib import Path


DATASET_DIR = Path(
    "datasets/processed/carolina/university_domains"
)

OUTPUT_PATH = Path(
    "datasets/processed/carolina/anomalies.jsonl"
)

HTML_PATTERN = re.compile(
    r"<\s*/?\s*[a-zA-Z][^>]*>"
)

CONTROL_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)

SHORT_DOCUMENT_LIMIT = 50
TOP_REPEATED_PARAGRAPHS = 100


def normalize_paragraph(text: str) -> str:
    return " ".join(text.casefold().split())


def paragraph_hash(text: str) -> str:
    normalized = normalize_paragraph(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def document_reference(
    document: dict,
    filename: str,
) -> dict:
    return {
        "document_id": document["document_id"],
        "file": filename,
        "title": document["metadata"].get("title"),
        "typology": document["source"].get("typology", []),
    }


def inspect_dataset() -> dict:
    html_documents = []
    control_documents = []
    short_documents = []

    paragraphs = defaultdict(
        lambda: {
            "text": None,
            "occurrences": [],
        }
    )

    total_documents = 0
    total_paragraphs = 0

    files = sorted(DATASET_DIR.glob("UNI*.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo JSONL encontrado em {DATASET_DIR}"
        )

    print("===== INSPEÇÃO DE ANOMALIAS =====")

    for file_path in files:
        print(f"\nAnalisando: {file_path.name}")

        with file_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                document = json.loads(line)

                reference = document_reference(
                    document,
                    file_path.name,
                )

                text = document["text"]
                word_count = document["statistics"]["word_count"]

                total_documents += 1

                # 1. Possíveis resíduos HTML.
                html_matches = [
                    match.group(0)
                    for match in HTML_PATTERN.finditer(text)
                ]

                if html_matches:
                    html_documents.append({
                        "type": "html_document",
                        **reference,
                        "matches": html_matches[:10],
                        "total_matches": len(html_matches),
                    })

                # 2. Caracteres de controle.
                control_matches = [
                    {
                        "character": repr(match.group(0)),
                        "codepoint": (
                            f"U+{ord(match.group(0)):04X}"
                        ),
                        "position": match.start(),
                    }
                    for match in CONTROL_PATTERN.finditer(text)
                ]

                if control_matches:
                    control_documents.append({
                        "type": "control_character_document",
                        **reference,
                        "matches": control_matches,
                    })

                # 3. Documentos curtos.
                if word_count < SHORT_DOCUMENT_LIMIT:
                    short_documents.append({
                        "type": "short_document",
                        **reference,
                        "word_count": word_count,
                        "text_preview": text[:500],
                    })

                # 4. Parágrafos repetidos.
                for paragraph_index, paragraph in enumerate(
                    document["paragraphs"]
                ):
                    normalized = normalize_paragraph(paragraph)

                    if not normalized:
                        continue

                    digest = paragraph_hash(paragraph)

                    entry = paragraphs[digest]

                    if entry["text"] is None:
                        entry["text"] = paragraph

                    entry["occurrences"].append({
                        "document_id": document["document_id"],
                        "file": file_path.name,
                        "paragraph_index": paragraph_index,
                    })

                    total_paragraphs += 1

    # Seleciona os 100 parágrafos com mais ocorrências.
    repeated = [
        {
            "type": "repeated_paragraph",
            "paragraph_hash": digest,
            "occurrence_count": len(entry["occurrences"]),
            "document_count": len({
                occurrence["document_id"]
                for occurrence in entry["occurrences"]
            }),
            "text": entry["text"],
            "occurrences": entry["occurrences"],
        }
        for digest, entry in paragraphs.items()
        if len(entry["occurrences"]) > 1
    ]

    repeated.sort(
        key=lambda item: (
            -item["occurrence_count"],
            item["paragraph_hash"],
        )
    )

    repeated = repeated[:TOP_REPEATED_PARAGRAPHS]

    return {
        "summary": {
            "files": len(files),
            "documents": total_documents,
            "paragraphs": total_paragraphs,
            "html_documents": len(html_documents),
            "control_documents": len(control_documents),
            "short_documents": len(short_documents),
            "unique_paragraphs": len(paragraphs),
            "repeated_paragraph_groups": sum(
                1 for entry in paragraphs.values()
                if len(entry["occurrences"]) > 1
            ),
            "repeated_paragraphs_reported": len(repeated),
        },
        "html_documents": html_documents,
        "control_documents": control_documents,
        "short_documents": short_documents,
        "repeated_paragraphs": repeated,
    }


def save_anomalies(results: dict) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = OUTPUT_PATH.with_suffix(".tmp")

    categories = [
        "html_documents",
        "control_documents",
        "short_documents",
        "repeated_paragraphs",
    ]

    with temporary_path.open("w", encoding="utf-8") as output:
        for category in categories:
            for anomaly in results[category]:
                output.write(
                    json.dumps(
                        anomaly,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    temporary_path.replace(OUTPUT_PATH)


def main() -> None:
    results = inspect_dataset()

    save_anomalies(results)

    summary = results["summary"]

    print("\n===== RESUMO DA INSPEÇÃO =====")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n===== DOCUMENTOS COM HTML =====")

    for document in results["html_documents"][:10]:
        print(
            document["document_id"],
            document["matches"],
        )

    print("\n===== DOCUMENTOS COM CARACTERES DE CONTROLE =====")

    for document in results["control_documents"]:
        print(
            document["document_id"],
            document["matches"],
        )

    print("\n===== DOCUMENTOS CURTOS =====")

    for document in results["short_documents"][:10]:
        print(
            document["document_id"],
            f'{document["word_count"]} palavras',
        )

    print("\n===== 10 PARÁGRAFOS MAIS REPETIDOS =====")

    for index, paragraph in enumerate(
        results["repeated_paragraphs"][:10],
        start=1,
    ):
        print(f"\n{index}.")
        print("Ocorrências:", paragraph["occurrence_count"])
        print("Documentos:", paragraph["document_count"])
        print("Texto:", paragraph["text"][:300])

        for occurrence in paragraph["occurrences"][:5]:
            print(
                "  -",
                occurrence["document_id"],
                occurrence["file"],
                f'parágrafo {occurrence["paragraph_index"]}',
            )

    print("\n===== INSPEÇÃO CONCLUÍDA =====")
    print("Relatório:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
