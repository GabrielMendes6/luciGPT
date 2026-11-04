
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


DATASET_DIR = Path(
    "datasets/processed/carolina/university_domains"
)

REPORT_PATH = Path(
    "datasets/processed/carolina/quality_report.json"
)

HTML_PATTERN = re.compile(
    r"<\s*/?\s*[a-zA-Z][^>]*>"
)

CONTROL_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)


def percentile(values: list[int], percent: float) -> int:
    if not values:
        return 0

    ordered = sorted(values)
    index = round((len(ordered) - 1) * percent)

    return ordered[index]


def text_hash(text: str) -> str:
    normalized = " ".join(text.casefold().split())

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def summarize(values: list[int]) -> dict:
    if not values:
        return {}

    ordered = sorted(values)

    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
        "median": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
    }


def analyze_dataset() -> dict:
    document_lengths = []
    document_hashes = Counter()
    paragraph_hashes = Counter()

    typologies = Counter()
    languages = Counter()

    total_documents = 0
    total_paragraphs = 0

    documents_with_html = 0
    documents_with_control_chars = 0
    documents_with_replacement_char = 0
    documents_with_excessive_whitespace = 0

    empty_paragraphs = 0
    short_documents = 0
    long_documents = 0

    files_summary = {}

    print("===== AUDITORIA DE QUALIDADE =====")

    files = sorted(DATASET_DIR.glob("UNI*.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"Nenhum JSONL encontrado em {DATASET_DIR}"
        )

    for file_path in files:
        file_documents = 0

        print(f"\nAnalisando: {file_path.name}")

        with file_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                document = json.loads(line)

                text = document["text"]
                statistics = document["statistics"]

                word_count = statistics["word_count"]

                document_lengths.append(word_count)
                document_hashes[text_hash(text)] += 1

                if word_count < 50:
                    short_documents += 1

                if word_count > 10000:
                    long_documents += 1

                if HTML_PATTERN.search(text):
                    documents_with_html += 1

                if CONTROL_PATTERN.search(text):
                    documents_with_control_chars += 1

                if "\ufffd" in text:
                    documents_with_replacement_char += 1

                if re.search(r"\s{3,}", text):
                    documents_with_excessive_whitespace += 1

                metadata = document["metadata"]
                source = document["source"]

                language = metadata.get("language")

                if language:
                    languages[language] += 1

                update_types = source.get("typology", [])

                for typology in update_types:
                    typologies[typology] += 1

                paragraphs = document["paragraphs"]

                total_paragraphs += len(paragraphs)

                for paragraph in paragraphs:
                    normalized = " ".join(
                        paragraph.casefold().split()
                    )

                    if not normalized:
                        empty_paragraphs += 1
                        continue

                    paragraph_hashes[
                        hashlib.sha256(
                            normalized.encode("utf-8")
                        ).hexdigest()
                    ] += 1

                total_documents += 1
                file_documents += 1

        files_summary[file_path.name] = file_documents

        print(
            f"  Documentos analisados: {file_documents}"
        )

    duplicated_documents = sum(
        count - 1
        for count in document_hashes.values()
        if count > 1
    )

    duplicated_paragraphs = sum(
        count - 1
        for count in paragraph_hashes.values()
        if count > 1
    )

    report = {
        "dataset": "Corpus Carolina - University Domains",

        "summary": {
            "files": len(files),
            "documents": total_documents,
            "paragraphs": total_paragraphs,
            "languages": dict(languages.most_common()),
            "typologies": dict(typologies.most_common()),
        },

        "document_length_words": summarize(
            document_lengths
        ),

        "duplicates": {
            "unique_document_texts": len(document_hashes),
            "duplicated_documents": duplicated_documents,
            "unique_paragraphs": len(paragraph_hashes),
            "duplicated_paragraphs": duplicated_paragraphs,
        },

        "quality_indicators": {
            "documents_under_50_words": short_documents,
            "documents_over_10000_words": long_documents,
            "documents_with_html": documents_with_html,
            "documents_with_control_characters": (
                documents_with_control_chars
            ),
            "documents_with_replacement_characters": (
                documents_with_replacement_char
            ),
            "documents_with_excessive_whitespace": (
                documents_with_excessive_whitespace
            ),
            "empty_paragraphs": empty_paragraphs,
        },

        "files": files_summary,
    }

    return report


def main() -> None:
    report = analyze_dataset()

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = REPORT_PATH.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(
            report,
            output,
            ensure_ascii=False,
            indent=4,
        )

    temporary_path.replace(REPORT_PATH)

    print("\n===== RESULTADO DA AUDITORIA =====")

    print(
        "Documentos:",
        report["summary"]["documents"],
    )

    print(
        "Parágrafos:",
        report["summary"]["paragraphs"],
    )

    print("\nDistribuição de palavras por documento:")

    for key, value in report["document_length_words"].items():
        print(f"  {key}: {value}")

    print("\nDuplicações:")

    for key, value in report["duplicates"].items():
        print(f"  {key}: {value}")

    print("\nIndicadores de qualidade:")

    for key, value in report["quality_indicators"].items():
        print(f"  {key}: {value}")

    print("\nRelatório salvo em:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
