
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    BASE_DIR
    / "processed"
    / "carolina"
    / "university_domains"
)

MANIFEST_PATH = (
    BASE_DIR
    / "processed"
    / "carolina"
    / "manifest.json"
)

SOURCE_FILES = [
    "UNIa.jsonl",
    "UNIb.jsonl",
    "UNIc.jsonl",
    "UNId.jsonl",
    "UNIe.jsonl",
    "UNIf.jsonl",
    "UNIg.jsonl",
]


def update_distribution(counter: Counter, values: list) -> None:
    for value in values:
        if value:
            counter[value] += 1


def create_manifest() -> dict:
    total_documents = 0
    total_words = 0
    total_characters = 0
    total_paragraphs = 0

    document_ids = set()

    languages = Counter()
    categories = Counter()
    typologies = Counter()
    corpus_licenses = {}
    original_licenses = {}

    files = {}

    print("===== GERANDO MANIFESTO =====")

    for filename in SOURCE_FILES:
        file_path = DATASET_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        file_documents = 0
        file_words = 0
        file_characters = 0

        print(f"\nAnalisando: {filename}")

        with file_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                document = json.loads(line)

                document_id = document["document_id"]

                if not document_id:
                    raise ValueError(
                        f"{filename}, linha {line_number}: ID vazio."
                    )

                if document_id in document_ids:
                    raise ValueError(
                        f"ID duplicado encontrado: {document_id}"
                    )

                document_ids.add(document_id)

                metadata = document["metadata"]
                source_metadata = document["source"]
                licenses = document["licenses"]
                statistics = document["statistics"]

                language = metadata.get("language")

                if language:
                    languages[language] += 1

                update_distribution(
                    categories,
                    source_metadata.get("category", []),
                )

                update_distribution(
                    typologies,
                    source_metadata.get("typology", []),
                )

                # Licença geral do Corpus Carolina.
                corpus_license = licenses.get("corpus", {})

                corpus_key = (
                    corpus_license.get("name"),
                    corpus_license.get("url"),
                )

                if corpus_key not in corpus_licenses:
                    corpus_licenses[corpus_key] = 0

                corpus_licenses[corpus_key] += 1

                # Licença da fonte original.
                original_license = licenses.get(
                    "original_source", {}
                )

                original_key = (
                    original_license.get("name"),
                    original_license.get("url"),
                    original_license.get("status"),
                )

                if original_key not in original_licenses:
                    original_licenses[original_key] = 0

                original_licenses[original_key] += 1

                words = statistics["word_count"]
                characters = statistics["character_count"]
                paragraphs = statistics["paragraph_count"]

                file_documents += 1
                file_words += words
                file_characters += characters

                total_documents += 1
                total_words += words
                total_characters += characters
                total_paragraphs += paragraphs

        files[filename] = {
            "documents": file_documents,
            "words": file_words,
            "characters": file_characters,
        }

        print(f"  Documentos: {file_documents}")
        print(f"  Palavras: {file_words}")

    manifest = {
        "dataset": {
            "name": "Corpus Carolina - University Domains",
            "version": "2.0.1",
            "language": "pt-BR",
            "format": "JSONL",
            "category": "UNIVERSITY_DOMAINS",
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        },

        "provenance": {
            "repository": (
                "https://huggingface.co/datasets/"
                "carolina-c4ai/corpus-carolina"
            ),
            "source_directory": (
                "corpus/university_domains/"
            ),
            "parser": "datasets/carolina_parser.py",
            "exporter": "datasets/export_carolina.py",
        },

        "licenses": {
            "corpus": [
                {
                    "name": name,
                    "url": url,
                    "documents": count,
                }
                for (name, url), count
                in corpus_licenses.items()
            ],
            "original_sources": [
                {
                    "name": name,
                    "url": url,
                    "status": status,
                    "documents": count,
                }
                for (name, url, status), count
                in original_licenses.items()
            ],
            "note": (
                "A licença geral do corpus não substitui "
                "as condições de uso das fontes originais. "
                "Verificar a licença e a atribuição de cada "
                "fonte antes de utilizar seus documentos."
            ),
        },

        "statistics": {
            "total_documents": total_documents,
            "unique_documents": len(document_ids),
            "total_words": total_words,
            "total_characters": total_characters,
            "total_paragraphs": total_paragraphs,
            "files_count": len(files),
            "languages": dict(languages.most_common()),
            "categories": dict(categories.most_common()),
            "typologies": dict(typologies.most_common()),
        },

        "files": files,
    }

    return manifest


def main() -> None:
    manifest = create_manifest()

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = MANIFEST_PATH.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(
            manifest,
            output,
            ensure_ascii=False,
            indent=4,
        )

    temporary_path.replace(MANIFEST_PATH)

    print("\n===== MANIFESTO CRIADO =====")
    print("Documentos:", manifest["statistics"]["total_documents"])
    print("Palavras:", manifest["statistics"]["total_words"])
    print("Caracteres:", manifest["statistics"]["total_characters"])
    print("Parágrafos:", manifest["statistics"]["total_paragraphs"])
    print("Arquivos:", manifest["statistics"]["files_count"])
    print("Saída:", MANIFEST_PATH)


if __name__ == "__main__":
    main()
