
import hashlib
import json

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


from datasets.text_cleaner import clean_text


BASE_DIR = Path("datasets/processed/carolina")

SOURCE_DIR = BASE_DIR / "university_domains"
OUTPUT_DIR = BASE_DIR / "cleaned" / "university_domains"

SOURCE_MANIFEST = BASE_DIR / "manifest.json"
OUTPUT_MANIFEST = BASE_DIR / "cleaned" / "manifest.json"


def calculate_statistics(
    text: str,
    paragraphs: list[str],
) -> dict:
    return {
        "paragraph_count": len(paragraphs),
        "character_count": len(text),
        "word_count": len(text.split()),
    }


def clean_document(document: dict) -> dict:
    """
    Limpa um documento preservando seus metadados originais.
    """

    cleaned_document = deepcopy(document)

    original_text = document.get("text", "")
    original_paragraphs = document.get("paragraphs", [])

    if not isinstance(original_text, str):
        raise TypeError("O campo text precisa ser uma string.")

    if not isinstance(original_paragraphs, list):
        raise TypeError("O campo paragraphs precisa ser uma lista.")

    # Limpa o texto completo.
    text_result = clean_text(original_text)

    # Limpa os parágrafos individualmente.
    cleaned_paragraphs = []
    paragraph_changes = {}

    for index, paragraph in enumerate(original_paragraphs):

        if not isinstance(paragraph, str):
            raise TypeError(
                f"Parágrafo {index} não é uma string."
            )

        result = clean_text(paragraph)

        if result.cleaned_text:
            cleaned_paragraphs.append(result.cleaned_text)

        if result.changes:
            paragraph_changes[str(index)] = result.changes

    # Reconstrói o texto a partir dos parágrafos limpos.
    reconstructed_text = "\n\n".join(cleaned_paragraphs)

    # Impede exportação caso haja divergência estrutural.
    if text_result.cleaned_text != reconstructed_text:
        raise ValueError(
            f"Divergência entre texto e parágrafos: "
            f"{document.get('document_id')}"
        )

    cleaned_document["text"] = reconstructed_text
    cleaned_document["paragraphs"] = cleaned_paragraphs

    # Atualiza estatísticas.
    cleaned_document["statistics"] = calculate_statistics(
        reconstructed_text,
        cleaned_paragraphs,
    )

    # Registra informações do processamento.
    cleaned_document.setdefault("processing", {})

    cleaned_document["processing"]["cleaning"] = {
        "changed": (
            text_result.changed
            or bool(paragraph_changes)
        ),
        "text_changes": text_result.changes,
        "paragraph_changes": paragraph_changes,
        "original_character_count": len(original_text),
        "cleaned_character_count": len(reconstructed_text),
        "original_paragraph_count": len(original_paragraphs),
        "cleaned_paragraph_count": len(cleaned_paragraphs),
    }

    return cleaned_document


def safe_key(value) -> str:
    if value is None:
        return "AUSENTE"

    if isinstance(value, (list, dict)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    return str(value)


def export_cleaned_corpus():

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_files = sorted(SOURCE_DIR.glob("UNI*.jsonl"))

    if not source_files:
        raise FileNotFoundError(
            f"Nenhum arquivo encontrado em {SOURCE_DIR}"
        )

    seen_ids = set()

    total_documents = 0
    changed_documents = 0
    total_words = 0
    total_characters = 0
    total_paragraphs = 0

    text_change_events = Counter()
    paragraph_change_events = Counter()

    domains = Counter()
    typologies = Counter()
    original_licenses = Counter()
    corpus_licenses = Counter()

    file_statistics = {}
    input_hashes = {}

    temporary_files = []

    try:

        for source_file in source_files:

            output_file = OUTPUT_DIR / source_file.name
            temporary_file = output_file.with_suffix(
                output_file.suffix + ".tmp"
            )

            file_documents = 0
            file_words = 0

            input_hash = hashlib.sha256()

            print(f"Processando: {source_file.name}")

            with source_file.open("rb") as input_stream, \
                 temporary_file.open(
                     "w",
                     encoding="utf-8",
                     newline="\n",
                 ) as output_stream:

                temporary_files.append(
                    (temporary_file, output_file)
                )

                for raw_line in input_stream:

                    input_hash.update(raw_line)

                    document = json.loads(
                        raw_line.decode("utf-8")
                    )

                    document_id = document.get("document_id")

                    if not document_id:
                        raise ValueError(
                            "Documento sem document_id."
                        )

                    if document_id in seen_ids:
                        raise ValueError(
                            f"ID duplicado: {document_id}"
                        )

                    seen_ids.add(document_id)

                    cleaned = clean_document(document)

                    # Estatísticas de limpeza.
                    cleaning = cleaned["processing"]["cleaning"]

                    if cleaning["changed"]:
                        changed_documents += 1

                    text_change_events.update(
                        cleaning["text_changes"]
                    )

                    for changes in cleaning[
                        "paragraph_changes"
                    ].values():
                        paragraph_change_events.update(changes)

                    # Estatísticas do corpus.
                    statistics = cleaned["statistics"]

                    total_documents += 1
                    file_documents += 1

                    total_words += statistics["word_count"]
                    file_words += statistics["word_count"]

                    total_characters += statistics["character_count"]
                    total_paragraphs += statistics["paragraph_count"]

                    # Proveniência e licenciamento.
                    metadata = cleaned.get("metadata", {})
                    licenses = cleaned.get("licenses", {})
                    source = cleaned.get("source", {})

                    url = metadata.get("url")

                    if isinstance(url, str) and url:
                        hostname = urlsplit(url).hostname

                        if hostname:
                            domains[hostname.lower()] += 1

                    corpus_license = licenses.get("corpus", {})
                    original_license = licenses.get(
                        "original_source", {}
                    )

                    corpus_licenses[
                        safe_key(corpus_license.get("name"))
                    ] += 1

                    original_licenses[
                        safe_key(original_license.get("name"))
                    ] += 1

                    typologies[
                        safe_key(source.get("typology"))
                    ] += 1

                    # Exporta JSONL.
                    output_stream.write(
                        json.dumps(
                            cleaned,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ) + "\n"
                    )

            input_hashes[source_file.name] = (
                input_hash.hexdigest()
            )

            file_statistics[source_file.name] = {
                "documents": file_documents,
                "words": file_words,
            }

            print(
                f"  Documentos: {file_documents} | "
                f"Palavras: {file_words}"
            )

        # Recupera informações do manifesto original.
        source_manifest_data = {}

        if SOURCE_MANIFEST.exists():
            with SOURCE_MANIFEST.open(
                encoding="utf-8"
            ) as file:
                source_manifest_data = json.load(file)

        # Monta manifesto do corpus limpo.
        manifest = {
            "dataset": "LUCI Cleaned Corpus",
            "source": "Corpus Carolina",
            "source_version": source_manifest_data.get(
                "version", "UNKNOWN"
            ),
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),

            "cleaning": {
                "method": "datasets.text_cleaner.clean_text",
                "reconstruction": "paragraphs_joined_with_double_newline",
                "changed_documents": changed_documents,
                "unchanged_documents": (
                    total_documents - changed_documents
                ),
                "text_change_events": dict(text_change_events),
                "paragraph_change_events": dict(
                    paragraph_change_events
                ),
            },

            "statistics": {
                "document_count": total_documents,
                "word_count": total_words,
                "character_count": total_characters,
                "paragraph_count": total_paragraphs,
                "file_count": len(source_files),
            },

            "distribution": {
                "domains": dict(domains),
                "typologies": dict(typologies),
                "corpus_licenses": dict(corpus_licenses),
                "original_source_licenses": dict(
                    original_licenses
                ),
            },

            "files": file_statistics,

            "source_files_sha256": input_hashes,

            "integrity": {
                "unique_document_ids": len(seen_ids),
                "duplicate_document_ids": 0,
                "text_paragraph_consistency": True,
            },
        }

        OUTPUT_MANIFEST.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_temp = OUTPUT_MANIFEST.with_suffix(
            ".json.tmp"
        )

        with manifest_temp.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                ensure_ascii=False,
                indent=2,
            )

        # Publica os arquivos somente após todos os
        # documentos terem sido processados com sucesso.
        for temporary_file, output_file in temporary_files:
            temporary_file.replace(output_file)

        manifest_temp.replace(OUTPUT_MANIFEST)

    except Exception:

        for temporary_file, _ in temporary_files:
            temporary_file.unlink(missing_ok=True)

        if "manifest_temp" in locals():
            manifest_temp.unlink(missing_ok=True)

        raise

    print("\n===== EXPORTAÇÃO CONCLUÍDA =====")
    print("Documentos:", total_documents)
    print("Alterados:", changed_documents)
    print("Inalterados:", total_documents - changed_documents)
    print("Palavras:", total_words)
    print("Caracteres:", total_characters)
    print("Parágrafos:", total_paragraphs)
    print("Arquivos:", len(source_files))

    print("\nSaída:", OUTPUT_DIR)
    print("Manifesto:", OUTPUT_MANIFEST)


if __name__ == "__main__":
    export_cleaned_corpus()
