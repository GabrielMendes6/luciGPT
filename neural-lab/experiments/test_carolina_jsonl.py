
import json
from collections import Counter
from pathlib import Path


DATASET_DIR = Path(
    "datasets/processed/carolina/"
    "university_domains"
)

required_fields = {
    "document_id",
    "source",
    "metadata",
    "licenses",
    "processing",
    "text",
    "paragraphs",
    "statistics",
}

document_ids = set()
languages = Counter()
categories = Counter()
typologies = Counter()
files_summary = {}

total_documents = 0
empty_text = 0
missing_language = 0
missing_source_license = 0
invalid_statistics = 0

print("===== VALIDAÇÃO GLOBAL DO CAROLINA =====")

for file_path in sorted(DATASET_DIR.glob("UNI*.jsonl")):
    file_count = 0

    print(f"\nVerificando: {file_path.name}")

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                document = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{file_path.name}, linha {line_number}: "
                    f"JSON inválido: {error}"
                ) from error

            if not isinstance(document, dict):
                raise ValueError(
                    f"{file_path.name}, linha {line_number}: "
                    "registro não é um objeto."
                )

            missing_fields = required_fields - document.keys()

            if missing_fields:
                raise ValueError(
                    f"{file_path.name}, linha {line_number}: "
                    f"campos ausentes: {missing_fields}"
                )

            document_id = document["document_id"]

            if not document_id:
                raise ValueError(
                    f"{file_path.name}, linha {line_number}: "
                    "ID vazio."
                )

            if document_id in document_ids:
                raise ValueError(
                    f"ID duplicado: {document_id}"
                )

            document_ids.add(document_id)

            metadata = document["metadata"]
            licenses = document["licenses"]
            source = document["source"]
            statistics = document["statistics"]

            language = metadata.get("language")

            if language:
                languages[language] += 1
            else:
                missing_language += 1

            if not document["text"].strip():
                empty_text += 1

            if not licenses.get("original_source", {}).get("name"):
                missing_source_license += 1

            for category in source.get("category", []):
                categories[category] += 1

            for typology in source.get("typology", []):
                typologies[typology] += 1

            actual_characters = len(document["text"])
            actual_words = len(document["text"].split())

            if (
                statistics.get("character_count") != actual_characters
                or statistics.get("word_count") != actual_words
                or statistics.get("paragraph_count")
                != len(document["paragraphs"])
            ):
                invalid_statistics += 1

            file_count += 1
            total_documents += 1

    files_summary[file_path.name] = file_count
    print(f"  Documentos: {file_count}")

print("\n===== RESULTADO GLOBAL =====")
print("Arquivos:", len(files_summary))
print("Documentos:", total_documents)
print("IDs únicos:", len(document_ids))
print("Textos vazios:", empty_text)
print("Idiomas ausentes:", missing_language)
print("Licenças de origem ausentes:", missing_source_license)
print("Estatísticas inconsistentes:", invalid_statistics)

print("\n===== IDIOMAS =====")
for language, count in languages.most_common():
    print(f"{language}: {count}")

print("\n===== CATEGORIAS =====")
for category, count in categories.most_common():
    print(f"{category}: {count}")

print("\n===== TIPOLOGIAS =====")
for typology, count in typologies.most_common():
    print(f"{typology}: {count}")

assert len(files_summary) == 7
assert total_documents == 26409
assert len(document_ids) == total_documents
assert empty_text == 0
assert missing_language == 0
assert missing_source_license == 0
assert invalid_statistics == 0

print("\nCORPUS CAROLINA VALIDADO COM SUCESSO!")
