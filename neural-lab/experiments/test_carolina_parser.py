
import json

from datasets.carolina_parser import (
    iter_carolina_documents,
)


FILE = (
    "datasets/sources/carolina/"
    "corpus/university_domains/UNIa.xml.gz"
)

print("===== PROCESSANDO CORPUS CAROLINA =====")

documents = iter_carolina_documents(FILE)

first_document = next(documents)

print("\n===== IDENTIFICAÇÃO =====")
print("ID:", first_document["document_id"])

print("\n===== METADADOS =====")
print(json.dumps(
    first_document["metadata"],
    indent=4,
    ensure_ascii=False,
))

print("\n===== LICENÇAS =====")
print(json.dumps(
    first_document["licenses"],
    indent=4,
    ensure_ascii=False,
))

print("\n===== ESTATÍSTICAS =====")
print(json.dumps(
    first_document["statistics"],
    indent=4,
    ensure_ascii=False,
))

print("\n===== AMOSTRA TEXTUAL =====")
print(first_document["text"][:1000])

print("\n===== VALIDAÇÕES =====")

assert first_document["document_id"] == "UNI000000001a"

assert first_document["metadata"]["title"] == (
    "Um Jabuti, dois Jabutis, três Jabutis"
)

assert first_document["metadata"]["language"] == "pt-BR"

assert first_document["licenses"]["corpus"]["name"] == "CC BY 4.0"

assert len(first_document["paragraphs"]) > 0

assert first_document["statistics"]["character_count"] > 0

print("Primeiro documento validado!")

print("\n===== CONTAGEM DE DOCUMENTOS =====")

total = 1

for document in documents:
    total += 1

print("Documentos processados:", total)

print("\nPARSER VALIDADO COM SUCESSO!")
