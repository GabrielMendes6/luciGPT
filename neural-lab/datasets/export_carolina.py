
import json
from pathlib import Path

from datasets.carolina_parser import iter_carolina_documents


BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = (
    BASE_DIR
    / "sources"
    / "carolina"
    / "corpus"
    / "university_domains"
)

OUTPUT_DIR = (
    BASE_DIR
    / "processed"
    / "carolina"
    / "university_domains"
)

SOURCE_FILES = [
    "UNIa.xml.gz",
    "UNIb.xml.gz",
    "UNIc.xml.gz",
    "UNId.xml.gz",
    "UNIe.xml.gz",
    "UNIf.xml.gz",
    "UNIg.xml.gz",
]

def export_file(filename: str) -> int:
    source_path = SOURCE_DIR / filename
    output_path = OUTPUT_DIR / filename.replace(".xml.gz", ".jsonl")

    if not source_path.exists():
        raise FileNotFoundError(
            f"Arquivo de origem não encontrado: {source_path}"
        )

    document_count = 0

    print(f"\nProcessando: {filename}")

    with output_path.open("w", encoding="utf-8") as output:
        for document in iter_carolina_documents(source_path):
            output.write(
                json.dumps(
                    document,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            document_count += 1

            if document_count % 500 == 0:
                print(
                    f"  Documentos exportados: {document_count}"
                )

    print(f"  Total: {document_count}")
    print(f"  Saída: {output_path}")

    return document_count


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_documents = 0

    for filename in SOURCE_FILES:
        count = export_file(filename)
        total_documents += count

    print("\n===== EXPORTAÇÃO CONCLUÍDA =====")
    print(f"Total de documentos: {total_documents}")
    print(f"Diretório: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
