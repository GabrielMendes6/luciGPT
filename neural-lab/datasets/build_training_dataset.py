
import hashlib
import json

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


INPUT_DIR = Path(
    "datasets/processed/carolina/cleaned/university_domains"
)

INPUT_MANIFEST = Path(
    "datasets/processed/carolina/cleaned/manifest.json"
)

OUTPUT_DIR = Path("datasets/training/carolina")

SEED = 42

SPLITS = ("train", "validation", "test")


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def text_hash(text: str) -> str:
    normalized = normalize_text(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def deterministic_order(digest: str) -> str:
    value = f"{SEED}:{digest}"

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


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


def extract_typologies(document: dict) -> tuple:
    typologies = document.get(
        "source", {}
    ).get("typology", [])

    if isinstance(typologies, str):
        typologies = [typologies]

    if not typologies:
        typologies = ["UNSPECIFIED"]

    return tuple(sorted(set(typologies)))


def allocate_groups(group_count: int) -> dict:
    """
    Distribui grupos de documentos entre os conjuntos.

    Para estratos pequenos, prioriza a presença de exemplos
    independentes em validação e teste.
    """

    if group_count <= 0:
        return {
            "train": 0,
            "validation": 0,
            "test": 0,
        }

    if group_count == 1:
        return {
            "train": 1,
            "validation": 0,
            "test": 0,
        }

    if group_count == 2:
        return {
            "train": 1,
            "validation": 1,
            "test": 0,
        }

    validation = max(1, round(group_count * 0.10))
    test = max(1, round(group_count * 0.10))

    train = group_count - validation - test

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


def build_training_record(
    document: dict,
    split: str,
) -> dict:

    metadata = document.get("metadata", {})
    corpus_source = document.get("source", {})
    licenses = document.get("licenses", {})

    url = metadata.get("url")

    domain = None

    if isinstance(url, str) and url:
        domain = urlsplit(url).hostname

    return {
        "document_id": document["document_id"],

        "text": document["text"],

        "language": metadata.get("language"),

        "source": {
            "corpus": corpus_source.get("corpus"),
            "corpus_version": corpus_source.get(
                "corpus_version"
            ),
            "category": corpus_source.get("category"),
            "typology": corpus_source.get("typology"),

            "url": url,
            "domain": domain,

            "title": metadata.get("title"),
            "authors": metadata.get("authors", []),
            "publication_date": metadata.get(
                "publication_date"
            ),
            "publisher": metadata.get("publisher"),
            "authority": metadata.get("authority"),
            "series": metadata.get("series"),
        },

        "license": {
            "corpus": licenses.get("corpus"),
            "original_source": licenses.get(
                "original_source"
            ),
        },

        "split": split,
    }


def load_documents() -> list:
    documents = []

    input_files = sorted(
        INPUT_DIR.glob("UNI*.jsonl")
    )

    if not input_files:
        raise FileNotFoundError(
            f"Nenhum JSONL encontrado em {INPUT_DIR}"
        )

    seen_ids = set()

    for input_file in input_files:

        print(f"Lendo: {input_file.name}")

        with input_file.open(
            encoding="utf-8"
        ) as file:

            for line_number, line in enumerate(file, 1):

                document = json.loads(line)

                document_id = document.get("document_id")

                if not document_id:
                    raise ValueError(
                        f"ID ausente: {input_file}:{line_number}"
                    )

                if document_id in seen_ids:
                    raise ValueError(
                        f"ID duplicado: {document_id}"
                    )

                seen_ids.add(document_id)

                if not document.get("text", "").strip():
                    raise ValueError(
                        f"Texto vazio: {document_id}"
                    )

                if not document.get("metadata", {}).get(
                    "language"
                ):
                    raise ValueError(
                        f"Idioma ausente: {document_id}"
                    )

                licenses = document.get("licenses", {})

                if not licenses.get("corpus"):
                    raise ValueError(
                        f"Licença do corpus ausente: {document_id}"
                    )

                if not licenses.get("original_source"):
                    raise ValueError(
                        f"Licença original ausente: {document_id}"
                    )

                documents.append(document)

    return documents


def assign_splits(documents: list) -> dict:
    """
    Agrupa textos idênticos antes de particionar.

    A tipologia de um grupo é a união das tipologias
    de todos os documentos que possuem o mesmo texto.
    """

    groups = {}

    for document in documents:

        digest = text_hash(document["text"])

        if digest not in groups:
            groups[digest] = {
                "documents": [],
                "typologies": set(),
            }

        groups[digest]["documents"].append(document)

        groups[digest]["typologies"].update(
            extract_typologies(document)
        )

    strata = defaultdict(list)

    for digest, group in groups.items():

        signature = tuple(
            sorted(group["typologies"])
        )

        strata[signature].append({
            "digest": digest,
            "documents": group["documents"],
        })

    assigned = {
        split: []
        for split in SPLITS
    }

    stratum_statistics = {}

    for signature, stratum_groups in sorted(
        strata.items()
    ):

        ordered_groups = sorted(
            stratum_groups,
            key=lambda group: deterministic_order(
                group["digest"]
            ),
        )

        allocation = allocate_groups(
            len(ordered_groups)
        )

        position = 0

        for split in SPLITS:

            amount = allocation[split]

            selected_groups = ordered_groups[
                position:position + amount
            ]

            position += amount

            for group in selected_groups:

                for document in group["documents"]:

                    assigned[split].append(
                        build_training_record(
                            document,
                            split,
                        )
                    )

        stratum_statistics[
            " | ".join(signature)
        ] = {
            "unique_text_groups": len(ordered_groups),
            "documents": sum(
                len(group["documents"])
                for group in ordered_groups
            ),
            "group_allocation": allocation,
        }

    for split in SPLITS:
        assigned[split].sort(
            key=lambda record: record["document_id"]
        )

    return assigned, groups, stratum_statistics


def calculate_manifest(
    assigned: dict,
    groups: dict,
    stratum_statistics: dict,
    source_manifest_hash: str,
) -> dict:

    split_statistics = {}

    total_documents = 0
    total_words = 0

    typologies_by_split = {}
    domains_by_split = {}

    for split, records in assigned.items():

        words = 0
        typologies = Counter()
        domains = Counter()
        corpus_licenses = Counter()
        original_licenses = Counter()

        for record in records:

            words += len(record["text"].split())

            source = record["source"]

            current_typologies = source.get(
                "typology", []
            )

            if isinstance(current_typologies, str):
                current_typologies = [current_typologies]

            for typology in current_typologies:
                typologies[typology] += 1

            if source.get("domain"):
                domains[source["domain"]] += 1

            corpus_licenses[
                safe_key(record["license"]["corpus"].get("name"))
            ] += 1

            original_licenses[
                safe_key(
                    record["license"]["original_source"].get("name")
                )
            ] += 1

        split_statistics[split] = {
            "documents": len(records),
            "words": words,
            "unique_domains": len(domains),
            "corpus_licenses": dict(corpus_licenses),
            "original_source_licenses": dict(
                original_licenses
            ),
        }

        typologies_by_split[split] = dict(typologies)
        domains_by_split[split] = dict(domains)

        total_documents += len(records)
        total_words += words

    duplicate_groups = sum(
        len(group["documents"]) > 1
        for group in groups.values()
    )

    duplicate_excess = sum(
        len(group["documents"]) - 1
        for group in groups.values()
        if len(group["documents"]) > 1
    )

    return {
        "dataset": "LUCI Training Dataset",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "configuration": {
            "seed": SEED,
            "target_ratios": {
                "train": 0.80,
                "validation": 0.10,
                "test": 0.10,
            },
            "stratification": "document_typology",
            "duplicate_grouping": "sha256_normalized_text",
            "duplicate_groups_never_split": True,
        },

        "statistics": {
            "document_count": total_documents,
            "word_count": total_words,
            "unique_text_groups": len(groups),
            "duplicate_groups": duplicate_groups,
            "duplicate_excess_documents": duplicate_excess,
        },

        "splits": split_statistics,

        "typologies_by_split": typologies_by_split,

        "domains_by_split": domains_by_split,

        "strata": stratum_statistics,

        "source": {
            "corpus": "Corpus Carolina",
            "cleaned_manifest_sha256": source_manifest_hash,
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_outputs(assigned: dict, manifest: dict):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_files = []

    try:

        for split, records in assigned.items():

            output_file = OUTPUT_DIR / f"{split}.jsonl"

            temporary_file = output_file.with_suffix(
                ".jsonl.tmp"
            )

            temporary_files.append(
                (temporary_file, output_file)
            )

            with temporary_file.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as file:

                for record in records:

                    file.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ) + "\n"
                    )

        # Registra hashes dos arquivos gerados.
        manifest["output_files_sha256"] = {
            split: sha256_file(
                OUTPUT_DIR / f"{split}.jsonl.tmp"
            )
            for split in SPLITS
        }

        manifest_temp = OUTPUT_DIR / "manifest.json.tmp"

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

        for temporary_file, output_file in temporary_files:
            temporary_file.replace(output_file)

        manifest_temp.replace(
            OUTPUT_DIR / "manifest.json"
        )

    except Exception:

        for temporary_file, _ in temporary_files:
            temporary_file.unlink(missing_ok=True)

        manifest_temp = OUTPUT_DIR / "manifest.json.tmp"
        manifest_temp.unlink(missing_ok=True)

        raise


def main():

    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(
            "Manifesto do corpus limpo não encontrado."
        )

    source_manifest_hash = sha256_file(
        INPUT_MANIFEST
    )

    print("===== CARREGAMENTO =====")

    documents = load_documents()

    print("Documentos carregados:", len(documents))

    print("\n===== PARTICIONAMENTO =====")

    assigned, groups, strata = assign_splits(documents)

    print("\n===== DISTRIBUIÇÃO =====")

    for split in SPLITS:
        print(
            f"{split}: {len(assigned[split])} documentos"
        )

    print("\n===== MANIFESTO =====")

    manifest = calculate_manifest(
        assigned,
        groups,
        strata,
        source_manifest_hash,
    )

    write_outputs(
        assigned,
        manifest,
    )

    print("\n===== CONCLUÍDO =====")

    print(
        json.dumps(
            manifest["statistics"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nSaída:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
