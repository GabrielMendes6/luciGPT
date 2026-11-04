
import gzip
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Iterator


TEI_NAMESPACE = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NAMESPACE}


def tag(name: str) -> str:
    return f"{{{TEI_NAMESPACE}}}{name}"


def normalize_text(element: ET.Element | None) -> str | None:
    if element is None:
        return None

    value = " ".join("".join(element.itertext()).split())

    return value or None


def get_element(
    parent: ET.Element | None,
    path: str,
) -> ET.Element | None:
    if parent is None:
        return None

    return parent.find(path, NS)


def get_text(
    parent: ET.Element | None,
    path: str,
) -> str | None:
    return normalize_text(get_element(parent, path))


def get_attribute(
    parent: ET.Element | None,
    path: str,
    attribute: str,
) -> str | None:
    element = get_element(parent, path)

    if element is None:
        return None

    return element.get(attribute)


def extract_authors(
    title_statement: ET.Element | None,
) -> list[str]:
    if title_statement is None:
        return []

    authors = []

    for author in title_statement.findall("tei:author", NS):
        name = normalize_text(author)

        if name:
            authors.append(name)

    return authors


def extract_measures(
    extent: ET.Element | None,
) -> list[dict]:
    if extent is None:
        return []

    measures = []

    for measure in extent.findall("tei:measure", NS):
        measures.append({
            "type": measure.get("type"),
            "unit": measure.get("unit"),
            "quantity": measure.get("quantity"),
        })

    return measures


def extract_catrefs(
    parent: ET.Element | None,
) -> list[dict]:
    if parent is None:
        return []

    result = []

    for item in parent.findall(".//tei:catRef", NS):
        result.append({
            "scheme": item.get("scheme"),
            "target": item.get("target", "").lstrip("#") or None,
        })

    return result


def extract_document(record: ET.Element) -> dict:
    header = get_element(record, "tei:teiHeader")

    if header is None:
        raise ValueError("Documento sem teiHeader.")

    file_desc = get_element(header, "tei:fileDesc")

    if file_desc is None:
        raise ValueError("Documento sem fileDesc.")

    # Identificação do documento no Corpus Carolina.
    title_statement = get_element(
        file_desc,
        "tei:titleStmt",
    )

    document_id = get_text(
        title_statement,
        "tei:title",
    )

    if not document_id:
        raise ValueError("Documento sem identificador.")

    # Metadados de publicação do próprio Carolina.
    publication = get_element(
        file_desc,
        "tei:publicationStmt",
    )

    corpus_license = get_element(
        publication,
        "tei:availability/tei:license",
    )

    # Informações da fonte original.
    bibl_full = get_element(
        file_desc,
        "tei:sourceDesc/tei:biblFull",
    )

    source_file = get_element(
        bibl_full,
        "tei:fileDesc",
    )

    source_title_statement = get_element(
        source_file,
        "tei:titleStmt",
    )

    source_title = get_text(
        source_title_statement,
        "tei:title",
    )

    source_title_element = get_element(
        source_title_statement,
        "tei:title",
    )

    source_media = get_element(
        source_title_element,
        "tei:media",
    )

    source_publication = get_element(
        source_file,
        "tei:publicationStmt",
    )

    source_license = get_element(
        source_publication,
        "tei:availability/tei:license",
    )

    source_profile = get_element(
        bibl_full,
        "tei:profileDesc",
    )

    # Tipologia e categoria do documento.
    source_categories = extract_catrefs(source_profile)

    document_profile = get_element(
        header,
        "tei:profileDesc",
    )

    document_categories = extract_catrefs(
        document_profile
    )

    # Conteúdo textual.
    body = get_element(
        record,
        "tei:text/tei:body",
    )

    if body is None:
        raise ValueError(
            f"Documento {document_id} sem body."
        )

    paragraphs = []

    for paragraph in body.findall(".//tei:p", NS):
        content = normalize_text(paragraph)

        if content:
            paragraphs.append(content)

    full_text = "\n\n".join(paragraphs)

    # Informações de tamanho e processamento.
    corpus_extent = get_element(
        file_desc,
        "tei:extent",
    )

    source_extent = get_element(
        source_file,
        "tei:extent",
    )

    return {
        "document_id": document_id,

        "source": {
            "corpus": "Corpus Carolina",
            "corpus_version": "2.0.1",
            "category": [
                item["target"]
                for item in document_categories
            ],
            "typology": [
                item["target"]
                for item in source_categories
            ],
        },

        "metadata": {
            "title": source_title,
            "authors": extract_authors(
                source_title_statement
            ),
            "language": get_attribute(
                source_profile,
                "tei:langUsage/tei:language",
                "ident",
            ),
            "publication_date": get_text(
                source_publication,
                "tei:date",
            ),
            "publisher": get_text(
                source_publication,
                "tei:publisher",
            ),
            "authority": get_text(
                source_publication,
                "tei:authority",
            ),
            "series": get_text(
                source_file,
                "tei:seriesStmt/tei:title",
            ),
            "url": (
                source_media.get("url")
                if source_media is not None
                else None
            ),
        },

        "licenses": {
            "corpus": {
                "name": normalize_text(corpus_license),
                "url": (
                    corpus_license.get("target")
                    if corpus_license is not None
                    else None
                ),
            },
            "original_source": {
                "name": normalize_text(source_license),
                "url": (
                    source_license.get("target")
                    if source_license is not None
                    else None
                ),
                "status": (
                    get_attribute(
                        source_publication,
                        "tei:availability",
                        "status",
                    )
                ),
            },
        },

        "processing": {
            "authority": get_text(
                publication,
                "tei:authority",
            ),
            "download_date": get_text(
                publication,
                "tei:date[@type='Download']",
            ),
            "extraction_date": get_text(
                publication,
                "tei:date[@type='Extraction']",
            ),
            "corpus_measures": extract_measures(
                corpus_extent
            ),
            "source_measures": extract_measures(
                source_extent
            ),
        },

        "text": full_text,
        "paragraphs": paragraphs,
        "statistics": {
            "paragraph_count": len(paragraphs),
            "character_count": len(full_text),
            "word_count": len(full_text.split()),
        },
    }


def iter_carolina_documents(
    file_path: str | Path,
) -> Iterator[dict]:
    """
    Percorre um arquivo XML comprimido e entrega
    um documento TEI por vez.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix != ".gz":
        raise ValueError(
            "O arquivo precisa estar comprimido em .gz."
        )

    with gzip.open(path, "rb") as xml_file:
        context = ET.iterparse(
            xml_file,
            events=("start", "end"),
        )

        _, root = next(context)

        for event, element in context:
            if (
                event == "end"
                and element.tag == tag("TEI")
            ):
                document = extract_document(element)

                yield document

                # Libera os elementos já processados.
                root.clear()
