
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path("datasets/sources/carolina/corpus/university_domains")

TARGETS = {
    "UNI000000071a",
    "UNI000000142a",
    "UNI000014764d",
}

XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
NS = {"tei": "http://www.tei-c.org/ns/1.0"}

found = set()

for file in sorted(BASE.glob("UNI*.xml.gz")):
    with gzip.open(file, "rb") as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag != f"{{{NS['tei']}}}TEI":
                continue

            document_id = element.get(XML_ID)

            if document_id not in TARGETS:
                element.clear()
                continue

            found.add(document_id)

            print("\n" + "=" * 90)
            print(f"DOCUMENTO: {document_id} | ARQUIVO: {file.name}")
            print("=" * 90)

            body = element.find(".//tei:body", NS)

            if body is None:
                print("BODY NÃO ENCONTRADO")
            else:
                for index, child in enumerate(body, start=1):
                    tag = child.tag.split("}")[-1]
                    text = " ".join("".join(child.itertext()).split())

                    print(f"\n[{index}] TAG: {tag}")
                    print(text[:1800] or "[SEM TEXTO]")

                    for nested in child.iter():
                        if nested is child:
                            continue

                        nested_tag = nested.tag.split("}")[-1]
                        nested_text = " ".join(
                            "".join(nested.itertext()).split()
                        )

                        print(
                            f"    NESTED: {nested_tag} | "
                            f"{nested_text[:300]}"
                        )

            element.clear()

            if found == TARGETS:
                break

    if found == TARGETS:
        break

print("\nDOCUMENTOS ENCONTRADOS:", sorted(found))
print("DOCUMENTOS AUSENTES:", sorted(TARGETS - found))
