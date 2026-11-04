
import html
import re
import unicodedata

from dataclasses import dataclass, field
from html.parser import HTMLParser

ANGLE_URL_PATTERN = re.compile(
    r"<https?://[^<>\s]+>",
    re.IGNORECASE,
)

CONTROL_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)

URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+",
    re.IGNORECASE,
)


@dataclass
class CleaningResult:
    original_text: str
    cleaned_text: str
    changes: dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.original_text != self.cleaned_text


class TextHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)

        self.parts = []
        self.removed_tags = 0
        self.extracted_alt = 0
        self.image_descriptions = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.removed_tags += 1

        if tag == "img":
            alt = attrs.get("alt", "").strip()

            if alt:
                index = len(self.image_descriptions)
                placeholder = f"\x00LUCI_ALT_{index}\x00"

                self.image_descriptions.append((placeholder, alt))
                self.parts.append(placeholder)

        elif tag in {
            "br", "p", "div", "li", "section",
            "article", "h1", "h2", "h3", "h4"
        }:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        self.removed_tags += 1

        if tag in {
            "p", "div", "li", "section",
            "article", "h1", "h2", "h3", "h4"
        }:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)

    def get_text(self):
        text = "".join(self.parts)

        for placeholder, description in self.image_descriptions:
            without_placeholder = text.replace(placeholder, "")

            if description.casefold() in without_placeholder.casefold():
                replacement = ""
            else:
                replacement = f" {description} "
                self.extracted_alt += 1

            text = text.replace(placeholder, replacement)

        return text

def contains_html_markup(text: str) -> bool:
    """
    Identifica marcação HTML sem classificar URLs entre <> como tags.
    """

    return bool(
        re.search(
            r"<\s*/?\s*[a-zA-Z][^>]*>",
            text,
        )
    )


def extract_html_text(text):
    """
    Extrai texto de HTML, normalizando aspas tipográficas
    somente dentro das tags.
    """

    def normalize_tag_quotes(match):
        tag = match.group(0)

        return (
            tag.replace("\u201c", '"')
               .replace("\u201d", '"')
               .replace("\u2018", "'")
               .replace("\u2019", "'")
        )

    normalized_text = re.sub(
        r"<[^<>]*>",
        normalize_tag_quotes,
        text,
    )

    quote_changes = sum(
        text.count(char)
        for char in ("\u201c", "\u201d", "\u2018", "\u2019")
    ) - sum(
        normalized_text.count(char)
        for char in ("\u201c", "\u201d", "\u2018", "\u2019")
    )

    parser = TextHTMLParser()
    parser.feed(normalized_text)
    parser.close()

    return parser.get_text(), {
        "html_tags_removed": parser.removed_tags,
        "image_descriptions_extracted": parser.extracted_alt,
        "html_quotes_normalized": quote_changes,
    }


def clean_text(text: str) -> CleaningResult:
    if not isinstance(text, str):
        raise TypeError("O conteúdo precisa ser uma string.")

    original = text
    changes: dict[str, int] = {}

    # 1. Decodifica entidades HTML.
    decoded = html.unescape(text)

    if decoded != text:
        changes["html_entities_decoded"] = 1

    text = decoded

    # 2. Protege URLs entre < > antes de interpretar HTML.
    protected_urls = []

    def protect_url(match):
        index = len(protected_urls)
        protected_urls.append(match.group(0))

        return f"LUCIURLPLACEHOLDER{index}TOKEN"

    text = ANGLE_URL_PATTERN.sub(protect_url, text)

    if protected_urls:
        changes["angle_urls_preserved"] = len(protected_urls)

    # 3. Remove marcação HTML sem destruir URLs.
    if contains_html_markup(text):
        text, html_changes = extract_html_text(text)
        changes.update(html_changes)

    # 4. Restaura as URLs originais.
    for index, url in enumerate(protected_urls):
        placeholder = f"LUCIURLPLACEHOLDER{index}TOKEN"
        text = text.replace(placeholder, url)

    # 5. Remove caracteres de controle.
    control_matches = CONTROL_PATTERN.findall(text)

    if control_matches:
        changes["control_characters_removed"] = len(
            control_matches
        )

        text = CONTROL_PATTERN.sub("", text)

    # 6. Normaliza Unicode sem eliminar acentos.
    normalized = unicodedata.normalize("NFC", text)

    if normalized != text:
        changes["unicode_normalized"] = 1

    text = normalized

    # 7. Normaliza espaços horizontais.
    normalized_spaces = re.sub(
        r"[ \t\f\v]+",
        " ",
        text,
    )

    if normalized_spaces != text:
        changes["horizontal_whitespace_normalized"] = 1

    text = normalized_spaces

    # 8. Normaliza espaços nas linhas.
    lines = [
        line.strip()
        for line in text.splitlines()
    ]

    text = "\n".join(lines)

    # 9. Reduz quebras de linha excessivas.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 10. Remove espaços externos.
    text = text.strip()

    return CleaningResult(
        original_text=original,
        cleaned_text=text,
        changes=changes,
    )
