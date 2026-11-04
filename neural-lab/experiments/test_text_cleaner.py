
from datasets.text_cleaner import clean_text


def test_html_image():
    text = (
        '<img src="imagem.jpg" '
        'alt="Descrição de uma imagem" />'
    )

    result = clean_text(text)

    assert result.cleaned_text == "Descrição de uma imagem"


def test_html_link():
    text = (
        '<a href="https://example.com">'
        'Texto do link'
        '</a>'
    )

    result = clean_text(text)

    assert result.cleaned_text == "Texto do link"


def test_url_preservation():
    text = (
        "Consulte <https://example.com/artigo> "
        "para mais informações."
    )

    result = clean_text(text)

    assert "<https://example.com/artigo>" in result.cleaned_text


def test_control_characters():
    text = "Referência \x7f Centro de Saúde"

    result = clean_text(text)

    assert "\x7f" not in result.cleaned_text
    assert "Referência " in result.cleaned_text
    assert "Centro de Saúde" in result.cleaned_text


def test_unicode_accents():
    text = "Inteligência artificial, coração e ação."

    result = clean_text(text)

    assert result.cleaned_text == text


def test_line_breaks():
    text = "Primeiro parágrafo\n\n\nSegundo parágrafo"

    result = clean_text(text)

    assert result.cleaned_text == (
        "Primeiro parágrafo\n\nSegundo parágrafo"
    )


def test_empty_text():
    result = clean_text("   ")

    assert result.cleaned_text == ""


def test_invalid_input():
    try:
        clean_text(123)
    except TypeError:
        pass
    else:
        raise AssertionError("Deveria rejeitar conteúdo que não seja string.")


if __name__ == "__main__":
    tests = [
        test_html_image,
        test_html_link,
        test_url_preservation,
        test_control_characters,
        test_unicode_accents,
        test_line_breaks,
        test_empty_text,
        test_invalid_input,
    ]

    for test in tests:
        test()
        print(f"[OK] {test.__name__}")

    print(f"\n{len(tests)} testes concluídos.")
