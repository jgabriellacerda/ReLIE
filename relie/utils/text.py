from unidecode import unidecode


def clean_text(text: str) -> str:
    """
    Cleans the given text by converting it to lowercase and removing colons and commas.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    strip_chars = "!#$\'*+:;<=>?[\\]^`{|}~."
    cleaned_text = unidecode(text)
    cleaned_text = text.lower()
    cleaned_text = cleaned_text.strip(strip_chars)
    return cleaned_text


def is_number(text: str) -> bool:
    return text.replace('.', '').replace(',', '').isdecimal()
