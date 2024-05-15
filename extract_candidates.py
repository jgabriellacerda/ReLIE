from dateparser.search import search_dates

from relie.document import Candidate, Sentence, Word
from typing import NamedTuple
import regex as re

from relie.utils.word import Utils, Word


def find_word_with_start_char(words: list[Word], start_char: int) -> Word:
    """
    Finds the word with the given start character.

    Args:
        words (List[Word]): A list of words to search for the word in.
        start_char (int): The start character of the word to find.

    Returns:
        Word: The word with the given start character.
    """
    filtered = list(filter(lambda t: start_char == t.start_char, words))
    if len(filtered) != 1:
        raise ValueError()
    return filtered[0]

def find_word_with_end_char(words: list[Word], end_char: int) -> Word:
    """
    Finds a word with the given end character.

    Args:
        words (List[Word]): A list of words to search through.
        end_char (int): The end character of the word to find.

    Returns:
        Word: The word with the given end character.

    Raises:
        ValueError: If there is not exactly one word with the given end character.
    """
    filtered = list(filter(lambda t: end_char == t.end_char, words))
    if len(filtered) != 1:
        raise ValueError()
    return filtered[0]

def find_in_text(
        text: str,
        text_list: list[str],
        words: list[Word],
    ) -> list[tuple[int, ...]]:
    """
    Finds all occurrences of the text in the given text.

    Args:
        text (str): The text to search for.
        word_to_image_rect (List[int]): A list of integers representing the indices of the image rectangles.
        words (List[Word]): A list of words to search for text in.
        same_image_rect (bool): Whether to search for text in the same image rectangle or not.

    Returns:
        List[Tuple[int, ...]]: A list of tuples representing the indices of the matched words.
    """
    patterns = [fr"(\s|^){pattern}(\s|$)" for pattern in text_list]
    matches_ids: list[tuple[int, ...]] = []
    for pattern in patterns:
        re_matches = re.finditer(pattern, text, overlapped=True)
        for match in re_matches:
            start_char = match.start() + 1 if match.group()[0] == " " else match.start()
            end_char = match.end() - 1 if match.group()[-1] == " " else match.end()
            start_word = find_word_with_start_char(words, start_char)
            end_word = find_word_with_end_char(words, end_char)
            match_ids = tuple(range(start_word.id, end_word.id + 1))
            matches_ids.append(match_ids)
    return matches_ids

def append_without_intersections(
    new_matches: list[tuple[int, ...]],
    matches_ids: list[tuple[int, ...]],
    matched_ids: set[int],
) -> tuple[list[tuple[int, ...]], set[int]]:
    """
    Appends new matches to the list of matches without intersections.

    Args:
        new_matches (List[Tuple[int, ...]]): A list of tuples representing the indices of the new matches.
        matches_ids (List[Tuple[int, ...]]): A list of tuples representing the indices of the matches.
        matched_ids (Set[int]): A set of integers representing the indices of the matched words.

    Returns:
        Tuple[List[Tuple[int, ...]], Set[int]]: A tuple containing the updated list of matches and the updated set of matched words.
    """
    for match_ids in new_matches:
        if set(match_ids).intersection(matched_ids):
            continue
        matched_ids.update(match_ids)
        matches_ids.append(match_ids)
    return matches_ids, matched_ids


def search_invoice_nums(words: list[Word]) -> list[Sentence]:
    invoice_no_re = r'^[0-9a-zA-Z-:]+$'
    sentences: list[Sentence] = []
    for word in words:
        if not re.search('\d', word.text):
            continue
        if len(word.text) < 3:
            continue
        result = re.findall(invoice_no_re, word.text)
        if result:
            word.type = "<INVOICE_NUMBER>"
            sentences.append(Sentence(-1, (word,), word.rect, "<INVOICE_NUMBER>"))
    return sentences


def search_document_dates(words: list[Word]) -> list[Sentence]:
    word_texts = [str(word) for word in words]
    all_text = " ".join(word_texts)
    matches = search_dates(all_text)
    match_texts = [m[0] for m in matches]
    all_matched_ids: set[int] = set()
    matches_ids: list[tuple[int, ...]] = []
    new_matches = find_in_text(all_text, match_texts, words)
    matches_ids, all_matched_ids = append_without_intersections(new_matches, matches_ids, all_matched_ids)
    sentences: list[Sentence] = []
    for match in matches_ids:
        sentence_words: list[Word] = []
        for word_id in match:
            word = words[word_id]
            assert word_id == word.id
            sentence_words.append(word)
        if all([w.type is None for w in sentence_words]):
            for word in sentence_words:
                word.type = "<DATE>"
            sentence = Sentence(-1, tuple(sentence_words), Utils.get_words_bounding_box(sentence_words), "<DATE>")
            sentences.append(sentence)
    return sentences


def search_amounts(words: list[Word]) -> list[Sentence]:
    amount_re = r"\$?([0-9]*,)*[0-9]{3,}(\.[0-9]+)?"
    sentences: list[Sentence] = []
    for word in words:
        if word.type is not None:
            continue
        if not re.search(amount_re, word.text):
            continue
        try:
            formatted_word = re.sub(r'[$,]', '', word.text)
            float(formatted_word)
            word.type = "<AMOUNT>"
            sentences.append(Sentence(-1, (word,), word.rect, "<AMOUNT>"))
        except ValueError:
            continue
    return sentences


def get_candidates(words: list[Word]) -> tuple[list[Candidate], list[Sentence]]:
    all_sentences: list[Sentence] = []
    all_sentences.extend(search_document_dates(words))
    all_sentences.extend(search_amounts(words))
    all_sentences.extend(search_invoice_nums(words))

    sentence_count = 0
    for sentence in all_sentences:
        sentence.id = sentence_count
        sentence_count += 1

    field_to_sentence_type = {
        "invoice_no": "<INVOICE_NUMBER>",
        "invoice_date": "<DATE>",
        "total": "<AMOUNT>",
    }

    candidate_count = 0
    candidates: list[Candidate] = []
    for field, sentence_type in field_to_sentence_type.items():
        type_sentences = [sentence for sentence in all_sentences if sentence.type == sentence_type]
        for sentence in type_sentences:
            candidate = Candidate(candidate_count, field, sentence)
            candidates.append(candidate)
            candidate_count += 1

    return candidates, all_sentences
