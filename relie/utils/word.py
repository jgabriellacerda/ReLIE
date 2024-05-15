from dataclasses import dataclass
from typing import Iterable

from relie.utils.rect import Rect
from relie.utils.text import clean_text


@dataclass
class Word:
    id: int
    text: str
    start_char: int
    end_char: int
    rect: Rect
    type: str | None
    cleaned_text: str | None = None

    def __post_init__(self):
        self.cleaned_text = self.cleaned_text or clean_text(self.text)

    def __hash__(self) -> int:
        return hash(self.id)
    
    def __str__(self) -> str:
        return self.text






class Utils:

    @staticmethod
    def get_words_inside_area(words: list[Word], area: Rect, threshhold: float = 0.9) -> list[Word]:
        words_inside = []
        intersecting_words = Utils.find_words_intersecting_area(area, words)
        for word in intersecting_words:
            if Utils.intersection_over_box_b(area, word.rect) > threshhold:
                words_inside.append(word)
        return words_inside

    @staticmethod
    def find_words_intersecting_area(area: Rect, words: Iterable[Word]) -> list[Word]:
        intersecting_words = []
        for word in words:
            if word.rect.x2 < area.x1:
                continue
            if word.rect.x1 > area.x2:
                continue
            if word.rect.y1 > area.y2:
                continue
            if word.rect.y2 < area.y1:
                continue
            intersecting_words.append(word)
        return intersecting_words

    @staticmethod
    def intersection_over_box_b(box_a: Rect, box_b: Rect) -> float:
        x_a = max(box_a.x1, box_b.x1)
        y_a = max(box_a.y1, box_b.y1)
        x_b = min(box_a.x2, box_b.x2)
        y_b = min(box_a.y2, box_b.y2)
        box_b_area = (box_b.x2 - box_b.x1) * (box_b.y2 - box_b.y1)
        if float(box_b_area) <= 0:
            return 0
        inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
        return inter_area / float(box_b_area)

    @staticmethod
    def get_words_bounding_box(words: list[Word]) -> Rect:
        x1 = min((word.rect.x1 for word in words))
        y1 = min((word.rect.y1 for word in words))
        x2 = max((word.rect.x2 for word in words))
        y2 = max((word.rect.y2 for word in words))
        width = x2 - x1
        height = y2 - y1
        return Rect(x1, y1, x2, y2, width, height)