from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from typing import Literal, NamedTuple

import numpy as np
import torch

from relie.utils.rect import ImageSize, Rect
from relie.utils.text import is_number
from relie.utils.word import  Utils, Word


@dataclass
class Sentence:
    id: int
    words: tuple[Word, ...]
    rect: Rect
    type: str | None = None

    def __hash__(self) -> int:
        return hash(self.id)
    
    def __str__(self) -> str:
        return " ".join([word.text for word in self.words])


@dataclass
class Candidate:
    id: int
    field: str
    sentence: Sentence
    neighbours: list[Word] | None = None
    label: Literal[0, 1] | None = None


class Document:

    def __init__(
        self,
        words: list[Word],
        sentences: list[Sentence],
        candidates: list[Candidate],
        image_size: ImageSize,
        class_mapping: dict[str, int],
        vocab: dict[str, int],
        max_neighbours: int,
        device: torch.device,
        sort_neighbours: bool = False,
    ) -> None:
        self.words = words
        self.sentences = sentences
        self.candidates = candidates
        self.class_mapping = class_mapping
        self.vocab = vocab
        self.max_neighbours = max_neighbours
        self.device = device
        self.size_proportion = image_size.width / image_size.height
        self.sort_neighbours = sort_neighbours


    def __attach_neighbours(self) -> None:
        for candidate in self.candidates:
            if candidate.neighbours is not None:
                continue
            neighbours = self.__search_neighbours(candidate.sentence)
            candidate.neighbours = neighbours

    @lru_cache(1024)
    def __search_neighbours(self, sentence: Sentence) -> list[Word]:
        # sentence = self.sentences[sentence_id]
        x_offset = 1
        y_offset = 0.333
        candidate_word_ids = set([w.id for w in sentence.words])
        valid_neighbour_words = [word for word in self.words if word.id not in candidate_word_ids]
        neighbours = []
        search_area = self.__get_search_area(sentence, x_offset, y_offset)
        neighbours = Utils.find_words_intersecting_area(search_area, valid_neighbour_words)
        if self.sort_neighbours:
            neighbours = self.__sort_neighbours(sentence, neighbours)
        return neighbours

    def __get_search_area(self, sentence: Sentence, x_offset: float, y_offset: float) -> Rect:
        x1 = sentence.rect.x1 - x_offset
        y1 = sentence.rect.y1 - y_offset
        x2 = sentence.rect.x2 + x_offset
        y2 = sentence.rect.y2 + y_offset
        width = x2 - x1
        height = y2 - y1
        return Rect(x1, y1, x2, y2, width, height)

    def __sort_neighbours(
        self,
        sentence: Sentence,
        neighbours: list[Word],
    ) -> list[Word]:
        """ Sort neighbours by candidate distance to prevent discarding relevant neighbours """
        class NeighbourDist(NamedTuple):
            neighbour: Word
            distance: float
        neighbour_tuples: list[NeighbourDist] = []
        for neighbour in neighbours:
            distance = self.__get_distance(sentence.rect, neighbour.rect)
            neighbour_tuples.append(NeighbourDist(neighbour=neighbour, distance=distance))
        neighbour_tuples.sort(key=lambda x: x.distance, reverse=False)
        neighbours = [neighbour_tuple.neighbour for neighbour_tuple in neighbour_tuples]
        return neighbours

    # @lru_cache(1024)
    def __get_distance(self, rect1: Rect, rect2: Rect, x_weight: float = 1, y_weight: float = 3) -> float:
        # Corrects the distortion caused by normalization on pages with width != height
        x_distance = (rect1.centroid.x - rect2.centroid.x) * self.size_proportion * x_weight
        y_distance = (rect1.centroid.y - rect2.centroid.y) * y_weight
        distance = sqrt(
            x_distance**2
            + y_distance**2
        )
        return distance

    def to_tensors(self):
        self.__attach_neighbours()
        field_ids = []
        candidate_cords = []
        neighbour_tokens = []
        neighbour_cords = []
        n_classes = len(self.class_mapping)
        for candidate in self.candidates:
            _neighbour_tokens, _neighbour_cords = self.__parse_neighbours(
                tuple([n.id for n in candidate.neighbours]), candidate.sentence.rect)
            field_ids.append(np.eye(n_classes)[self.class_mapping[candidate.field]])
            candidate_cords.append([candidate.sentence.rect.centroid.x, candidate.sentence.rect.centroid.y])
            neighbour_tokens.append(_neighbour_tokens)
            neighbour_cords.append(_neighbour_cords)
        return (
            torch.tensor(np.array(field_ids), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(candidate_cords), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(neighbour_tokens), dtype=torch.int64, device=self.device),
            torch.tensor(np.array(neighbour_cords), dtype=torch.float32, device=self.device),
        )

    @lru_cache(1024)
    def __parse_neighbours(self, neighbour_ids: tuple[int], candidate_rect: Rect):
        neighbours = [self.words[id] for id in neighbour_ids]
        neighbour_tokens: list[int] = []
        neighbour_cords: list[list[float]] = []
        for neighbour in neighbours:
            token = self.__get_word_token(neighbour)
            neighbour_tokens.append(token)
            # Neighbour centroid relative to the candidate centroid
            neighbour_x = neighbour.rect.centroid.x - candidate_rect.centroid.x
            neighbour_y = neighbour.rect.centroid.y - candidate_rect.centroid.y
            neighbour_cords.append([neighbour_x, neighbour_y])
        len_neighbours = len(neighbours)
        if len_neighbours != self.max_neighbours:
            if len_neighbours > self.max_neighbours:
                neighbour_tokens = neighbour_tokens[:self.max_neighbours]
                neighbour_cords = neighbour_cords[:self.max_neighbours]
            else:
                neighbour_tokens.extend([self.vocab['<PAD>']] * (self.max_neighbours - len_neighbours))
                neighbour_cords.extend([[0., 0.]] * (self.max_neighbours - len_neighbours))
        return neighbour_tokens, neighbour_cords

    @lru_cache(1024)
    def __get_word_token(self, word: Word):
        """
        Retrieves the token corresponding to the given text based on the provided vocabulary.

        Args:
            text (str): The input text.
            text_type (str): The type of the text.
            vocabulary (dict[str, int]): The vocabulary containing the mapping of words to tokens.

        Returns:
            int: The token corresponding to the given text.
        """
        cleaned_text = word.cleaned_text
        if word.type in self.vocab:
            return self.vocab[word.type]
        if cleaned_text in self.vocab:
            return self.vocab[cleaned_text]
        if is_number(word.text):
            return self.vocab['<NUMBER>']
        return self.vocab['<RARE>']
