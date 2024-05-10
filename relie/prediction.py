

import json
from pathlib import Path
import numpy as np
import torch
from torch import Tensor

from relie.config import ReLIEConfig
from relie.document import Candidate, Document
from network.model import Model
from relie.utils.rect import ImageSize
from relie.utils.word import TypedSizesWord


class ReLIEPredictor():
    def __init__(
            self,
            model_path: Path,
            language: str,
            device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),

    ):
        self.model_path = model_path
        self.language = language
        self.device = device
        self.relie, self.vocab, self.class_mapping = self.get_model()

    def get_model(self) -> tuple[Model, dict[str, int], dict[str, int]]:
        self.config = ReLIEConfig.from_path(self.model_path / "config.json")
        vocab: dict = json.loads((self.model_path / "vocab.json").read_text())
        class_mapping: dict = self.config.CLASS_MAPPING
        model = Model(
            len(vocab),
            len(self.config.CLASS_MAPPING),
            self.config.EMBEDDING_SIZE,
            self.config.HEADS,
            self.config.NEIGHBOURS,
            # self.config.DROPOUT
        )
        map_location = torch.device('cpu') if not torch.cuda.is_available() else None
        with open(self.model_path / "model.pth", "rb") as model_file:
            model.load_state_dict(torch.load(model_file, map_location=map_location))
        relie = model.to(self.device)
        return relie, vocab, class_mapping

    def run(
        self,
        original_words: list[TypedSizesWord],
        typed_words: list[TypedSizesWord],
        field_to_candidates_indexes: dict[str, list[int]],
        image_size: ImageSize,
    ) -> tuple[dict[str, int], dict[str, float]]:
        document = Document(
            original_words=original_words,
            typed_words=typed_words,
            field_to_candidates_indexes=field_to_candidates_indexes,
            image_size=image_size,
            class_mapping=self.class_mapping,
            vocab=self.vocab,
            max_neighbours=self.config.NEIGHBOURS,
            device=self.device,
        )
        data_tensors = document.to_tensors()
        output_scores, output_class_id = self.__predict(data_tensors)
        return self.__postprocess(output_scores, output_class_id, document.candidates)

    def __postprocess(
        self,
        output_scores: np.ndarray,
        output_class_ids: np.ndarray,
        candidates: list[Candidate],
    ) -> tuple[dict[str, int], dict[str, float]]:
        field_to_best_candidate_id = {}
        field_to_best_score = {}
        class_id_to_field = {class_id: field for field, class_id in self.class_mapping.items()}
        for class_id in np.unique(output_class_ids):
            field = class_id_to_field[class_id]
            field_scores = output_scores[np.where(output_class_ids == class_id)]
            best_input_id = np.argmax(field_scores)
            best_score = np.max(field_scores)
            field_candidates = list(filter(lambda c: c.field == field, candidates))
            best_candidate = field_candidates[best_input_id]
            field_to_best_candidate_id[field] = best_candidate.sentence.id
            field_to_best_score[field] = best_score
        return field_to_best_candidate_id, field_to_best_score

    def __predict(self, data: tuple[Tensor, Tensor, Tensor, Tensor]) -> tuple[np.ndarray, np.ndarray]:
        field_ids, candidate_cords, neighbours, neighbour_cords = data
        field_ids = field_ids.to(self.device)
        candidate_cords = candidate_cords.to(self.device)
        neighbours = neighbours.to(self.device)
        neighbour_cords = neighbour_cords.to(self.device)
        # field_ids.detach().to('cpu').numpy()
        with torch.no_grad():
            self.relie.eval()
            val_outputs = self.relie(field_ids, candidate_cords, neighbours, neighbour_cords, None)
        field_idx_candidate = np.argmax(field_ids.detach().to('cpu').numpy(), axis=1)
        val_outputs = val_outputs.to('cpu').numpy()
        return val_outputs, field_idx_candidate
