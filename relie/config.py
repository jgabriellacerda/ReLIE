

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Type


@dataclass
class ReLIEConfig:
    CLASS_MAPPING: dict[str, int]
    NEIGHBOURS: int
    HEADS: int
    EMBEDDING_SIZE: int
    VOCAB_SIZE: int
    BATCH_SIZE: int
    EPOCHS: int
    LR: float
    DROPOUT: float
    FL_GAMMA: float
    WEIGHTS_ALPHA: float

    dict = asdict

    @staticmethod
    def from_path(file_path: Path) -> 'ReLIEConfig':
        config_dict = json.loads(file_path.read_text())
        return ReLIEConfig(**config_dict)
