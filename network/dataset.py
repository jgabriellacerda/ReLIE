import json
import torch
from torch.utils import data
from relie.config import ReLIEConfig
from utils import xml_parser, Neighbour, candidate
from utils import operations as op
from utils import preprocess

from utils.config import CANDIDATE_DIR, OCR_DIR, OUTPUT_DIR, XML_DIR


def preprocess_dataset(config: ReLIEConfig, split_name='train', vocab: dict | None = None) -> tuple[tuple, dict[str, int]]:
    print("Preprocessed data not available")
    annotation, classes_count = xml_parser.get_data(XML_DIR, split_name)
    print("Class Mapping:", config.CLASS_MAPPING)
    print("Classs counts:", classes_count)
    annotation = candidate.attach_candidate(annotation, CANDIDATE_DIR)
    annotation, gen_vocab = Neighbour.attach_neighbour(annotation, OCR_DIR, vocab_size=config.VOCAB_SIZE)
    vocab = vocab if vocab is not None else gen_vocab
    annotation = op.normalize_positions(annotation)
    data = preprocess.parse_input(annotation, config.CLASS_MAPPING, config.NEIGHBOURS, vocab)
    if split_name == "train":
        (OUTPUT_DIR / "vocab.json").write_text(json.dumps(vocab))
    print("Done !!")
    return data, vocab


class DocumentsDataset(data.Dataset):
    """Stores the annotated documents dataset."""
    
    def __init__(self, data: tuple):
        """ Initialize the dataset with preprocessing """
        self.field_ids, self.candidate_cords, self.neighbours, self.neighbour_cords, self.mask, self.labels = data
    
    def __len__(self):
        return len(self.field_ids)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.field_ids[idx]).type(torch.FloatTensor),
            torch.tensor(self.candidate_cords[idx]).type(torch.FloatTensor),
            torch.tensor(self.neighbours[idx]),
            torch.tensor(self.neighbour_cords[idx]).type(torch.FloatTensor),
            torch.tensor(self.mask[idx]).type(torch.FloatTensor),
            torch.tensor(self.labels[idx]).type(torch.FloatTensor)
        )
