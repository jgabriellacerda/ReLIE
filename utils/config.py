from pathlib import Path

CLASS_MAPPING = {
    "invoice_no": 0,
    "invoice_date": 1,
    "total": 2,
}
NEIGHBOURS = 5
HEADS = 4
EMBEDDING_SIZE = 16
VOCAB_SIZE = 1000
BATCH_SIZE = 2
EPOCHS = 10
LR = 0.001
FL_GAMMA = 5

current_directory = Path.cwd()
XML_DIR = current_directory / "sample_dataset" / "xmls"
OCR_DIR = current_directory / "sample_dataset" / "tesseract_results"
IMAGE_DIR = current_directory / "sample_dataset" / "images"
CANDIDATE_DIR = current_directory / "sample_dataset" / "candidates"
SPLIT_DIR = current_directory / "sample_dataset" / "split"
OUTPUT_DIR = current_directory / "output"

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)