from pathlib import Path
from typing import Type
import torch
import cv2
from relie.prediction import ReLIEPredictor
from relie.utils.rect import ImageSize
import generate_tesseract_results
import extract_candidates
import argparse
import os

from pydantic import BaseModel, Field

class Arguments(BaseModel):
    model_path: str = Field(default="output/", description="Directory to load the model")
    image: str = Field(default="sample_dataset/images/sample_1.jpg", description="Path to the document image")
    visualize: bool = Field(True, description="Create images to visualize results")
    cuda: bool = Field(True, description="Use CUDA")

def add_model(parser, model: Type[BaseModel]):
    "Add Pydantic model to an ArgumentParser"
    fields = model.model_fields
    for name, field in fields.items():
        parser.add_argument(
            f"--{name}", 
            dest=name, 
            type=field.annotation, 
            default=field.default,
            help=field.description,
        )

def parse_args() -> Arguments:
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    add_model(parser, Arguments)
    parsed_args = parser.parse_args()
    args = Arguments(**vars(parsed_args))
    return args


def main():
    args = parse_args()
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if not os.path.exists(args.image):
        raise Exception("Image not found")
    device = torch.device('cuda:0' if args.cuda else 'cpu')
    image = cv2.imread(args.image)
    height, width, _ = image.shape
    image_size = ImageSize(width, height)
    ocr_results = generate_tesseract_results.get_tesseract_results(args.image)
    words = generate_tesseract_results.tesseract_results_to_words(ocr_results, image_size)
    candidates, sentences = extract_candidates.get_candidates(words)
    predictor = ReLIEPredictor(Path(args.model_path), device)
    field_to_best_candidate_id, field_to_best_score = predictor.run(words, sentences, candidates, image_size)
    true_candidate_color = (0, 255, 0)
    output_candidates = {}
    output_image = image.copy()
    for field, best_candidate_id in field_to_best_candidate_id.items():
        best_candidate = [sentence for sentence in sentences if sentence.id == best_candidate_id][0]
        output_candidates[field] = " ".join([w.text for w in best_candidate.words])
        cv2.rectangle(
            output_image, 
            (int(best_candidate.rect.x1 * image_size.width), int(best_candidate.rect.y1 * image_size.height)), 
            (int(best_candidate.rect.x2 * image_size.width), int(best_candidate.rect.y2 * image_size.height)),
            true_candidate_color, 
            5
        )
    if args.visualize:
        # cv2.imshow('Visualize', output_image)
        # cv2.waitKey(0)
        cv2.imwrite("output/result.png", output_image)
    print(output_candidates)
    return output_candidates


if __name__ == '__main__':
    main()
