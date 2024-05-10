from typing import Any, Dict

import cv2
import numpy as np

from relie.utils.rect import SizesRect


def draw_rects_alpha(
    img: np.ndarray,
    name_to_rect: Dict[str, SizesRect],
    color=(255, 0, 0),
    alpha=0.5,
    text_bellow: bool = False
) -> np.ndarray:
    new_img = img.copy()
    # Initialize blank mask image of same dimensions for drawing the shapes
    shapes: np.ndarray[Any, np.dtype[np.generic]] = np.zeros_like(img, np.uint8)
    mask: np.ndarray[Any, np.dtype[np.generic]] = np.zeros_like(img, np.uint8)
    for name, rect in name_to_rect.items():
        p1 = (rect.x, rect.y)
        x2: int = rect.x + rect.width
        y2: int = rect.y + rect.height
        new_img = cv2.rectangle(new_img, p1, (x2, y2), color, thickness=2)
        mask = cv2.rectangle(mask, p1, (x2, y2), (255, 255, 255), thickness=2)
        shapes = cv2.rectangle(shapes, p1, (x2, y2), color, thickness=-1)
        mask = cv2.rectangle(mask, p1, (x2, y2), (255, 255, 255), thickness=-1)
        if text_bellow:
            new_img = cv2.putText(new_img, name, (rect.x, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            mask = cv2.putText(mask, name, (rect.x, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            new_img = cv2.putText(new_img, name, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            mask = cv2.putText(mask, name, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Generate output by blending image with shapes image, using the shapes
    # images also as mask to limit the blending to those parts
    # cv2.imwrite("tmp/shapes.png", shapes)
    out = new_img.copy()
    mask = mask.astype(bool)
    # cv2.imwrite("tmp/mask.png", np.uint8(mask) * 255)
    out[mask] = cv2.addWeighted(new_img, alpha, shapes, 1 - alpha, 0)[mask]
    return out


def draw_rect_dict(
    img: np.ndarray,
    rects: Dict[str, SizesRect],
    color: tuple = (255, 0, 0)
) -> np.ndarray:
    new_img = img.copy()
    for name, rect in rects.items():
        p1 = (rect.x, rect.y)
        p2 = (rect.x + rect.width, rect.y + rect.height)
        new_img = cv2.rectangle(new_img, pt1=p1, pt2=p2, color=color, thickness=2)
        new_img = cv2.putText(new_img, name, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,)
    return new_img
