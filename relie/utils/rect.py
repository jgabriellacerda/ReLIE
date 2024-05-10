from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple


class Point(NamedTuple):
    x: float
    y: float


class ImageSize(NamedTuple):
    width: int
    height: int


class NormRect(NamedTuple):
    """ Rectangle with values normalized based on image size.
        Values in range 0 to 1.
    """
    x: float
    y: float
    width: float
    height: float

    def __str__(self) -> str:
        return str((round(self.x, 2), round(self.y, 2), round(self.width, 2), round(self.height, 2)))


class PointsRect(NamedTuple):
    """ Rectangle points in integer pixels. """
    x1: int
    y1: int
    x2: int
    y2: int


class SizesRect(NamedTuple):
    """ Rectangle sizes in integer pixels. """
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Rect:
    """ Normalized rectangle. """
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float

    @cached_property
    def centroid(self):
        """ Normalized centroid. """
        return Point(self.x1 + self.width / 2, self.y1 + self.height / 2)
