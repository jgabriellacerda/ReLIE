

from relie.utils.rect import ImageSize, NormRect, PointsRect, Rect, SizesRect


class RectAdapter:

    @staticmethod
    def sizes_to_points(rect: SizesRect) -> PointsRect:
        return PointsRect(x1=rect.x, y1=rect.y, x2=rect.x+rect.width, y2=rect.y+rect.height)

    @staticmethod
    def points_to_sizes(rect: PointsRect) -> SizesRect:
        return SizesRect(x=rect.x1, y=rect.y1, width=rect.x2-rect.x1, height=rect.y2-rect.y1)

    @staticmethod
    def sizes_to_norm(sizes_rect: SizesRect, image_size: ImageSize) -> NormRect:
        return NormRect(
            x=sizes_rect.x / image_size.width,
            y=sizes_rect.y / image_size.height,
            width=sizes_rect.width / image_size.width,
            height=sizes_rect.height / image_size.height,
        )

    @staticmethod
    def sizes_to_default(sizes_rect: SizesRect, image_size: ImageSize) -> Rect:
        x1 = sizes_rect.x / image_size.width
        y1 = sizes_rect.y / image_size.height
        width = sizes_rect.width / image_size.width
        height = sizes_rect.height / image_size.height
        x2 = x1 + width
        y2 = y1 + height
        return Rect(x1, y1, x2, y2, width, height)

    @staticmethod
    def default_to_sizes(rect: Rect, image_size: ImageSize) -> SizesRect:
        x1 = int(rect.x1 * image_size.width)
        y1 = int(rect.y1 * image_size.height)
        width = int(rect.width * image_size.width)
        height = int(rect.height * image_size.height)
        return SizesRect(x1, y1, width, height)

    @staticmethod
    def norm_to_sizes(rect: NormRect, image_size: ImageSize) -> SizesRect:
        return SizesRect(
            int(rect.x * image_size.width),
            int(rect.y * image_size.height),
            int(rect.width * image_size.width),
            int(rect.height * image_size.height)
        )

    @staticmethod
    def norm_to_points(rect: NormRect, image_size: ImageSize) -> PointsRect:
        return PointsRect(
            int(rect.x * image_size.width),
            int(rect.y * image_size.height),
            int((rect.x + rect.width) * image_size.width),
            int((rect.y + rect.height) * image_size.height)
        )


def bb_intersection_over_union(box_a: PointsRect, box_b: PointsRect) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = float(box_a_area + box_b_area - inter_area)
    if union_area == 0:
        return 0
    return inter_area / union_area


def bb_intersection_over_box_b(box_a: PointsRect, box_b: PointsRect) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    if float(box_b_area) <= 0:
        return 0
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    return inter_area / float(box_b_area)
