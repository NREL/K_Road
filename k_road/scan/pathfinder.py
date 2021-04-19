import sys
from typing import (
    List,
    Optional,
    Tuple,
)

import pymunk
from pymunk import (
    PointQueryInfo,
    Vec2d,
)

from k_road.constants import Constants
from k_road.entity.entity_category import EntityCategory
from k_road.entity.entity_type import EntityType
from k_road.entity.path import Path


def clean_pqi(pqi, position):
    if pqi.distance > 0.0:
        return pqi
    return PointQueryInfo(pqi.shape, position, 0.0, pqi.gradient)


def find_nearby_paths(
        space: pymunk.Space,
        position: Vec2d,
        radius: float,
        # group_id=None,
        sequence=None,
        previous_path=None,
) -> List[Tuple[pymunk.PointQueryInfo, Path]]:
    """
    Finds nearby paths to the search position, within the search radius.
    :param space: the pymunk space to search in
    :param position: search position
    :param radius: search radius
    :param sequence: if not None, will only return paths with this sequence
    :param current_path: if not None, will exclude this path
    :return: a list of (PointQueryInfo, Path) tuples of the nearby paths
    """

    shape_filter: pymunk.ShapeFilter = pymunk.ShapeFilter(categories=EntityCategory.hint_sensor)

    return [(clean_pqi(pqi, position), pqi.shape.body.entity) for pqi in
            space.point_query(position, radius, shape_filter)
            if pqi.shape is not None
            and pqi.shape.body is not None
            and pqi.shape.body.entity is not None
            and pqi.shape.body.entity.type_ == EntityType.path
            and pqi.shape.body.entity is not previous_path
            and (sequence is None or pqi.shape.body.entity.sequence is sequence)]


def make_path_order_function(previous_path: Path):
    previous_sequence_id = 0
    previous_sequence = None
    sequence_length = sys.maxsize
    if previous_path is not None:
        previous_sequence_id = previous_path.sequence_id
        previous_sequence = previous_path.sequence
        sequence_length = previous_path.sequence_length
        if sequence_length == 0:
            sequence_length = sys.maxsize

    def order_path(e: Tuple[PointQueryInfo, Path]):
        pqi, path = e

        equal_window = 4
        sequence_priority = 1

        sequence_delta = path.sequence_id - previous_sequence_id
        # print('pth: ', abs(path.sequence_id - previous_sequence) % sequence_length, path.sequence_id,
        # previous_sequence,
        #       sequence_length)
        if previous_path is not None and \
                path.sequence is previous_sequence and \
                min(sequence_delta % sequence_length, -sequence_delta % sequence_length) <= equal_window:
            sequence_priority = 0

        # score = -direction.dot(entity.direction) + (1 - pqi.distance / radius)
        # distance = (position - pqi.point).length  # pqi.distance can be wrong...
        return sequence_priority, pqi.distance

    return order_path


def find_next_path(space: pymunk.Space,
                   position: Vec2d,
                   radius: float = Constants.lane_width,
                   sequence=None,
                   previous_path: Optional[Path] = None,
                   scan_window=32,
                   offset=0,
                   ) -> Tuple[Optional[PointQueryInfo], Optional[Path]]:
    sequence = previous_path.sequence if sequence is None else sequence
    sequence_index = offset + previous_path.sequence_id
    best = None
    for delta in range(scan_window):
        i = delta + sequence_index
        if i >= len(sequence):
            break

        candidate = sequence[i]
        bail = True
        for shape in candidate.body.shapes:
            score = shape.point_query(position)
            if best is None or best[0] > score[0]:
                best = score
                bail = False
            else:
                break
        # if bail:
        #     break

    if best is not None:
        return clean_pqi(best[1], position), best[1].shape.body.entity
    return None, None


def find_best_path_hint(
        space: pymunk.Space,
        position: Vec2d,
        radius: float = Constants.lane_width,
        sequence=None,
        previous_path: Optional[Path] = None,
        scan_window=32,
) -> Tuple[Optional[PointQueryInfo], Optional[Path]]:
    # FIXME: Sometimes these make some jumps whereas find_best_path doesn't
    if previous_path is not None:
        pqi, path = find_next_path(space, position, radius, sequence, previous_path, scan_window)
        if path is not None:
            return pqi, path

    return find_best_path(space, position, radius, sequence)


def find_best_path(
        space: pymunk.Space,
        position: Vec2d,
        radius: float = Constants.lane_width,
        sequence=None,
        # previous_path: Optional[Path] = None,
) -> Tuple[Optional[PointQueryInfo], Optional[Path]]:
    paths: List[Tuple[PointQueryInfo, Path]] = find_nearby_paths(space, position, radius, sequence=sequence)

    if len(paths) == 0:
        return None, None

    order_path = make_path_order_function(None)
    return min(paths, key=order_path)
