import bisect
import itertools
from typing import (
    Optional,
    Union,
)

import rtree as rtree
from pymunk import Vec2d

from k_road.constants import Constants
from k_road.entity import entity_factory
from k_road.entity.entity import Entity
from k_road.entity.entity_category import EntityCategory
from k_road.entity.entity_type import EntityType
from k_road.entity.linear_entity import LinearEntity
from k_road.travel_direction import TravelDirection


class RoadBuilder:

    def __init__(
            self,
            parent: 'KRoadProcess',
            start_point: Optional[Vec2d],
            center_points: [Vec2d],
            end_point: Optional[Vec2d],
            lane_layout: [Union[EntityType, float, TravelDirection]]
    ):
        """
            Creates a roadway with lanes, curbs, paths, etc
        """
        self.parent = parent
        start_point: Vec2d = center_points[0] - (center_points[1] - center_points[0]).normalized() \
            if start_point is None else start_point

        end_point: Vec2d = center_points[-1] + (center_points[-1] - center_points[-2]).normalized() \
            if end_point is None else end_point

        self.lane_layout: [Union[EntityType, float, TravelDirection]] = lane_layout

        self.segment_points: [Vec2d] = [start_point, *center_points, end_point]

        # build a segment point index
        self.segment_point_index: rtree.index.Index = rtree.index.Index(interleaved=False)
        for i, point in enumerate(self.segment_points):
            self.segment_point_index.insert(i, (point.x, point.x, point.y, point.y))

        self.segment_vectors: [Vec2d] = \
            list([self.segment_points[i + 1] - self.segment_points[i]
                  for i in range(len(self.segment_points) - 1)])

        self.segment_directions: [Vec2d] = list([v.normalized() for v in self.segment_vectors])
        self.segment_normals: [Vec2d] = list([v.perpendicular() for v in self.segment_directions])

        self.slices = \
            [(self.segment_normals[i] + self.segment_normals[i + 1]).normalized()
             for i in range(len(self.segment_normals) - 1)]
        self.slices = list(
            [self.slices[i] / (self.slices[i].dot(self.segment_normals[i]) + 0.000000001)
             for i in range(len(self.slices))])

        self.segment_lengths: [float] = list([vector.length for vector in self.segment_vectors])

        self.segment_distances: [float] = list([0, *itertools.accumulate(self.segment_lengths)])

        self.lane_center_offsets: [float] = []
        self.num_lanes: int = 0
        # self.length: float = 0

        self.entities: [[Entity]] = []

        last_type: EntityType = EntityType.null
        spacing: Optional[float] = None
        offset: float = 0.0
        self.offsets: [float] = []
        travel_direction: TravelDirection = TravelDirection.forward

        for i, element in enumerate(self.lane_layout):
            # group_id = 'r_' + str(i)

            if isinstance(element, float):
                spacing = element

            elif isinstance(element, TravelDirection):
                travel_direction = element

            elif isinstance(element, EntityType):
                if element.category == EntityCategory.lane:
                    pass  # create a lane

                elif element.category == EntityCategory.lane_marking:
                    if last_type.category == EntityCategory.lane:
                        spacing = Constants.lane_width if spacing is None else spacing
                        lane_center_offset: float = offset + spacing / 2
                        self._place_entities(EntityType.path, lane_center_offset, travel_direction)
                        self.num_lanes = self.num_lanes + 1
                        self.lane_center_offsets.append(lane_center_offset)
                    else:
                        spacing = 0 if spacing is None else spacing

                    # create a lane marking
                    self._place_entities(element, offset + spacing, travel_direction)

                elif element == EntityType.curb:
                    spacing = Constants.gutter_width if spacing is None else spacing

                    self._place_entities(element, offset + spacing, travel_direction)

                elif element == EntityType.path:
                    self._place_entities(EntityType.path, offset, travel_direction)
                    pass

                else:
                    assert False, 'Unsupported EntityType found in lane_layout.'

                if spacing is not None:
                    offset = offset + spacing
                self.offsets.append(offset)
                spacing = None
                last_type = element

            else:
                assert False, 'Unsupported type found in lane_layout.'
        pass

    @property
    def width(self) -> float:
        return self.offsets[-1]

    @property
    def num_segment_points(self) -> int:
        return len(self.segment_points)

    @property
    def num_segments(self) -> int:
        return self.num_segment_points - 1

    @property
    def length(self) -> float:
        return self.segment_distances[-1]

    def _place_entities(
            self,
            entity_type: EntityType,
            offset: float,
            travel_direction: TravelDirection):
        sequence: [LinearEntity] = []
        points: [Vec2d] = [self.segment_points[i + 1] + self.slices[i] * offset
                           for i in range(self.num_segment_points - 2)]

        if travel_direction == TravelDirection.backward:
            points = reversed(points)

        sequence_length = self.num_segments - 2
        for i in range(sequence_length):
            sequence.append(entity_factory.make_entity(
                entity_type, self.parent, points[i], points[i + 1], sequence=sequence, sequence_id=i,
                sequence_length=sequence_length))

        self.entities.append(sequence)

    def road_to_global(self, road_vector: Vec2d, angle: float = 0) -> (Vec2d, float):
        """
        Transforms a road position and angle (distance along road, offset from centerline) to a global position
        (x, y) and angle.
        """
        distance: float = road_vector.x + 1
        offset: float = road_vector.y
        segment: int = min(self.find_segment(distance), self.num_segments - 1)
        distance_along_segment: float = distance - self.segment_distances[segment]

        return self.segment_points[segment] + \
               distance_along_segment * self.segment_directions[segment] + \
               offset * self.segment_normals[segment], \
               angle + self.segment_directions[segment].angle

    def global_to_road(self, global_vector: Vec2d) -> Vec2d:
        """
        Transforms a global position (x, y) to a road position (distance along road, offset from centerline)
        """
        i: int = list(self.segment_point_index.nearest(global_vector, objects=False))[0]

        segment: int = 0
        if i <= 0:
            segment = 0
        elif i >= self.num_segments:
            segment = self.num_segments - 1
        else:
            # determine which side of the dividing line this point is on
            relative_position = global_vector - self.segment_points[i]
            cross = relative_position.cross(self.slices[i - 1])
            segment = i - (1 if cross <= 0 else 0)

        # project the position onto the segment
        relative_position = global_vector - self.segment_points[segment]
        distance_along_segment = relative_position.dot(self.segment_directions[segment])
        longitudinal_offset = distance_along_segment + self.segment_distances[segment] - 1
        lateral_offset = relative_position.dot(self.segment_normals[segment])
        return Vec2d(longitudinal_offset, lateral_offset)

    def find_segment(self, distance: float) -> (int, float):
        return min(max(0, bisect.bisect_right(self.segment_distances, distance) - 1), self.num_segments - 1)
