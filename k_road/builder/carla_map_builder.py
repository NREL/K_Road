import json
from math import inf

# import rtree as rtree
from pymunk import Vec2d

from k_road.entity.entity_type import EntityType


class CarlaMapBuilder:

    def __init__(
            self,
            parent: 'KRoadProcess',
            filename: str,
    ):
        """
            Creates a roadway with lanes, curbs, paths, etc
            
            TODO: Angled multi-point roads:
                + define line at .5*angle between ever three points
                + use that line as a cutting plane to chop off offset lines
                + extension = offset / tan(angle / 2)
            
        """
        self.parent = parent
        self.filename = filename
        self.path_waypoints = []

        segment_lookup = {
            'off_road': EntityType.curb,
            'path': EntityType.path,
            'lane_marking_white': EntityType.lane_marking_white,
            'lane_marking_yellow_dashed': EntityType.lane_marking_yellow_dashed
        }

        lines = json.loads(open(filename).read())
        path = [x for x in lines['segments'] if x['road_entity'] == 'path'][0]['segment']
        lastv = Vec2d(inf, inf)
        for p in path:
            v = Vec2d(p[0], p[1])  ### negate x due to bug(?) in Monte's json files?--or maybe on purpose for Charles
            if v != lastv:
                self.path_waypoints.append(v)
            lastv = v
            # print ("POINT", p, v)
        # for line in lines:
        #     try:
        #         start_point = Vec2d(line['segment'][0])
        #         end_point = Vec2d(line['segment'][1])
        #         print ("JUST read points", start_point, end_point, line)
        #         entity_type = segment_lookup.get(line.get('entity_type'))
        #         if entity_type is not None:
        #             if entity_type == EntityType.path:
        #                 print(start_point)
        #                 self.path_waypoints.append(start_point)
        #             # print(line.get('entity_type'), entity_type)
        #             if (parent != None):
        #                 entity_factory.make_entity(entity_type, self.parent, start_point, end_point)
        #     except KeyError:
        #         pass

        # start_point: Vec2d = center_points[0] - (center_points[1] - center_points[0]).normalized() \
        #     if start_point is None else start_point

        # end_point: Vec2d = center_points[-1] + (center_points[-1] - center_points[-2]).normalized() \
        #     if end_point is None else end_point

        # self.lane_layout: [Union[EntityType, float, TravelDirection]] = lane_layout

        # self.segment_points: [Vec2d] = [start_point, *center_points, end_point]

        # # build a segment point index
        # self.segment_point_index: rtree.index.Index = rtree.index.Index(interleaved=False)
        # for i, point in enumerate(self.segment_points):
        #     self.segment_point_index.insert(i, (point.x, point.x, point.y, point.y))

        # self.segment_vectors: [Vec2d] = \
        #     list([self.segment_points[i + 1] - self.segment_points[i]
        #           for i in range(len(self.segment_points) - 1)])

        # self.segment_directions: [Vec2d] = list([v.normalized() for v in self.segment_vectors])
        # self.segment_normals: [Vec2d] = list([v.perpendicular() for v in self.segment_directions])

        # self.slices = \
        #     [(self.segment_normals[i] + self.segment_normals[i + 1]).normalized()
        #      for i in range(len(self.segment_normals) - 1)]
        # self.slices = list(
        #     [self.slices[i] / (self.slices[i].dot(self.segment_normals[i]))
        #      for i in range(len(self.slices))])

        # self.segment_lengths: [float] = list([vector.length for vector in self.segment_vectors])

        # self.segment_distances: [float] = list([0, *itertools.accumulate(self.segment_lengths)])

        # self.num_lanes: int = 0
        # # self.length: float = 0

        # last_type: EntityType = EntityType.null
        # spacing: Optional[float] = None
        # offset: float = 0.0
        # self.offsets: [float] = []
        # travel_direction: TravelDirection = TravelDirection.forward

        # for element in self.lane_layout:
        #     if isinstance(element, float):
        #         spacing = element

        #     elif isinstance(element, TravelDirection):
        #         direction = element

        #     elif isinstance(element, EntityType):
        #         if element.group == EntityGroup.lane:
        #             pass  # create a lane

        #         elif element.group == EntityGroup.lane_marking:
        #             if last_type.group == EntityGroup.lane:
        #                 if spacing is None:
        #                     spacing = Constants.lane_width
        #                 self._place_entities(EntityType.path, offset + spacing / 2, travel_direction)
        #                 self.num_lanes = self.num_lanes + 1
        #             else:
        #                 if spacing is None:
        #                     spacing = 0

        #             # create a lane marking
        #             self._place_entities(element, offset + spacing, travel_direction)

        #         elif element == EntityType.curb:
        #             if spacing is None:
        #                 spacing = Constants.gutter_width

        #             self._place_entities(element, offset + spacing, travel_direction)

        #         else:
        #             assert False, 'Unsupported EntityType found in lane_layout.'

        #         if spacing is not None:
        #             offset = offset + spacing
        #         self.offsets.append(offset)
        #         spacing = None
        #         last_type = element

        #     else:
        #         assert False, 'Unsupported type found in lane_layout.'
        # pass

    # @property
    # def width(self) -> float:
    #     return self.offsets[-1]

    # @property
    # def num_segment_points(self) -> int:
    #     return len(self.segment_points)

    # @property
    # def num_segments(self) -> int:
    #     return self.num_segment_points - 1

    # @property
    # def length(self) -> float:
    #     return self.segment_distances[-1]

    # def _place_entities(self, entity_type: EntityType, offset: float, travel_direction: TravelDirection) -> None:
    #     points: [Vec2d] = [self.segment_points[i + 1] + self.slices[i] * offset
    #                        for i in range(self.num_segment_points - 2)]

    #     if travel_direction == TravelDirection.backward:
    #         points = reversed(points)

    #     for i in range(self.num_segments - 2):
    #         entity_factory.make_entity(entity_type, self.parent, points[i], points[i + 1])
