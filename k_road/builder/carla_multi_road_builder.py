import itertools
import json
from math import inf

import rtree as rtree
from pymunk import Vec2d

from k_road.builder.road_builder import RoadBuilder
from k_road.constants import Constants
from k_road.entity import entity_factory
from k_road.entity.entity_type import EntityType


# ------------------------------------------------#


### NOTE again whether we are mapping to right hand coords here or not.
# This ends up by negating all y coordinates
# The trick is to be consistent w.r.t. when this conversion happens.
# which so far we have not been succeeding. sorry
###

class CarlaMapBuilder(RoadBuilder):

    def __init__(
            self,
            parent: 'KRoadProcess',
            filename: str,
            route_num=None
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
        self.path_waypoint_labels = []
        self.current_route = None

        segment_lookup = {
            'off_road': EntityType.curb,
            'path': EntityType.path,
            'lane_marking_white': EntityType.lane_marking_white,
            'lane_marking_yellow_dashed': EntityType.lane_marking_yellow_dashed,
            'road_right': EntityType.lane_marking_yellow,
            'road_left': EntityType.lane_marking_yellow,
            'right': EntityType.lane_marking_white,
            'left': EntityType.lane_marking_white,
        }

        lines = json.loads(open(filename).read())
        self.lines = lines  ## for later re-use

        ### NOTE: Choosing here in this translation from CARLA to k_road to finally invert y to switch from left to
        # right hand
        ### coordinates

        just_path = False
        single_route = False
        if just_path:
            path = [x for x in lines['segments'] if x['road_entity'] == 'path'][0]['segment']
            lastv = Vec2d(inf, inf)
            for p in path:
                v = Vec2d(p[0], -p[1])
                if v != lastv:
                    self.path_waypoints.append(v)
                lastv = v
                # print ("POINT", p, v)
        elif single_route:
            self.meta = lines['meta']
            segments = lines['segments']
            # sr = [x for x in segments if x['road_entity'] == 'right']
            # srr = [x for x segments if x['road_entity'] == 'road_right']
            # sl = [x for x segments if x['road_entity'] == 'left']
            # slr = [x for x segments if x['road_entity'] == 'road_left']
            path = [x for x in segments if x['road_entity'] == 'path'][0]['segment']  # path is special
            segments = [x for x in segments if x['road_entity'] != 'path']

            # sall = sr + sl + srr + sall
            for line in segments:
                start_point = Vec2d(line['segment'][0])
                end_point = Vec2d(line['segment'][1])
                start_point[1] *= -1
                end_point[1] *= -1
                entity_type = segment_lookup.get(line.get('road_entity'))
                if entity_type is not None and parent is not None:
                    #           print ("MAKING", entity_type, start_point, end_point)
                    entity_factory.make_entity(entity_type, self.parent, start_point, end_point)

            lastv = Vec2d(inf, inf)
            for i in range(len(path) - 1):
                p = path[i]
                v = Vec2d(p[0], -p[1])
                if v != lastv:
                    self.path_waypoints.append(v)
                    if i > 0:
                        entity_factory.make_entity(EntityType.path, self.parent, lastv, v)
                        print("MAKING PATH", lastv, v)

                lastv = v

            # HACKS
            self.num_lanes = 2
            self.offsets = [self.num_lanes * Constants.lane_width]
            self.lane_center_offsets = [0.0 * Constants.lane_width, 1.0 * Constants.lane_width]
        #            self.lane_center_offsets = [0.5 * Constants.lane_width, 1.5 * Constants.lane_width]

        else:  # multi-route
            self.meta = lines['meta']
            if 'segments' in lines:
                segments = lines['segments']
                for line in segments:
                    start_point = Vec2d(line['segment'][0])
                    end_point = Vec2d(line['segment'][1])
                    start_point[1] *= -1
                    end_point[1] *= -1
                    entity_type = segment_lookup.get(line.get('road_entity'))
                    if entity_type is not None and parent is not None:
                        #                    print ("MAKING", entity_type, start_point, end_point)
                        entity_factory.make_entity(entity_type, self.parent, start_point, end_point)

            boundaries = lines['boundaries']
            for line in boundaries:
                xs = line['xs']
                ys = line['ys']
                entity_type = EntityType.curb
                #      print ("MAKING", entity_type)
                p0 = Vec2d(xs[0], -ys[0])
                for i in range(1, len(xs)):
                    p1 = Vec2d(xs[i], -ys[i])
                    entity_factory.make_entity(entity_type, self.parent, p0, p1)
                    p0 = p1

            routes = range(self.meta['num_routes']) if route_num is None else [route_num]
            for il in routes:
                path = lines['route%d' % il]['segment']
                lastv = Vec2d(inf, inf)
                for i in range(len(path) - 1):
                    p = path[i]
                    v = Vec2d(p[0], -p[1])
                    if v != lastv:
                        ## This makes route0 or specified route the special waypoint path
                        if il == 0 or route_num != None:
                            self.path_waypoints.append(v)
                            self.path_waypoint_labels.append(
                                p[2])  # this is the "road option", ie what the route says we do now (e.g turn left)

                        ## This makes a k_road "path" entity in pymunk for all the routes
                        if i > 0:
                            entity_factory.make_entity(EntityType.path, self.parent, lastv, v)
                    #      print ("MAKING PATH", lastv, v)

                    lastv = v
                if il == 0 or route_num != None:
                    self.meta['length'] = lines['route%d' % il]['length']
            self.num_routes = len(routes)

            # HACKS
            self.num_lanes = 1
            self.offsets = [self.num_lanes * Constants.lane_width]
            self.lane_center_offsets = [0.0 * Constants.lane_width, 1.0 * Constants.lane_width]

    #            self.lane_center_offsets = [0.5 * Constants.lane_width, 1.5 * Constants.lane_width]

    def route_to_global(self, route, pos):
        if self.current_route != route:
            self.set_route(route)
        return self.road_to_global(pos)

    def global_to_route(self, route, pos):
        if self.current_route != route:
            self.set_route(route)
        return self.global_to_road(pos)

    def route_length(self, route):
        if self.current_route != route:
            self.set_route(route)
        return self.length

    def set_route(self, route_num):
        self.current_route = route_num
        path = self.lines['route%d' % route_num]['segment']
        self.path_waypoints = []
        lastv = Vec2d(inf, inf)
        for i in range(len(path) - 1):
            p = path[i]
            v = Vec2d(p[0], -p[1])
            if v != lastv:
                self.path_waypoints.append(v)
        self.meta['length'] = self.lines['route%d' % route_num]['length']
        self.finish_up()  ## reset path segments and stuff.

    #    print ("switched waypoint route num to route", route_num)

    def finish_up(self):
        self.segment_points: [Vec2d] = self.path_waypoints

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

        # for s in self.segment_vectors:
        #     print (s)

    @property
    def length(self) -> float:
        return self.meta['length']
