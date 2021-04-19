import numpy as np
import pymunk
from cavs_environments.framework.observer import Observer
from cavs_environments.vehicle.deep_road.deep_road_constants import DeepRoadConstants
from gym import spaces
from pymunk import Vec2d

from k_road.entity.entity_type import EntityType
from k_road.k_road_view import KRoadView
from scenario import RoadProcess


class RoadOccupancyGridObserver(Observer):

    def __init__(self,
                 scaling: float = 1.0,
                 grid_length=9,  # units? "vehicle lengths"
                 grid_width=5,  # units?  meters, "lanes" ?
                 #                frames_to_observe = 1
                 ):
        self.scaling: float = scaling
        self.grid_length = grid_length
        self.grid_width = grid_width
        #        self.frames_to_observe = frames_to_observe
        #        self.cell_width = DeepRoadConstants.lane_width + 1
        #        self.cell_length = DeepRoadConstants.car_length + 2
        self.cell_width = DeepRoadConstants.lane_width / 2
        self.cell_length = DeepRoadConstants.car_length / 2
        self.rel_vel_max = 20

        self.cts_obs_val = False

        self.observation_space = spaces.Box(low=-self.scaling, high=self.scaling,
                                            shape=(self.grid_length, self.grid_width, 1), dtype=np.float)

    ###        self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)

    def get_observation_space(self, process: RoadProcess):
        return self.observation_space

    def get_observation(self, process: RoadProcess):
        # compute observation
        observation = np.zeros(self.observation_space.shape)

        ego_vehicle = process.ego_vehicle
        position = ego_vehicle.position
        velocity = ego_vehicle.velocity
        yaw = ego_vehicle.angle

        space = process.space
        # Then ones from occupancy grid
        sf = pymunk.ShapeFilter()

        #      print ("\n\n\n")
        v_e = ego_vehicle.velocity
        for i in range(self.grid_length):
            for j in range(self.grid_width):
                ## get bb
                bb = self.cell_bb(i, j, position)
                ## probe space for stuff at that spot
                box_body = pymunk.Body(0, 0)  # 2
                box = pymunk.Poly.create_box_bb(box_body, bb)
                box.cache_bb()
                # hits = space.bb_query(bb,sf)
                hits = space.shape_query(box)
                #             print (i,j, "%d hits" % len(hits))
                value_veh = 0
                value_curb = 0
                vehicle_in_cell = False
                curb_in_cell = False
                for h in hits:
                    shape = h.shape
                    body = shape.body
                    entity = body.entity
                    #                    print (shape, body, entity, entity.type_,entity.id)
                    # print (i,j,bb,hits)
                    if entity.type_ == EntityType.vehicle:
                        v_b = entity.velocity
                        p_b = entity.position
                        v_eb = v_e - v_b
                        p_be = (p_b - position).normalized()
                        rel_speed = v_eb.dot(p_be)
                        value_veh = rel_speed
                        vehicle_in_cell = True
                    #   print (i,j,"veh", v_e, v_b, position, p_b, p_be, rel_speed)
                    elif entity.type_ == EntityType.curb:
                        p_b = entity.position
                        p_be = (p_b - position).normalized()
                        rel_speed = v_e.dot(p_be)
                        #   print (i,j,"curb", v_e, p_be, rel_speed)
                        value_curb = rel_speed
                        curb_in_cell = True
                ## store it in obs
                clipped_value_veh = min(self.rel_vel_max, max(value_veh, -self.rel_vel_max)) / self.rel_vel_max
                clipped_value_curb = min(self.rel_vel_max, max(value_curb, -self.rel_vel_max)) / self.rel_vel_max
                if not self.cts_obs_val:
                    clipped_value_veh = self.scaling if vehicle_in_cell else -self.scaling
                    clipped_value_curb = self.scaling if curb_in_cell else -self.scaling
                #                observation[i,j,0] = clipped_value_veh
                observation[i, j, 0] = clipped_value_curb

        #    print ("")
        #        print ("OBS_veh\n", np.array(occ_map_veh).reshape(self.grid_length, self.grid_width).T)
        #        print ("OBS_curb\n", np.array(occ_map_curb).reshape(self.grid_length, self.grid_width).T)
        #    print ("\n\n\n")

        #    print (observation)
        self.observation = observation
        return observation

    def cell_bb(self, i, j, position):
        ## figure out where (i,j) actually is and make pymunk bounding box
        half_grid_width = (self.grid_width - 0.5) * self.cell_width / 2
        half_grid_length = (self.grid_length - 0.5) * self.cell_length / 2
        x = position[0] + i * self.cell_length - half_grid_length
        y = position[1] + j * self.cell_width - half_grid_width + 1
        left = x - self.cell_length / 2
        right = x + self.cell_length / 2
        bottom = y - self.cell_width / 2
        top = y + self.cell_width / 2
        bb = pymunk.BB(left, bottom, right, top)
        return bb

    def render(self, process: RoadProcess, view: KRoadView) -> None:
        self.draw_grid(process, view)

    def draw_bb(self, bb, view, value):
        view.draw_line((55, 111, 0), Vec2d(bb.left, bb.bottom), Vec2d(bb.left, bb.top))
        view.draw_line((55, 111, 0), Vec2d(bb.left, bb.bottom), Vec2d(bb.right, bb.bottom))
        view.draw_line((55, 111, 0), Vec2d(bb.right, bb.top), Vec2d(bb.left, bb.top))
        view.draw_line((55, 111, 0), Vec2d(bb.right, bb.top), Vec2d(bb.right, bb.bottom))
        if (value):
            box_body = pymunk.Body(0, 0)
            box = pymunk.Poly.create_box_bb(box_body, bb)
            box.cache_bb()
            red = np.array([255., 0, 0])
            green = np.array([0, 255., 0])
            v01 = (value + 1) / 2.
            col = v01 * red + (1 - v01) * green
            col = tuple(int(x) for x in col)
            view.draw_shape_polygon(box_body, box, col, True)

    def draw_grid(self, process: RoadProcess, view: KRoadView) -> None:
        position = process.ego_vehicle.position
        field = 0
        for i in range(self.grid_length):
            for j in range(self.grid_width):
                idx = i * self.grid_width + j
                bb = self.cell_bb(i, j, position)
                #                self.draw_bb(bb, view, self.observation[idx + self.occupancy_grid_start_index])
                self.draw_bb(bb, view, self.observation[i, j, field])
