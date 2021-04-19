import math
from typing import Optional

import numpy as np
import pymunk
from gym import spaces
from pymunk import Vec2d

from factored_gym import Observer
from k_road.constants import Constants
from k_road.entity.entity_type import EntityType
from k_road.entity.path import Path
from k_road.k_road_view import KRoadView
from k_road.scan import pathfinder
from k_road.util import *
from scenario.road import RoadProcess
from scenario.road.ray_scanner import RayScanner


class FactoredRoadObserver(Observer):

    def __init__(self,
                 obs_config,
                 scaling: float = 1.0,
                 ):

        ## Oh no, config dict!
        ## To connect to Dave, config is a list of dicts.
        # each one has a "type" key,
        # and a "sensors" key, which itself is a list
        # and maybe some type-specific keys

        #                frames_to_observe = 1

        self.scaling: float = scaling
        #        self.frames_to_observe = frames_to_observe
        self.cell_width = 3.7 * 0.2
        self.cell_length = 4.9 * 0.2
        self.rel_vel_max = 20

        self.cts_obs_val = False
        self.separate_obs_val = False

        # whole thing is a dict whose keys access different types of observations
        self._obs_config = obs_config
        self.obs_space = {}  # Will construct an observation space as we setup sensors
        self.setup_sensors()

    def setup_sensors(self):
        self.odometry_length = 0
        if "odometry" in self._obs_config:
            # 0
            self.distance_to_target_index: int = self.odometry_length
            self.odometry_length += 1

            # 1
            self.speed_along_path_index: int = self.odometry_length
            self.odometry_length += 1

            # 2
            self.heading_along_path_index: int = self.odometry_length
            self.odometry_length += 1

            # 3
            self.cross_track_error_index: int = self.odometry_length
            self.odometry_length += 1

            # 4
            self.yaw_rate_index: int = self.odometry_length
            self.odometry_length += 1

            # 5
            self.steer_angle_index: int = self.odometry_length
            self.odometry_length += 1

            # 6
            self.acceleration_index: int = self.odometry_length
            self.odometry_length += 1

            # 7
            self.lateral_velocity_index: int = self.odometry_length
            self.odometry_length += 1

            # 8
            self.longitudinal_velocity_index: int = self.odometry_length
            self.odometry_length += 1

            self.obs_space['odometry'] = spaces.Box(low=-self.scaling, high=self.scaling, shape=(self.odometry_length,))

        if "occupancy_grid" in self._obs_config:
            config = self._obs_config['occupancy_grid']
            self.grid_length = config['grid_dims'][0]  # units? "vehicle lengths"
            self.grid_width = config['grid_dims'][1]  # units?  meters, "lanes" ?

            self.occupancy_grid_length = 0
            self.separate_obs_val = config['grid_dims'][2] > 1
            if (self.separate_obs_val):
                self.nchannels = 2
            else:
                self.nchannels = 1
            print("occupancy grid params length, width, nchannels", self.grid_length, self.grid_width, self.nchannels)
            self.obs_space['occupancy_grid'] = spaces.Box(low=-self.scaling, high=self.scaling,
                                                          shape=(self.grid_length, self.grid_width, self.nchannels),
                                                          dtype=np.float)

        if "lidar" in self._obs_config:
            config = self._obs_config['lidar']
            self.forward_scan_angle = (50.0 / 360) * 2 * math.pi  # was 30
            self.forward_scan_resolution: int = config['forward_scan_resolution']  # 33  # 33,
            forward_scan_radius: float = .2  # .1  # 3.7 / 2.0,
            forward_scan_distance: float = 200  # 150-200m industry standard forward scan distance
            self.rear_scan_resolution: int = config['rear_scan_resolution']  # 33,
            rear_scan_radius: float = .2  # 3.7 / 2.0,
            rear_scan_distance: float = 30

            self.forward_scan_offset = self.forward_scan_angle / 2
            self.forward_scanner = \
                RayScanner(forward_scan_distance, self.forward_scan_angle, forward_scan_radius,
                           self.forward_scan_resolution)

            rear_scan_step = (2 * math.pi - self.forward_scan_angle) / (self.rear_scan_resolution + 2)
            rear_scan_angle = rear_scan_step * self.rear_scan_resolution
            self.rear_scan_offset = -self.forward_scan_offset + self.forward_scan_angle + rear_scan_step
            self.rear_scanner = \
                RayScanner(rear_scan_distance, rear_scan_angle, rear_scan_radius, self.rear_scan_resolution)
            self.num_scan_arrays = config[
                'lidar_channels']  # this number separates different types of stuff being detected (e.g. curb vs car), which may not be realistic

            self.forward_scan_vehicle_index: int = 0

            self.lidar_length: int = self.forward_scan_resolution * self.num_scan_arrays

            self.rear_scan_vehicle_index: int = self.lidar_length
            self.lidar_length: int = self.lidar_length + self.rear_scan_resolution * self.num_scan_arrays
            self.forward_scan_results = None
            self.rear_scan_results = None
            self.obs_space['lidar'] = spaces.Box(low=-self.scaling, high=self.scaling, shape=(self.lidar_length,))

        ## TODO: multiple frames
        # self.frame_length = self.observation_length
        # self.observation_length *= self.frames_to_observe

    ###        self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)

    def get_observation_space(self, process: RoadProcess):
        d = spaces.Dict(self.obs_space)
        return d

    def render(self, process: RoadProcess, view: KRoadView) -> None:
        if "occupancy_grid" in self.obs_space:
            self.draw_grid(process, view)
        if "lidar" in self.obs_space:
            self.draw_scan(process, view, self.forward_scanner, self.forward_scan_results)
            self.draw_scan(process, view, self.rear_scanner, self.rear_scan_results, (96, 0, 0))

    def get_observation(self, process: RoadProcess):
        data_dict = {}

        ego_vehicle = process.ego_vehicle
        position = ego_vehicle.position
        velocity = ego_vehicle.velocity
        yaw = ego_vehicle.angle

        space = process.space

        ### TODO:
        # first load past time steps frames if we have them.
        # for i in range(1, self.frames_to_observe):
        #     observation[i*self.frame_length: (i+1)*self.frame_length] = observation[(i-1)*self.frame_length: i*self.frame_length]

        if "odometry" in self.obs_space:
            # compute observation
            observation = np.zeros(self.obs_space['odometry'].shape)
            # standard ones from RoadObserver (could easily be superclass)
            scan_position: Vec2d = position + Vec2d(ego_vehicle.length / 2, 0).rotated(yaw)
            path_pqi: Optional[pymunk.PointQueryInfo] = \
                pathfinder.find_best_path(space, scan_position, 1 * Constants.lane_width)[0]
            cross_track_error: float = 0 if path_pqi is None else (path_pqi.point - position).length
            best_path: Optional[Path] = None if path_pqi is None else path_pqi.shape.body.entity
            # best_path = None
            # cross_track_error: float = 0

            observation[self.distance_to_target_index] = \
                min(1.0, max(-1.0, 2 * (process.distance_to_target / process.target_offset) - 1))

            # maxed at 1
            speed_along_path: float = \
                velocity.dot(best_path.direction) if best_path is not None else 0
            observation[self.speed_along_path_index] = min(1.0,
                                                           max(-1.0, speed_along_path / (1.1 * Constants.max_speed)))

            angle_agreement_with_path: float = \
                signed_delta_angle(best_path.direction.angle, velocity.angle) if best_path is not None else 0
            observation[self.heading_along_path_index] = min(1.0, max(-1.0, angle_agreement_with_path / math.pi))

            observation[self.cross_track_error_index] = min(1.0,
                                                            max(-1.0, cross_track_error / (Constants.lane_width * 2)))

            # maxed at 1.0?
            observation[self.yaw_rate_index] = \
                min(1.0, max(-1.0, ego_vehicle.angular_velocity / (.04 * 2 * math.pi)))  # 4

            observation[self.steer_angle_index] = \
                min(1.0, max(-1.0, ego_vehicle.steer_angle / Constants.max_steer_angle))  # 5

            # minned at -1.0?
            observation[self.acceleration_index] = \
                min(1.0, max(-1.0,
                             ego_vehicle.acceleration / (1.1 * Constants.max_acceleration)
                             if ego_vehicle.acceleration >= 0 else
                             ego_vehicle.acceleration / (1.1 * Constants.max_deceleration)))  # 6
            # print('acc: ', ego_vehicle.acceleration, observation[self.acceleration_index])

            # 0'ed
            observation[self.lateral_velocity_index] = \
                min(1.0, max(-1.0, ego_vehicle.lateral_velocity / 1.0))  # 1 m/s cap (7)

            # 1'ed
            observation[self.longitudinal_velocity_index] = \
                min(1.0, max(-1.0, ego_vehicle.longitudinal_velocity / (1.1 * Constants.max_speed)))

            data_dict['odometry'] = observation

        if ('occupancy_grid' in self.obs_space):
            observation = np.zeros(self.obs_space['occupancy_grid'].shape)

            # Then ones from occupancy grid
            sf = pymunk.ShapeFilter()

            #      print ("\n\n\n")
            v_e = ego_vehicle.velocity
            for i in range(self.grid_length):
                for j in range(self.grid_width):
                    idx = i * self.grid_width + j
                    ## get bb
                    vtx, box = self.cell_bb(i, j, position, yaw)
                    ## probe space for stuff at that spot
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
                            #                            if (p_b - position).get_length() > 5:     # this is preventing ego itself from being detected and basically blocking itself (came up in path planning)
                            if (
                                    ego_vehicle.id != entity.id):  # this is preventing ego itself from being detected and basically blocking itself (came up in path planning)
                                p_be = (p_b - position).normalized()
                                rel_speed = v_eb.dot(p_be)
                                value_veh = rel_speed

                                vehicle_in_cell = True
                        #   print (i,j,"veh", v_e, v_b, position, p_b, p_be, rel_speed)
                        elif entity.type_ == EntityType.curb and (
                                not vehicle_in_cell or self.separate_obs_val):  # vehicle in cell takes precedence
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
                    if (self.separate_obs_val):
                        observation[i, j, 0] = clipped_value_veh
                        observation[i, j, 1] = clipped_value_curb
                    else:
                        #   clipped_value = clipped_value_veh if vehicle_in_cell else clipped_value_curb
                        clipped_value = clipped_value_curb
                        observation[i, j, 0] = clipped_value

            data_dict['occupancy_grid'] = observation

        if "lidar" in self.obs_space:
            lidar_obs = self.get_lidar_obs(process)
            data_dict['lidar'] = lidar_obs

        #    print (observation)
        self.observation = data_dict
        return data_dict

    ########### Lidar ### move to separate class
    def get_lidar_obs(self, process):
        ego_vehicle = process.ego_vehicle
        position = ego_vehicle.position
        velocity = ego_vehicle.velocity
        yaw = ego_vehicle.angle
        space = process.space
        observation = np.zeros(self.obs_space['lidar'].shape)

        self.forward_scan_results = \
            self.forward_scanner.scan_closest_of_each_type(
                position,
                yaw - self.forward_scan_offset,
                space,
                lambda rsr: rsr.entity != ego_vehicle)
        self.__extract_scan_data(
            self.forward_scan_results,
            observation,
            self.forward_scan_vehicle_index,
            position,
            velocity)

        self.rear_scan_results = \
            self.rear_scanner.scan_closest_of_each_type(
                position, yaw + self.rear_scan_offset, space,
                lambda rsr: rsr.entity != ego_vehicle)
        self.__extract_scan_data(
            self.rear_scan_results,
            observation,
            self.rear_scan_vehicle_index,
            position,
            velocity)
        return observation

    def draw_scan(self, process: RoadProcess, view: KRoadView, ray_scanner, scan_results, color=(64, 64, 64)) -> None:
        if scan_results is None:
            return

        position = process.ego_vehicle.position
        contact_size = min(.2, ray_scanner.beam_radius / 2)
        for (endpoint, ray_contacts) in scan_results:
            ray_vector = (endpoint - position)
            view.draw_segment(color, False, position, endpoint, ray_scanner.beam_radius)

            def draw_contact(color, type_):
                if type_ in ray_contacts:
                    view.draw_circle(
                        color,
                        position + ray_vector * ray_contacts[type_].alpha,
                        contact_size)

            draw_contact((0, 255, 0), EntityType.curb)
            # draw_contact((0, 255, 0), EntityType.lane_marking_white_dashed)
            draw_contact((255, 0, 0), EntityType.vehicle)

    def __extract_scan_data(self, scan_results, observation, index_offset, position, velocity):
        num_rays = len(scan_results)

        for i in range(num_rays):
            (endpoint, ray_contacts) = scan_results[i]
            #            print (endpoint, ray_contacts)

            ray_unit = (endpoint - position).normalized()

            # def get_contact(type_):
            #     if type_ in ray_contacts:
            #         return 2 * ray_contacts[type_].alpha - 1
            #     else:
            #         return 1.0

            def get_contact(type_):
                if type_ in ray_contacts:
                    # print ("TYPE",type)
                    # print( ray_contacts, ray_contacts[type_].alpha)
                    # print (ray_contacts[type_])
                    return 1 - ray_contacts[type_].alpha
                else:
                    return 0.0

            def get_contact_closing_speed(type_):
                if type_ not in ray_contacts:
                    return 0.0
                rcr = ray_contacts[type_]
                relative_velocity = rcr.entity.velocity - velocity
                return max(-1.0, min(1.0, relative_velocity.dot(ray_unit) / Constants.max_speed))

            offset = 0

            veh_contact = get_contact(EntityType.vehicle)
            # offset = offset + 1
            #
            # observation[index_offset + num_rays * offset + i] = get_contact_closing_speed(EntityType.vehicle)
            # offset = offset + 1

            curb_contact = get_contact(EntityType.curb)

            if self.num_scan_arrays == 1:
                observation[index_offset + num_rays * offset + i] = max(veh_contact, curb_contact)
            elif self.num_scan_arrays == 2:
                observation[index_offset + num_rays * offset + i] = veh_contact
                offset = offset + 1
                observation[index_offset + num_rays * offset + i] = curb_contact

            offset = offset + 1

    ########### Occupancy grid ### move to separate class
    def cell_to_pt_local(self, i, j, position, yaw):
        half_grid_width = (self.grid_width - 0.5) * self.cell_width / 2
        half_grid_length = (self.grid_length - 0.5) * self.cell_length / 2
        x = (i + 0.5) * self.cell_length - half_grid_length
        y = (j + 0.5) * self.cell_width - half_grid_width
        return Vec2d(x, y)

    def pt_to_cell(self, pt, position, yaw):
        half_grid_width = (self.grid_width - 0.5) * self.cell_width / 2
        half_grid_length = (self.grid_length - 0.5) * self.cell_length / 2
        pt = pt - position
        pt = pt.rotated(-yaw)
        iraw = (pt.x + half_grid_length) / self.cell_length - 0.5
        jraw = (pt.y + half_grid_width) / self.cell_width - 0.5
        i = int(np.round(iraw))
        j = int(np.round(jraw))
        #        print ("RAW", iraw, i,  "   ", jraw, j)
        return i, j

    def cell_bb(self, i, j, position, yaw):
        ## figure out where (i,j) actually is and make pymunk bounding box
        pt = self.cell_to_pt_local(i, j, position, yaw)
        x = pt[0]
        y = pt[1]

        # for fun, see if we can do round trip:
        pt = pt.rotated(yaw)
        pt = pt + position
        ii, jj = self.pt_to_cell(pt, position, yaw)
        #        print ("IN/OUT", i,j, pt, ii,jj)
        assert (ii == i and jj == j)

        # left = x - self.cell_length / 2 + 1
        # right = x + self.cell_length / 2 - 1
        # bottom = y - self.cell_width / 2 
        # top = y + self.cell_width / 2 - 1
        left = x - self.cell_length / 2
        right = x + self.cell_length / 2
        bottom = y - self.cell_width / 2
        top = y + self.cell_width / 2
        vtx = [Vec2d(left, bottom), Vec2d(left, top), Vec2d(right, top), Vec2d(right, bottom)]
        vtx = [v.rotated(yaw) for v in vtx]
        vtx = [v + position for v in vtx]

        # make pymunk body 
        box_body = pymunk.Body(0, 0)  # 2
        box = pymunk.Poly(box_body, vtx, radius=0.1)
        box.cache_bb()
        return vtx, box

    def check_status_of_point(self, pt, process):
        # return <is point free>, <is point off front of grid>
        position = process.ego_vehicle.position
        yaw = process.ego_vehicle.angle
        i, j = self.pt_to_cell(pt, position, yaw)

        vtx, _ = self.cell_bb(i, j, position, yaw)
        #        print ("STAT:", pt, vtx, i,j, position, yaw)
        # iego,jego = self.pt_to_cell(process.ego_vehicle.position, process)
        # if (i==iego and j==jego):
        #     print ("")
        #     return True, False # we are still in same cell as ego, say it's ok and keep growing path
        # print ("point to cell", pt, i, j, process.ego_vehicle.position, iego, jego)
        if i >= self.grid_length and j >= 0 and j < self.grid_width:
            #            print ("off end of grid", i,j )
            return True, True  ## off grid, but in a good way (off the front)
        if i < 0 or i >= self.grid_length or j < 0 or j >= self.grid_width:
            #            print ("off side of grid", i, j)
            return False, False  # off grid, not in a good way
        assert (not self.cts_obs_val)  # code below relies on this
        for k in range(self.nchannels):
            #            print ("ij", pt, i,j, self.observation['occupancy_grid'][i,j,k])
            if self.observation['occupancy_grid'][i, j, k] != -self.scaling:
                #                print ("occupied", self.observation['occupancy_grid'][i,j,k])
                return False, False  # space not free, path not complete
        return True, False  # space free, path not complete yet (still in grid)

    def draw_bb(self, vtx, box, view, value):
        if (value):
            red = np.array([255., 0, 0])
            green = np.array([0, 255., 0])
            v01 = (value + 1) / 2.
            col = v01 * red + (1 - v01) * green
            col = tuple(int(x) for x in col)
        #            view.draw_shape_polygon(box.body, box, col, True)
        else:
            col = (55, 111, 0)
        for i in range(4):
            if v01 > 0.5:
                view.draw_line(col, vtx[i], vtx[(i + 1) % 4])

    def draw_grid(self, process: RoadProcess, view: KRoadView) -> None:
        position = process.ego_vehicle.position
        yaw = process.ego_vehicle.angle
        for i in range(self.grid_length):
            for j in range(self.grid_width):
                idx = i * self.grid_width + j
                vtx, box = self.cell_bb(i, j, position, yaw)
                for k in range(self.nchannels):
                    self.draw_bb(vtx, box, view, self.observation['occupancy_grid'][i, j, k])

    def dump_og(self):
        #        position = process.ego_vehicle.position
        #        yaw = process.ego_vehicle.angle
        print("dump_og")
        s = "   "
        for j in range(self.grid_width):
            s = "%s %d" % (s, j)
        print(s)
        for i in range(self.grid_length):
            s = "%d  " % i
            for j in range(self.grid_width):
                #                idx = i * self.grid_width + j
                #                vtx, box = self.cell_bb(i,j,position, yaw)
                if self.observation['occupancy_grid'][i, j, 0] == 1 and self.observation['occupancy_grid'][
                    i, j, 1] == 1:
                    c = "+"
                elif self.observation['occupancy_grid'][i, j, 0] == 1:
                    c = "v"
                elif self.observation['occupancy_grid'][i, j, 1] == 1:
                    c = "c"
                else:
                    c = "-"
                s = "%s %s" % (s, c)
            print(s)
