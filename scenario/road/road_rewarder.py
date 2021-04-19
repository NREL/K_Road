import pymunk
from pymunk import Vec2d

from factored_gym.rewarder import Rewarder
from k_road.constants import Constants
from k_road.entity import Path
from k_road.entity.vehicle.vehicle import VehicleDBM
from k_road.scan import pathfinder
from scenario import RoadProcess


class RoadRewarder(Rewarder):

    def get_reward(self, process: RoadProcess, observation, terminated: bool) -> float:

        target_speed: float = process.speed_mean * 1.1
        shape_weight: float = 0.0

        reward: float = 0.0
        shape_reward: float = 0.0
        goal_reward: float = 0.0
        collision_penalty = 1

        ego_vehicle: VehicleDBM = process.ego_vehicle
        position: Vec2d = ego_vehicle.position
        velocity: Vec2d = ego_vehicle.velocity
        yaw: float = ego_vehicle.angle

        space: pymunk.Space = process.space

        if terminated:
            if ego_vehicle.collided:
                reward = -collision_penalty  # 0
                print("collision penalty")
            elif process.ego_in_end_zone:
                reward = 100
            else:
                reward = 0
                print("time out penalty")
            # else:
            #     # reward = 100 * min(1.0, max(-1.0, process.ego_vehicle.velocity[0] / Constants.max_speed))
            #     reward = 100
            # if process.distance_to_target <= 0:
            #     # reward = .25 + .75 * (1.0 - process.time / process.time_limit)
            #     # reward = .5 + .5 * (process.max_starting_distance_to_target / process.time)
            #     reward = 100 + min(1.0, (process.road_length / process.time) / Constants.max_speed)
            #     # reward = 100
            # else:
            #     reward = 0
            #     # reward = 1 * (1.0 - process.distance_to_target / process.road_length)
            #     # reward = .25 * (
            #     #     (1.0 - process.distance_to_target / process.road_length) +
            #     #     ((process.road_length - process.distance_to_target) / process.time) / Constants.max_speed
            #     # )
            # # if process.ego_vehicle.collided:
            # #     reward = reward - 10
            # reward = shape_reward * shape_weight + goal_reward * goal_weight
            # reward = goal_reward
            pass
        else:

            # speed along nearest path
            scan_position: Vec2d = position + Vec2d(ego_vehicle.length / 2, 0).rotated(yaw)
            path_pqi = pathfinder.find_best_path(space, scan_position, 2 * Constants.lane_width)[0]
            if path_pqi is not None and path_pqi.shape is not None:
                best_path: Path = path_pqi.shape.body.entity
                if best_path is not None:
                    speed_along_path = velocity.dot(best_path.direction)
                    normalized_speed = speed_along_path / target_speed
                    goal_reward = goal_reward + 1.0 * max(-100.0, min(1.0, normalized_speed))
                #     print('path ', best_path.direction, speed_along_path, normalized_speed, goal_reward)
                # else:
                #     print('no path!')
            #     pass
            # # reward = reward + 1.0 * \
            # #          min(1.0, max(-1.0, process.ego_vehicle.velocity[0] / Constants.max_speed))
            # # reward = reward + .001 * (1.0 - process.distance_to_target / process.road_length)
            #
            # scan_radius = 10
            # shape_filter = pymunk.ShapeFilter(categories=EntityCategory.scan)
            # pqis = space.point_query(position, scan_radius, shape_filter)
            #
            # max_closing_time = 10
            # worst_closing_time = max_closing_time
            #
            # closest_entity = scan_radius
            # on_lane_line = False
            # for pqi in pqis:
            #     if pqi.shape is None:
            #         continue
            #
            #     entity: Entity = pqi.shape.body.entity
            #     distance = pqi.distance
            #     if entity == ego_vehicle:
            #         continue
            #
            #     type_ = entity.type_
            #     category = entity.category
            #
            #     if category == EntityCategory.dynamic:
            #         closest_entity = min(closest_entity, distance)
            #
            #     if type_ == EntityType.lane_marking_white_dashed and distance < ego_vehicle.width / 2:
            #         on_lane_line = True
            #
            #     if category == EntityCategory.dynamic or category == EntityCategory.off_road:
            #         relative_velocity = entity.velocity - velocity
            #         relative_position = pqi.point - position
            #         closing_speed = -relative_velocity.dot(relative_position.normalized())
            #         if closing_speed > 0:
            #             closing_time = relative_position.length / closing_speed
            #             worst_closing_time = min(worst_closing_time, closing_time)
            #
            weight_share = 1.0 / 1.0
            #
            # normalized_nearest_distance = (closest_entity / scan_radius)
            # shape_reward = shape_reward + weight_share * normalized_nearest_distance * normalized_nearest_distance
            #
            # shape_reward = shape_reward + weight_share * (worst_closing_time / max_closing_time)
            #
            normalized_yaw_rate = ego_vehicle.angular_velocity / .5
            shape_reward = shape_reward + weight_share * max(0.0, (1 - normalized_yaw_rate * normalized_yaw_rate))
            #
            # normalized_lateral_velocity = min(1.0, math.fabs(ego_vehicle.lateral_velocity / 5))
            # shape_reward = shape_reward + weight_share * max(0.0, normalized_lateral_velocity)

            # if not on_lane_line:
            #     goal_reward = goal_reward + .1

            reward = shape_reward * shape_weight + goal_reward
            reward = reward * process.time_step_length
            # reward = reward + .1 * max(0.0, (1 - math.fabs(ego_vehicle.lateral_velocity) / 2))
            # reward = .001 * (1.0 - process.distance_to_target / process.road_length)
        # reward = reward * math.pow(.99, process.time)
        #        print ("step reward" , reward)
        #        if terminated:
        #            print ("terminated")
        return reward
