from factored_gym.rewarder import Rewarder
from scenario import RoadProcess


class RoadGoalRewarder(Rewarder):

    def get_reward(self, process: RoadProcess, observation, terminated: bool) -> float:

        target_speed = process.speed_mean * 1.1
        shape_weight = .05

        reward = 0.0
        shape_reward = 0.0
        goal_reward = 0.0

        ego_vehicle = process.ego_vehicle
        position = ego_vehicle.position
        velocity = ego_vehicle.velocity

        if terminated:

            # overall_time = process.time
            # overall_travel = process.ego_starting_distance - process.distance_to_target
            # overall_speed = overall_travel / overall_time

            if ego_vehicle.collided:
                reward = 0
            elif process.ego_in_end_zone:
                reward = 100
            else:
                reward = .01

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
            forward_speed = velocity[0]
            reward = 0.01 * min(1.0, max(-1.0, forward_speed / target_speed))
            reward = reward * process.time_step_length
            # goal_reward = goal_reward + 1.0 * max(-1.0, min(1.0, normalized_speed))
            # # reward = reward + 1.0 * \
            # #          min(1.0, max(-1.0, process.ego_vehicle.velocity[0] / Constants.max_speed))
            # # reward = reward + .001 * (1.0 - process.distance_to_target / process.road_length)
            #
            # space = process.get_k_road_process().get_space()
            #
            # scan_radius = 10
            # shape_filter = pymunk.ShapeFilter(categories=CollisionCategory.scan)
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
            #     collision_category = entity.collision_category
            #
            #     if collision_category == CollisionCategory.dynamic:
            #         closest_entity = min(closest_entity, distance)
            #
            #     if type_ == EntityType.lane_marking_white_dashed and distance < ego_vehicle.width / 2:
            #         on_lane_line = True
            #
            #     if collision_category == CollisionCategory.dynamic or type_ == EntityType.off_road:
            #         relative_velocity = entity.velocity - velocity
            #         relative_position = pqi.point - position
            #         closing_speed = -relative_velocity.dot(relative_position.normalized())
            #         if closing_speed > 0:
            #             closing_time = relative_position.length / closing_speed
            #             worst_closing_time = min(worst_closing_time, closing_time)
            #
            # weight_share = 1.0 / 4.0
            #
            # normalized_nearest_distance = (closest_entity / scan_radius)
            # shape_reward = shape_reward + weight_share * normalized_nearest_distance * normalized_nearest_distance
            #
            # shape_reward = shape_reward + weight_share * (worst_closing_time / max_closing_time)
            #
            # normalized_yaw_rate = ego_vehicle.yaw_rate / .5
            # shape_reward = shape_reward + weight_share * max(0.0, (1 - normalized_yaw_rate * normalized_yaw_rate))
            #
            # normalized_lateral_velocity = min(1.0, math.fabs(ego_vehicle.lateral_velocity / 5))
            # shape_reward = shape_reward + weight_share * max(0.0, normalized_lateral_velocity)
            #
            # # if not on_lane_line:
            # #     goal_reward = goal_reward + .1
            #
            # reward = shape_reward * shape_weight + goal_reward
            # reward = reward * process.time_step_length
            # reward = reward + .1 * max(0.0, (1 - math.fabs(ego_vehicle.lateral_velocity) / 2))
            # reward = .001 * (1.0 - process.distance_to_target / process.road_length)
        # reward = reward * math.pow(.99, process.time)
        return reward
