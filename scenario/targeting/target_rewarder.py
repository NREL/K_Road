from factored_gym.rewarder import Rewarder
from scenario.targeting import TargetProcess


class TargetRewarder(Rewarder):

    def __init__(self, time_limit: float):
        self.time_limit: float = time_limit

    def get_reward(self, process: TargetProcess, observation, terminated: bool) -> float:
        reward = 0.0
        scale = 1e2

        if terminated:

            if process.distance_to_target <= 0:
                # reward = .25 + .75 * (1.0 - process.time / process.time_limit)
                # reward = .5 + .5 * (process.max_starting_distance_to_target / process.time)
                reward = reward + 1
                overall_time_reward = -1e-1 * process.time / self.time_limit
                reward = reward + overall_time_reward
            else:
                reward = reward + 1e-1 * (1 - process.distance_to_target / process.max_distance_from_target)
                # reward = reward - 0
        else:
            vector_to_target = process.target.position - process.ego_vehicle.position

            # speed_to_target = process.ego_vehicle.velocity.dot(vector_to_target.normalized())
            # speed_to_target_reward = 1.0e-2 * process.time_step_length * \
            #                          max(0.0, min(1.0, speed_to_target / Constants.max_speed))
            # reward = reward + speed_to_target_reward
            # speed_to_target_reward = 1.0e-2 * process.time_step_length * \
            #                          min(1.0, speed_to_target / Constants.max_speed)

            # speed = process.ego_vehicle.velocity.length
            # speed_reward = 1.0e-3 * process.time_step_length * min(1.0, speed / Constants.max_speed)
            # reward = reward + speed_reward

            # reward = reward + .001 * process.time_step_length * speed / Constants.max_speed
            # reward = reward - 1.0 * process.time_step_length

            # if terminated or random.random() < 1e-2:
            #     print(reward,
            #           speed_to_target / Constants.max_speed,
            #           speed_to_target_reward,
            #           speed / Constants.max_speed,
            #           speed_reward)
        return scale * reward
