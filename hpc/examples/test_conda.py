import time

import ray

ray.init()
print("===OK: ray.init() did not crash")

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

print("===Running PPO Trainer, ctrl-c to exit")
time.sleep(3)

try:
    tune.run(PPOTrainer, config={"env": "CartPole-v0"})
except Exception as e:
    print("stopped traininer on exception: {}".format(e))
