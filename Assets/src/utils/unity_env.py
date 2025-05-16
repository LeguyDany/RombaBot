from mlagents_envs.environment import UnityEnvironment
import numpy as np
from mlagents_envs.environment import ActionTuple

class UnityEnv:
    ENV_PATH = "/Users/danyleguy/Documents/EPITECH/MSc2/AI/mujoco/mujoco_RL/Romba1.app"

    def __init__(self, action_size: int = 4, env_path: str = ENV_PATH):
        self.env = UnityEnvironment(file_name=env_path)
        self.env.reset()

        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        spec = self.env.behavior_specs[self.behavior_name]

        self.obs_size = int()
        self.action_size = action_size

        for specs in spec.observation_specs:
            for shape in specs[0]:
                self.obs_size += shape

    def reset(self):
        self.env.reset()

    def end(self):
        self.env.close()

    def step(self):
        self.env.step()

    def get_steps(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        return decision_steps, terminal_steps

    def set_actions(self, actions):
        action_tuple = ActionTuple()
        action_tuple.add_discrete(np.vstack(actions))
        self.env.set_actions(self.behavior_name, action_tuple)