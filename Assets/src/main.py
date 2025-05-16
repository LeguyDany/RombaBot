import os
import sys

from model.base_mlp import RombaNetwork
from utils.unity_env import UnityEnv
from utils.trainer import Trainer

# src_dir = os.path.join(os.getcwd())
# sys.path.append(os.path.abspath(src_dir))

env = UnityEnv()
model = RombaNetwork(env.obs_size, env.action_size)
trainer = Trainer(model, env)

trainer.train(num_episodes=100)


# env_path = "/Users/danyleguy/Documents/EPITECH/MSc2/AI/mujoco/mujoco_RL/Romba1.app"

# env = UnityEnvironment(file_name=env_path)
# env.reset()

# behavior_name = list(env.behavior_specs.keys())[0]
# spec = env.behavior_specs[behavior_name]


# obs_size = int()
# action_size = int()

# for specs in spec.observation_specs:
#     for shape in specs[0]:
#         obs_size += shape

# model = RombaNetwork(obs_size, 4)

# for episode in range(100):
#     env.reset()
#     decision_steps, terminal_steps = env.get_steps(behavior_name)

#     while len(decision_steps) > 0:
#         actions = []

#         for agent_id in decision_steps.agent_id:
#             obs_list = decision_steps[agent_id].obs
#             obs = np.concatenate([o.flatten() for o in obs_list])
#             obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

#             logits, value = model(obs_tensor)

#             action = torch.argmax(logits, dim=1).detach().cpu().numpy()

#             probs = torch.softmax(logits, dim=1)
#             action = torch.multinomial(probs, num_samples=1).detach().cpu().numpy()

#             actions.append(action)

#         action_tuple = ActionTuple()
#         action_tuple.add_discrete(np.vstack(actions))
#         env.set_actions(behavior_name, action_tuple)
#         env.step()

#         decision_steps, terminal_steps = env.get_steps(behavior_name)

# env.close()
