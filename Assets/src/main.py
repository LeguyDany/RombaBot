from Assets.src.model.policy_network import PolicyNetwork
from Assets.src.model.value_network import ValueNetwork
from utils.unity_env import UnityEnv
from utils.trainer import Trainer

env = UnityEnv()
value_model = ValueNetwork(env.obs_size, env.action_size)
policy_model = PolicyNetwork(env.obs_size, env.action_size)
trainer = Trainer(
    policy_model=policy_model, 
    value_model=value_model, 
    unityEnv=env
)

trainer.train(num_episodes=100)
