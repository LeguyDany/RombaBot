import torch
import numpy as np
from tqdm import tqdm
from utils.math import Math
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss

class Trainer:
    def __init__(
            self,
            policy_model,
            value_model,
            unityEnv,
            clip_epsilon=0.2,
            gamma=0.99,
            lmbda=0.95,
            entropy_epsilon=1e-4,
            device="cpu"
        ):
        self.policy_model = policy_model
        self.value_model = value_model
        self.unityEnv = unityEnv
        self.episodes_data = {}

        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_model,
            critic_network=self.value_model,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_epsilon),
            entropy_coef=entropy_epsilon,
            critic_coef=1.0,
            loss_critic_type="smooth_l1",
        )

        self.advantage_module = GAE(
            gamma=gamma,
            lmbda=lmbda,
            value_network=value_model,
            average_gae=True,
            device=device,
        )
        self.optim = torch.optim.Adam(self.loss_module.parameters(), self.lr)

    
    def train(self, num_episodes):
        self.unityEnv.reset()

        with tqdm(total=num_episodes, desc=f"Episode", unit="batch") as pbar:
            for episode in range(num_episodes):
                self.unityEnv.reset()
                decision_steps, _ = self.unityEnv.get_steps()

                results = self.__training_step(decision_steps)
                advantage = Math.compute_value_loss(results.rewards, results.values)

                value_loss = self.__back_propagate_value_network(advantage, results)
                policy_loss = self.__back_propagate_policy_network(advantage, results)

                self.episodes_data[episode] = results
                pbar.update(1)

        self.unityEnv.end()

    
    def __training_step(self, decision_steps):
        """
            Returns: actions, values, rewards, states
        """
        while len(decision_steps) > 0:
            actions = []
            rewards = []
            values = []
            states = []

            for agent_id in decision_steps.agent_id:
                obs_list = decision_steps[agent_id].obs
                obs = np.concatenate([o.flatten() for o in obs_list])
                obs = Math.standardize([obs])
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

                logits = self.policy_model(obs_tensor)
                value = self.value_model(obs_tensor)

                action = torch.argmax(logits, dim=1).detach().cpu().numpy()

                probs = torch.softmax(logits, dim=1)
                action = torch.multinomial(probs, num_samples=1).detach().cpu().numpy()

                actions.append(action)
                values.append(value)
                rewards.append(decision_steps[agent_id].reward)

            self.unityEnv.set_actions(actions)
            self.unityEnv.step()
            decision_steps, _ = self.unityEnv.get_steps()

            return {
                "actions": actions,
                "rewards": rewards,
                "values": values,
                "states": states,
            }

    def __back_propagate_value_network(self, advantage, results):
        pass

    def __back_propagate_policy_network(self, advantage, results):
        pass
