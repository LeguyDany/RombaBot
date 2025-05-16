import torch
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(self, model, env):
        self.model = model
        self.env = env
    
    def train(self, num_episodes):
        self.env.reset()

        with tqdm(total=num_episodes, desc=f"Episode", unit="batch") as pbar:
            for episode in range(num_episodes):
                self.env.reset()
                decision_steps, _ = self.env.get_steps()

                while len(decision_steps) > 0:
                    actions = []

                    for agent_id in decision_steps.agent_id:
                        obs_list = decision_steps[agent_id].obs
                        obs = np.concatenate([o.flatten() for o in obs_list])
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

                        logits, value = self.model(obs_tensor)

                        action = torch.argmax(logits, dim=1).detach().cpu().numpy()

                        probs = torch.softmax(logits, dim=1)
                        action = torch.multinomial(probs, num_samples=1).detach().cpu().numpy()

                        actions.append(action)

                    self.env.set_actions(actions)
                    self.env.step()
                    decision_steps, _ = self.env.get_steps()

                pbar.update(1)

        self.env.end()