import numpy as np

class Math:
    @staticmethod
    def standardize(lists):
        all_values = np.concatenate(lists)
        mean = np.mean(all_values)
        std = np.std(all_values)
        standardized = [(x - mean) / std for x in all_values]
        return standardized

    @staticmethod
    def compute_value_loss(rewards, values):
        advantages = [reward - value for reward, value in zip(rewards, values)]
        mean = np.mean(advantages)
        MSE = np.sum(np.sqrt(advantages / mean))

        return MSE