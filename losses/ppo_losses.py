import numpy as np
import torch


class GAE:
    """Generalized Advantage Estimator."""
    def __init__(self, policy, gamma=0.99, lambda_=0.95):
        self.policy = policy
        self.gamma = gamma
        self.lambda_ = lambda_

    def __call__(self, trajectory):
        gamma = self.gamma
        lambda_ = self.lambda_

        latest_observation = trajectory["state"]["latest_observation"]
        action = self.policy.act(latest_observation)

        values = trajectory["values"].flatten()
        values = np.append(values, action["values"])

        delta = 0.
        A = []
        for i in range(trajectory["rewards"].shape[0] - 1, -1, -1):
            reward = trajectory["rewards"][i]
            gamma_t = (1 - trajectory["resets"][i]) * gamma
            delta = reward + gamma_t * values[i + 1] - values[i] + gamma_t * lambda_ * delta
            A.append(delta)

        A.reverse()
        trajectory["advantages"] = torch.tensor(A)
        trajectory["value_targets"] = (trajectory["advantages"].flatten() + trajectory["values"].flatten()).reshape(-1, 1)
