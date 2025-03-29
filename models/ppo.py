import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


class PolicyModel(nn. Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.policy_model = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.action_dim * 2),
        )

        self.value_model = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
        )

    def get_policy(self, x):
        means, var = self.policy_model(x).split(self.action_dim, dim=-1)
        var = F.softplus(var)
        return means, var

    def get_value(self, x):
        out = self.value_model(x.float())
        return out

    def forward(self, x):
        policy = self.get_policy(x)
        value = self.get_value(x)

        return policy, value
    
class Policy:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def act(self, inputs, training=False):
        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)

        if inputs.ndim == 1:
          inputs = inputs.unsqueeze(0)

        mean, var = self.model.get_policy(inputs)
        cov_matrix = torch.diag_embed(var).to(self.device)
        normal_distr = MultivariateNormal(mean, cov_matrix)

        actions = normal_distr.sample()
        log_probs = normal_distr.log_prob(actions)

        values =  self.model.get_value(inputs)

        if training:
            return {"distribution": normal_distr, "values": values.squeeze()}

        return {
            "actions": actions.cpu().numpy().squeeze(),
            "log_probs": log_probs.detach().cpu().numpy().squeeze(),
            "values": values.detach().cpu().numpy().squeeze(),
        }
