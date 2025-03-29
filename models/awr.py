import torch
from torch import nn, distributions


class ActorAWR(nn.Module):
    def __init__(self, **hyper_ps):
        super(ActorAWR, self).__init__()

        hidden_size = hyper_ps['a_hidden_size']
        hidden_layers = hyper_ps['a_hidden_layers']
        action_dim = hyper_ps['action_dim']

        fcs = [nn.Linear(hyper_ps['state_dim'], hidden_size), nn.ReLU()]
        for _ in range(hidden_layers):
            fcs.append(nn.Linear(hidden_size, hidden_size))
            fcs.append(nn.ReLU())

        self.fc_base = nn.Sequential(*fcs)
        self.fc_mean = nn.Linear(hidden_size, action_dim)
        self.fc_logsd = nn.Linear(hidden_size, action_dim)

        self.optimiser = torch.optim.SGD(
            lr=hyper_ps['a_learning_rate'],
            momentum=hyper_ps['a_momentum'],
            params=self.parameters()
        )

    def forward(self, state):
        x = self.fc_base(state)
        mean = self.fc_mean(x)
        sd = self.fc_logsd(x).exp()

        normal = distributions.Normal(loc=mean, scale=sd)
        action = normal.sample()

        return normal, action

    def backward(self, loss):
        self.optimiser.zero_grad()
        loss.backward(torch.tensor([1. for _ in range(loss.size()[0])]))
        self.optimiser.step()

    def evaluate(self, x):
        self.eval()
        with torch.no_grad():
            y = self.forward(x)
        self.train()

        return y


class CriticAWR(nn.Module):
    def __init__(self, **hyper_ps):
        super(CriticAWR, self).__init__()

        hidden_size = hyper_ps['c_hidden_size']
        hidden_layers = hyper_ps['c_hidden_layers']

        fcs = [
            nn.Linear(hyper_ps['state_dim'], hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
        ]
        for _ in range(hidden_layers):
            fcs.append(nn.Linear(hidden_size, hidden_size))
            fcs.append(nn.ReLU())
            fcs.append(nn.BatchNorm1d(hidden_size))
        fcs.append(nn.Linear(hidden_size, 1))

        self.fc = nn.Sequential(*fcs)

        self.optimiser = torch.optim.SGD(
            lr=hyper_ps['c_learning_rate'],
            momentum=hyper_ps['c_momentum'],
            params=self.parameters()
        )
        self.criterion = torch.nn.MSELoss()
        

    def forward(self, state):
        return self.fc(state)

    def backward(self, out, target):
        loss = self.criterion(out, target.float())

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss
    
    def evaluate(self, x):
        self.eval()
        with torch.no_grad():
            y = self.forward(x)
        self.train()

        return y
