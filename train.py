import gymnasium as gym
from logger import TensorboardSummaries as Summaries
import torch
import numpy as np
from tqdm import tqdm


from models import SAC


seed = 13
np.random.seed(seed)
torch.manual_seed(seed);


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make()
    env = Summaries(env, ...);

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 

    gamma=0.99                    # discount factor
    max_buffer_size = 10**5       # size of experience replay
    start_timesteps = 5000        # size of experience replay when start training
    timesteps_per_epoch=1         # steps in environment per step of network updates
    batch_size=128                # batch size for all optimizations
    max_grad_norm=10              # max grad norm for all optimizations
    tau=0.005                     # speed of updating target networks
    policy_update_freq=1          # frequency of actor update; vanilla choice is 1 for SAC
    alpha=0.1                     # temperature for SAC

    model = ....to(DEVICE)

    # start train
    for n_iterations in tqdm(range(0, 1000000, timesteps_per_epoch)):
        loss = model.run_iter(env=env)


if __name__ == "__main__":
    main()
