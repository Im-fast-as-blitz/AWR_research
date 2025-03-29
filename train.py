import gymnasium as gym
from logger import TensorboardSummaries as Summaries
import torch
import numpy as np
from tqdm import trange
import os


from utils import parse_args, get_model


seed = 13
np.random.seed(seed)
torch.manual_seed(seed);


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()

    env = gym.make(args.task, render_mode="rgb_array", include_cfrc_ext_in_observation=False)
    env = Summaries(env, args.model + " on " + args.task);

    global_params = {}
    global_params["state_dim"] = env.observation_space.shape[0]
    global_params["action_dim"] = env.action_space.shape[0]
    global_params["device"] = DEVICE

    timesteps_per_epoch=1

    model = get_model(args.model, global_params)

    # start train
    for _ in trange(0, args.nepochs, timesteps_per_epoch):
        model.run_iter(env=env)


    # save actor
    dir_path = "save"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    torch.save({
        'model_state_dict': model.get_actor().state_dict(),
    }, f'{dir_path}/model_{args.model}_checkpoint.pth')


if __name__ == "__main__":
    main()
