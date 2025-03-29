import gymnasium as gym
import torch
import numpy as np
from tqdm import trange
import os


from utils import get_model, TensorboardSummaries as Summaries


seed = 13
np.random.seed(seed)
torch.manual_seed(seed);


def train(**kwargs):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    timesteps_per_epoch = 1

    task = kwargs["task"]
    model = kwargs["model"]
    nepochs = kwargs["nepochs"]

    env = gym.make(task, render_mode="rgb_array", include_cfrc_ext_in_observation=False)
    env = Summaries(env, model + " on " + task);

    global_params = {}
    global_params["state_dim"] = env.observation_space.shape[0]
    global_params["action_dim"] = env.action_space.shape[0]
    global_params["timesteps_per_epoch"] = timesteps_per_epoch
    global_params["device"] = DEVICE
    global_params["env"] = env

    model = get_model(model, global_params)

    # start train
    
    for _ in trange(0, nepochs, timesteps_per_epoch):
        model.run_iter(env=env)


    # save actor
    dir_path = "save"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    torch.save({
        'model_state_dict': model.get_actor().state_dict(),
    }, f'{dir_path}/model_{model}_checkpoint.pth')
