
from utils import *

import os
from environment_PID import SimEnv
from config import env_params, action_map
from settings import *

def run():
    try:
        
        # state_dim = INPUT_DIMENSION
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        num_actions = 1 # a single continuous action
        episodes = 10000

        #replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        #model = DQN(num_actions, state_dim, in_channels, device)

        # this only works if you have a model in your weights folder. Replace this by that file
        #model.load('weights/model_ep_4400')

        # set to True if you want to run with pygame
        env = SimEnv(visuals=True, **env_params)

        for ep in range(episodes):
            env.create_actors()
            env.generate_episode(ep, eval=False)
            env.reset()
    finally:
        env.reset()
        env.quit()

if __name__ == "__main__":
    run()
