from RL_env import HSSEnv
import numpy as np
import torchaudio
import matplotlib.pyplot as plt

# Run this file to test if the environment is working

env = HSSEnv(type='train')

obs = env.reset()
done = False

while not done:

    env.render()
    action = env.action_space.sample()
    print(action)
    dir, Y_pred = action % 8, action // 8
    print("Agent moved %s" % (['N','NE','E','SE','S','SW',"W","NW"][dir]))
    print("Agent guessed %d" % Y_pred)

    _, reward, done, _,_,_,_ = env.step(action)
    print\
        ("Received reward %.1f on step %d" % (reward, env.steps))


