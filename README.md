# MetaRL_heartsound
This is a project we started for EMBC2025. In this project, we extracted spectrograms from existing heart sound data and reimplemented the spectrograms into a grid world to set up a reinforcement learning environment. Then, we trained and evaluated agents using two reinforcement learning algorithms, A2C and PPO, in this environment. In addition, to solve the inherent problems of medical data, we tried to train an agent that can perform generalized performance with less data using MAML.


## 🔧 Dependencies

Install required libraries in requirements.txt.

🚀 Training (Meta-RL)
You can train using either MAML + Actor-Critic or just plain Actor-Critic:
1. Modify training configuration in train.py
# Whether to use MAML
use_maml = True   # or False


2. Launch training
python train.py --batch_size 64 --iters 10000 --verbose
All training progress will be logged to Weights & Biases.
You have to set the key for wandb from train.py

