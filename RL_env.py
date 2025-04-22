import gym #강화학습 라이브러리
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
from torchaudio import transforms as T
from torch.utils.data import DataLoader
import random

WINDOW_SIZE = 25
RANDOM_LOC = False
AUDIO_DIR = "audio_dir"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
Notes:
    Agent navigation for image classification. We propose
    an image classification task starting with a masked image
    where the agent starts at a random location on the image. It
    can unmask windows of the image by moving in one of 8 directions.
     At each timestep it
    also outputs a probability distribution over possible classes
    C. The episode ends when the agent correctly classifies the
    image or a maximum of 20 steps is reached. The agent receives a 
    -0.1 reward at each timestep that it misclassifies the
    image. The state received at each time step is the full image
    with unobserved parts masked out.

    -- for now, agent outputs direction of movement and class prediction (0-9)
    -- correct guess ends game
'''
class HeartSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation,
                 target_sample_rate
                 ,num_samples,device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples * 5
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return int(len(self.annotations))

    #len(SSD)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)

        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        signal = self.log_mel_spectrogram(signal)
        signal = (signal - signal.min()) / (signal.max() - signal.min())


        return signal, label

    def _cut_if_necessary(self, signal):
        total_samples = signal.shape[1]
        if total_samples > self.num_samples:
            start_sample = (total_samples - self.num_samples) // 2
            end_sample = start_sample + self.num_samples
            signal = signal[:, start_sample:end_sample]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self,signal, sr): #오직 타켓SR과 원래SR이 다를때 리셈플링)
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self,signal): #여러채널을 가진 오디오채널을 한개채널로 감소
        if signal.shape[0] > 1:    #(2,1000)
            signal = torch.mean(signal, dim = 0, keepdim=True)
        return signal

    def _get_audio_sample_path(self,index):
        fold = f"data"
        path = os.path.join(self.audio_dir, self.annotations.iloc[
            index, 0])

        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]

    def log_mel_spectrogram(self, mel_spec):
        log_mel_spec = torch.log(mel_spec + 1e-9)  # 로그 변환, 작은 값 추가로 수치 안정성 보장
        return log_mel_spec


def create_data_loader(train_data, val_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    return train_dataloader, val_dataloader


def load_hss( download=False, batch_size=100, shift_pixels=2):

    kwargs = {'num_workers': 1, 'pin_memory': True}
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        win_length=1024,
        hop_length=862,
        n_mels=128,
        norm='slaney'
    )
    # transform = torchaudio.transforms.MFCC(
    #     sample_rate=22050,
    #     n_mfcc=64,
    #     melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 64, "center": False},
    # )

    hss_train = HeartSoundDataset(r"C:\Users\asus\Desktop\interspeech 2024 experiment\binary_lab\labels_train.csv",
                              AUDIO_DIR,
                              mel_spectrogram,
                              SAMPLE_RATE,
                              NUM_SAMPLES,
                              'cpu')
    hss_test = HeartSoundDataset("/Users/asus/Desktop/interspeech 2024 experiment/binary_lab/labels_test.csv",
                              AUDIO_DIR,
                              mel_spectrogram,
                              SAMPLE_RATE,
                              NUM_SAMPLES,
                              'cpu')

    hss_val = HeartSoundDataset("/Users/asus/Desktop/interspeech 2024 experiment/binary_lab/labels_devel.csv",
                              AUDIO_DIR,
                              mel_spectrogram,
                              SAMPLE_RATE,
                              NUM_SAMPLES,
                              'cpu')

    #train_dataloader, val_dataloader = create_data_loader(ssd_t, ssd_v, BATCH_SIZE)


    #return train_dataloader, val_dataloader
    return hss_train, hss_test, hss_val



class HSSEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, type='train'): #UAR75.2%
        self.hss_train, self.hss_test, self.hss_eval = load_hss(download=False)


        x_train, y_train = ten_to_np(self.hss_train)
        x_test, y_test = ten_to_np(self.hss_test)
        x_val, y_val = ten_to_np(self.hss_eval)
        self.MAX_STEPS = 5

        if type == 'train':
            self.X = x_train
            self.Y = y_train
            self.n = len(y_train)



        elif type == 'test':
            self.X = x_test
            self.Y = y_test
            self.n = len(y_test)
            self.MAX_STEPS = 3


        elif type == 'val':
            self.X = x_val
            self.Y = y_val
            self.n = len(y_val)


        h, w = self.X[0].shape
        self.h = h // WINDOW_SIZE
        self.w = w // WINDOW_SIZE


        self.mask = np.zeros((h, w))


        # action is an integer in {0, ..., 280}
        # see 'step' for interpretation
        self.action_space = spaces.Discrete(16) # 8방향 * 2가지 예측
        #self.action_space = spaces.Discrete(8) 원래꺼
        self.observation_space = spaces.Box(0, 255, [h, w])


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # action consists of:
        #   1. direction in {N, S, E, W}, determined by a % 4
        #   2. predicted class (0 or 1), determined by floor(a / 4)
        assert (self.action_space.contains(action))
        dir, Y_pred = action % 8, action // 8
        self.predict_bool = False

        self.steps += 1

        move_map = {
            0: [0, 1],  # N
            1: [1, 1],  # NE
            2: [1, 0],  # E
            3: [1, -1],  # SE
            4: [0, -1],  # S
            5: [-1, -1],  # SW
            6: [-1, 0],  # W
            7: [-1, 1]  # NW
        }

        # Move the agent and reveal the square
        self.pos = np.clip(self.pos + move_map[dir], 0,
                           [self.h - 1, self.w - 1])  # Restrict movement within the bounds of the grid
        self._reveal()  # Reveal the area at the new position

        # State (observation) consists of a masked image (h x w)
        obs = self._get_obs()  # Get the current state (observation), which is used as input for the next learning step

        # -0.1 penalty for each additional timestep
        # +1.0 for correct prediction
        reward = -0.1 + int(
            Y_pred == self.Y[self.i])  # Define the reward structure: -0.1 per step and +1 for correct prediction

        # End the episode if the prediction is correct or max steps are reached
        done = Y_pred == self.Y[self.i] or self.steps >= self.MAX_STEPS
        if Y_pred == self.Y[self.i]:
            self.predict_bool = True
        else:
            self.predict_bool = False

        # Return observation, reward, done flag, and additional info
        return obs, reward, done, {}, self.predict_bool, Y_pred, self.Y[self.i]

    def reset(self):
        # Resets the environment and returns the initial observation
        # Reset the mask, move to a random location, and choose a new image

        # Initialize at a random location or image center
        if RANDOM_LOC:  # Set a random position in the environment
            self.pos = np.array([np.random.randint(self.h),
                                 np.random.randint(self.w)])
        else:
            self.pos = np.array([int(self.h / 2), int(self.w / 2)])  # Otherwise, start at the center of the image

        self.mask[:, :] = 0  # Initialize the mask to 0 (nothing revealed)
        self._reveal()  # Reveal part of the image
        self.i = np.random.randint(self.n)  # Select a random image from the dataset
        self.steps = 0  # Reset the step counter

        return self._get_obs()  # Return the initial observation

    def _get_obs(self):
        # Returns the current observation
        masked_img = self.X[self.i] * self.mask  # Apply the mask to the current image
        obs = masked_img / 255  # Normalize the image values
        return obs  # Return the masked and normalized image

    def _reveal(self):
        # Reveal a window at the current position (self.pos)

        h, w = self.pos  # Get the current position
        h_low, h_high = h * WINDOW_SIZE, (h + 1) * WINDOW_SIZE  # Calculate the vertical bounds of the window
        w_low, w_high = w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE  # Calculate the horizontal bounds of the window

        self.mask[h_low:h_high, w_low:w_high] = 1  # Set the window in the mask to 1, revealing that area

    """For visualization, you can modify this as needed"""

    def render(self, mode='rgb_array', close=False):
        # Display the mask, full image, and masked image

        plt.figure(figsize=(5, 3), dpi=300)
        plt.suptitle("Step %d" % self.steps)

        ticks = range(0, 128, 30)

        # Display the full original image
        plt.imshow(self.X[485])
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.savefig('original_image', dpi=300)

        # Show the visualization
        plt.show()


def ten_to_np(dataset):
    data_list = []
    label_list = []

    for i in range((len(dataset)) - 1):
        data, label = dataset[i]

        data_np = data.numpy()
        data_np = data_np.reshape(data_np.shape[1],data_np.shape[2])

        data_list.append(data_np)

        label_list.append(int(label))

    data_array = np.array(data_list)
    label_array = np.array(label_list)

    return (data_array,label_array)

if __name__ == "__main__":
    import argparse
    import os

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=30, type=int)
    parser.add_argument('--lr', default=0.0003, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.0005 * 784, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")  # num_routing should > 0
    parser.add_argument('--shift_pixels', default=2, type=int,
                        help="Number of pixels to shift at most in each direction.")
    parser.add_argument('--data_dir', default='./data',
                        help="Directory of data. If no data, use \'--download\' flag to download it")
    parser.add_argument('--download', action='store_true',
                        help="Download the required data.")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    #print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)



    hss_train, hss_test, hss_eval = load_hss(download=False, batch_size=args.batch_size)

    print(hss_train[0].shape)
