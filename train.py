import wandb
from mnist_env import HSSEnv, ten_to_np
from actor_critic_agent import ActorCriticNNAgent
from PPO_agent import HSSNet, PPO,torch_to_numpy, numpy_to_torch
import torch.optim as optim
from torch import nn
import numpy as np
import time
import argparse
import sys
import torch
import matplotlib.pyplot as plt
import learn2learn as l2l
from mnist_env import load_hss
import random
import copy
import gc
import torch.nn.functional as F

UAR = []
SEN = []
SPE = []
train_times = []
num_seeds = 10
list_seeds = [486, 123, 456, 789, 1010, 2069, 3030, 4040, 5050, 6060]

wandb.login(key='5cd7de9df578ecd7951639d8b7c9274e559f8514')
STEPS = 0
MAX_STEPS = 700000
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_REWARD = []
REWARD_10 = []
REWARD_100 = []
REWARD_1000 = []


def create_episodic_data(data, n_way, k_shot, q_query, num_episodes):
    """
    data: list of tuples (input, label)
    n_way: number of classes per episode
    k_shot: number of samples per class in the support set
    q_query: number of samples per class in the query set
    num_episodes: number of episodes to generate
    """
    # Group data by class
    data_by_class = {}
    for x, y in data:
        if y not in data_by_class:
            data_by_class[y] = []
        data_by_class[y].append(x)

    # Create episodes
    episodes = []
    classes = list(data_by_class.keys())
    for _ in range(num_episodes):
        selected_classes = random.sample(classes, n_way)
        support_set = []
        query_set = []
        for cls in selected_classes:
            samples = random.sample(data_by_class[cls], k_shot + q_query)
            support_set.extend([(s, cls) for s in samples[:k_shot]])
            query_set.extend([(s, cls) for s in samples[k_shot:]])
        episodes.append((support_set, query_set))
    return episodes



class FocalLoss(nn.Module):
    def __init__(self, gamma=None, alpha=[0.5,0.5], size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.to(input.device)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def compute_loss(policy, dataset):
    loss = 0.0
    #criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류를 위한 손실 함수
    criterion =FocalLoss(gamma=2)
    for state, label in dataset:
        inputs = torch.tensor(state, dtype=torch.float32)
        inputs = obs_to_input(inputs)
        label = torch.tensor([label], dtype=torch.long)
        y1, y2 = policy(numpy_to_torch(inputs))
        loss += criterion(y1, label)
    return loss


def initialize_wandb(project, group, name, retries=5, wait=5):
    for attempt in range(retries):
        try:
            wandb.init(project=project, group=group, name=name, reinit=True)
            return
        except wandb.errors.CommError as e:
            print(f"WandB connection failed: {e}. Retrying ({attempt + 1}/{retries})...")
            time.sleep(wait)
    raise RuntimeError("Failed to initialize WandB after several retries.")

def transform_data(data):
    inputs, labels = data
    transformed_data = [(inputs[i], labels[i]) for i in range(len(labels))]
    return transformed_data


def main():
    ''' Argument parsing.'''
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',default= 64, type=int)
    parser.add_argument('--iters', default= 10000, type=int)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    global STEPS

    meta_iter = 20
    num_episodes = 5

"""for MAML algorthm , if you want use it just remove annotation and use trained model for make instance"""

    for seed in list_seeds:

        # ## MAML for HSSNet 시작
        # ###수정본시작
        # original_model = HSSNet()
        #
        # # MAML 래퍼를 사용하여 모델 준비
        # maml_model = l2l.algorithms.MAML(original_model, lr=0.0003, first_order=False)
        #
        # # 최적화기 설정
        # meta_optimizer = optim.Adam(maml_model.parameters(), lr=0.0001)
        #
        # start = time.time()
        #
        # n_way = 2
        # k_shot = 1
        # q_query = 10
        #
        #
        # print("Meta learning started...")
        #
        # for epoch in range(meta_iter):  # 전체 메타학습 과정이 몇번 이뤄지나?
        #     meta_loss = 0.0
        #     hss_train, _, _ = load_hss(download=False, batch_size=args.batch_size)
        #     task_data = ten_to_np(hss_train)
        #     data = transform_data(task_data)
        #     episodes = create_episodic_data(data, n_way, k_shot, q_query, num_episodes = num_episodes)
        #
        #     for support_set, query_set in episodes:  # 에포크당 작업 수가 어떻게 되나?
        #         # 적응 학습
        #         learner = maml_model.clone()
        #         support_loss = compute_loss(learner, support_set)
        #         learner.adapt(support_loss)
        #
        #         # Compute query loss on adapted clone
        #         query_loss = compute_loss(learner, query_set)
        #         meta_loss += query_loss
        #
        #     meta_optimizer.zero_grad()
        #     meta_loss.backward()
        #     meta_optimizer.step()
        #
        #     print(f"Iteration {epoch + 1}/{meta_iter}, Meta Loss: {meta_loss.item()}")
        #
        #
        # end = time.time()
        # print("metrix. Completed %d iterations in %.3f s" % \
        #       (meta_iter, end - start))
        #
        # ### 수정본 끝
        #
        #
        # trained_model = maml_model.module


       #wandb.init(project="HSSEnv",group="PPO",name=f'seed{seed}',reinit=True)
        initialize_wandb("0829A2C ", "results", f'seed{seed}')

        seed_all(seed)
        print("Training...")
        trained_agent = train(HSSNet(),args.iters, args.batch_size, verbose=args.verbose, seed = seed )
        #trained_agent = train(trained_model,args.iters, args.batch_size, verbose=args.verbose, seed = seed )
        test_agent = trained_agent.copy()

        print("Testing...")
        print("num_seed : ",seed)
        test(test_agent)
        STEPS = 0
        wandb.finish()



# 학습률 감소를 위한 함수
def adjust_learning_rate(optimizer, lr_decay_factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay_factor

def obs_to_input(obs):  # 관측값을 신경망의 입력 형식으로 변환한다. 입력을 1,28,28 형태로 재구성
    # reshape to (1, 256, 256)
    return obs[np.newaxis, ...]

def validate(agent, env, episodes=50):
    print("Validation...")
    total_rewards = []
    for _ in range(episodes):
        observation = env.reset()
        agent.new_episode()
        total_reward = 0
        done = False
        #agent.trainable = False
        while not done:
            action = agent.act(observation, env, display=False)
            observation, reward, done, _,_,_,_ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    agent.replay = []

    return np.mean(total_rewards)
#
def train(model,iterations, episodes, verbose=False, seed = None): # 반복횟수, 에피소드 수, 상세 로그 출력 여부 매개변수
    ''' Method to train a model, currently fixed on
    using ActorCritic; will change when updated to use
    modular DeepRL env'''
    global STEPS
    global MAX_STEPS
    # initialize agent
    agent = ActorCriticNNAgent(model, obs_to_input=obs_to_input, df=0.95)
    #agent = PPO(model, obs_to_input=obs_to_input, df=0.95)
    # Actor critic 객체를 생성해 에이전트 초기화,
    best_val_reward = -np.inf

    # intialize environment
    env = HSSEnv(type='train') #환경 초기화, train모드로 설정되어 있다.
    env.seed(seed)


    # training loop
    start = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    for iter in range(iterations): #iter단위로 반복실행

        if iter % 100 == 0: print("반복 횟수를 시작합니다. Starting iteration %d" % iter) #10번마다 한번씩 iter 횟수 보고
        rewards = []
        #print("iter:", iter)

        # play out each episode 여러 에피소드를 통해 에이전트 학습
        for ep in range(episodes):
            #print("ep:",ep)

            if verbose and iter % 100 == 0 and ep == 0:
                display = True
            else:
                display = False

            observation = env.reset()
            agent.new_episode()
            total_reward = 0

            done = False

            while not done:
                action = agent.act(observation, env, display=display)
                observation, reward, done, info, _, _, _ = env.step(action)

                if display: print("실제 보상 값. Actual reward:", reward)
                agent.store_reward(reward)  # 선택에 대한 보상 저장, 총 보상 누적.
                total_reward += reward
                STEPS+=1


            # while 루프가 정상적으로 종료된 경우에만 실행
            rewards.append(total_reward)  # 총 보상으로 보상값 축적.
            avg_train_reward = np.mean(rewards)


        agent.update()
        #     rewards.append(total_reward) # 총 보상으로 보상값 축척.
        #     avg_train_reward = np.mean(rewards)
        #
        # agent.update()  # 에피소드를 통해 학습된 결과로 에이전트를 업데이트한다.


        # print performance for this iteration
        ALL_REWARD.append(np.mean(rewards))
        if iter % 10 == 0 :
            REWARD_10.append(np.mean(rewards))


        if iter % 100 == 0:
            print("STEPS : ", STEPS)

            REWARD_100.append(np.mean(rewards))
            gc.collect()
            print("각 반복의 성능출력. 평균 총 보상 출력 Mean total reward / episode: %.3f" % np.mean(rewards))
            # print("Testing...")
            # print("num_seed : ", seed)
            # test(agent)

            # env_val = HSSEnv(type='val')
            # env_val.seed(seed + 10)
            # print("Starting velidation %d" % iter)
            # val_reward = validate(agent, env_val)
            # print(f"Iteration {iter}: Train Reward = {avg_train_reward:.4f}, Validation Reward = {val_reward:.4f}")
            # if val_reward > best_val_reward:
            #     best_val_reward = val_reward
            # else:
            #     adjust_learning_rate(optimizer, lr_decay_factor=0.3)

        if iter % 1000 == 0:
            #print("각 반복의 성능출력. 평균 총 보상 출력 Mean total reward / episode: %.3f" % np.mean(rewards))
            REWARD_1000.append(np.mean(rewards))

            env_val = HSSEnv(type='val')
            env_val.seed(seed+10)
            print("Starting velidation %d" % iter)
            val_reward = validate(agent, env_val)
            print(f"Iteration {iter}: Train Reward = {avg_train_reward:.4f}, Validation Reward = {val_reward:.4f}")
            if val_reward > best_val_reward:
                best_val_reward = val_reward
            else:
                adjust_learning_rate(optimizer, lr_decay_factor=0.3)


        wandb.log({'Reward': avg_train_reward, 'STEPS': STEPS})

        #
        # if iter % 2500 == 0:
        #         print("Starting velidation %d" % iter)
        #         val_reward = validate(agent, env_val)
        #         print(f"Iteration {iter}: Train Reward = {avg_train_reward:.4f}, Validation Reward = {val_reward:.4f}")
        #         if val_reward > best_val_reward:
        #             best_val_reward = val_reward
        #         else:
        #             adjust_learning_rate(optimizer, lr_decay_factor=0.3)
        # adjust agent parameters based on played episodes

        if STEPS >= MAX_STEPS:
            print(f'Max steps {MAX_STEPS} reached. Stopping training.')
            end = time.time()
            print("학습에 걸링 총 시간 반환. Completed %d iterations of %d episodes in %.3f s" % \
                  (iterations, episodes, end - start))
            train_times.append(end - start)
            return agent


    # return trained agent
    return agent



# def val(agent, n_test=900):  # 815
#     # calculate test average reward
#     print("validating...")
#
#     env = HSSEnv(type='val', seed=None)
#     tp = [0, 0]
#     fn = [0, 0]
#
#     rewards = []
#
#     start = time.time()
#     for _ in range(n_test):
#
#
#         observation = env.reset()
#         total_reward = 0
#
#         done = False
#         while not done:
#
#             action = agent.act(observation, env, display=False)
#             observation, reward, done, info, predict_bool, Y_pred, Y_label = env.step(action)
#             total_reward += reward
#
#             predicted_class = Y_pred
#             actual_class = Y_label
#
#             if predicted_class == actual_class:
#                 tp[actual_class] += 1
#             else:
#                 fn[actual_class] += 1
#
#
#         rewards.append(total_reward)
#
#
#     recall = [tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] > 0 else 0 for i in range(len(tp))]
#
#     uar = sum(recall) / len(recall)
#     print("Unweighted Average Recall (UAR): %.3f" % uar)
#
#     print("Mean total reward / episode: %.3f" % np.mean(rewards))
#
#     end = time.time()
#     print("validation time %.3f s" % \
#           (end - start))


def test(agent, n_test=815): # 815
    # calculate test average reward
    # env = HSSEnv(type='val')
    # tp = [0, 0]
    # fn = [0, 0]
    #
    #
    # rewards = []
    #
    # start = time.time()
    # for _ in range(n_test):
    #
    #     observation = env.reset()
    #     total_reward = 0
    #
    #     done = False
    #     while not done:
    #
    #         action = agent.act(observation, env, display=False)
    #         observation, reward, done, info, predict_bool, Y_pred, Y_label = env.step(action)
    #         total_reward += reward
    #
    #         predicted_class = Y_pred
    #         actual_class = Y_label
    #
    #         if predicted_class == actual_class:
    #             tp[actual_class] += 1
    #
    #         else:
    #             fn[actual_class] += 1
    #
    #
    #     # if predict_bool is True:
    #     #     true_pred += 1
    #     rewards.append(total_reward)
    #
    # print(tp[0], tp[1])
    # print(fn[0], fn[1])
    #
    # recall = [tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] > 0 else 0 for i in range(len(tp))]
    #
    # uar = sum(recall) / len(recall)
    # print("Unweighted Average Recall (UAR): %.3f" % uar)
    # UAR.append(uar)
    #
    # print("Mean total reward / episode: %.3f" % np.mean(rewards))
    # end = time.time()
    # print("testing time %.3f s" % \
    #       (end - start))
    # print(accuracy)
    # print(accuracydd)


    env = HSSEnv(type='test')
    # tp = [0,0]
    # fn = [0,0]
    #
    #
    # rewards = []
    #
    # start = time.time()
    # for _ in range(n_test):
    #
    #
    #     observation = env.reset()
    #     total_reward = 0
    #
    #     done = False
    #     while not done:
    #
    #         action = agent.act(observation, env, display=False)
    #         observation, reward, done, info, predict_bool,Y_pred, Y_label = env.step(action)
    #         total_reward += reward
    #
    #         predicted_class = Y_pred
    #         actual_class = Y_label
    #
    #         if predicted_class == actual_class:
    #             tp[actual_class] += 1
    #
    #         else:
    #             fn[actual_class] += 1
    #
    #
    #
    #     # if predict_bool is True:
    #     #     true_pred += 1
    #     rewards.append(total_reward)
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    rewards = []

    start = time.time()
    for _ in range(n_test):

        observation = env.reset()
        total_reward = 0

        done = False
        while not done:

            action = agent.act(observation, env, display=False)
            observation, reward, done, info, predict_bool, Y_pred, Y_label = env.step(action)
            total_reward += reward

            predicted_class = Y_pred
            actual_class = Y_label

            if predicted_class == 1 and actual_class == 1:
                tp += 1  # True Positive (양성을 양성으로 예측)
            elif predicted_class == 0 and actual_class == 0:
                tn += 1  # True Negative (음성을 음성으로 예측)
            elif predicted_class == 1 and actual_class == 0:
                fp += 1  # False Positive (음성을 잘못 양성으로 예측)
            elif predicted_class == 0 and actual_class == 1:
                fn += 1  # False Negative (양성을 잘못 음성으로 예측)

        rewards.append(total_reward)

    # 계산된 TP, TN, FP, FN 값 출력
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    # 전체 모델의 SEN (Sensitivity) 계산
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 전체 모델의 SPE (Specificity) 계산
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0

    # UAR (Unweighted Average Recall) 계산
    # UAR은 양성 클래스와 음성 클래스에 대한 리콜(민감도)의 평균입니다.
    print("sen : %.3f" % sen)
    print("spe : %.3f" % spe)
    recall_positive = sen  # 양성 클래스에 대한 민감도는 이미 계산된 SEN입니다.
    recall_negative = spe  # 음성 클래스에 대한 민감도는 SPE와 동일합니다.

    uar = (recall_positive + recall_negative) / 2

    #recall = [tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] > 0 else 0 for i in range(len(tp))]

    #uar = sum(recall) / len(recall)
    print("Unweighted Average Recall (UAR): %.3f" % uar)
    UAR.append(uar)
    SEN.append(sen)
    SPE.append(spe)

    #print("Mean total reward / episode: %.3f" % np.mean(rewards))
    end = time.time()
    print("testing time %.3f s" % \
         (end - start))

if __name__ == '__main__':


    main()
    print("UAR : " , UAR)
    print("SEN : ", SEN)
    print("SPE : ", SPE)
    print("train_times : ",train_times)
    # plt.plot(ALL_REWARD, label='A2C')
    # plt.legend()
    #
    # plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # plt.yticks(rotation=45)
    #
    # # 그래프에 제목 추가
    #
    # # X축과 Y축에 라벨 추가
    # plt.xlabel("Iters*10")
    # plt.ylabel("episode return")
    #
    # # 그래프 표시
    # plt.show()
    # plt.savefig('result.png', dpi=300)
    #
    # print(REWARD_10)
    # print(REWARD_100)
    # print(REWARD_1000)

