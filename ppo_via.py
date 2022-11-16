import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
from torch.distributions.multivariate_normal import MultivariateNormal

class PPOmemory:
    def __init__(self, mini_batch_size):
        self.states = []  # 状态
        self.actions = []  # 实际采取的动作
        self.probs = []  # 动作概率
        self.vals = []  # critic输出的状态值
        self.rewards = []  # 奖励


        self.mini_batch_size = mini_batch_size  # minibatch的大小

    def sample(self):
        n_states = len(self.states)  # memory记录数量=20
        batch_start = np.arange(0, n_states, self.mini_batch_size)  # 每个batch开始的位置[0,5,10,15]
        indices = np.arange(n_states, dtype=np.int64)  # 记录编号[0,1,2....19]
        np.random.shuffle(indices)  # 打乱编号顺序[3,1,9,11....18]
        mini_batches = [indices[i:i + self.mini_batch_size] for i in batch_start]  # 生成4个minibatch，每个minibatch记录乱序且不重复

        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards),  mini_batches

    # 每一步都存储trace到memory
    def push(self, state, action, prob, val, reward):
        self.states.append(state)
        self.actions.append(action.numpy())
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)


    # 固定步长更新完网络后清空memory
    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []



# actor:policy network
class Actor(nn.Module):
    def __init__(self, n_states, n_actions, cfg, chkpt_dir='/home/zhiyuan/notebook_script/cem_promp/ppo_model'):
        super(Actor, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.stds = torch.Tensor([ 0.1,  0.1, 0.01])
        # nn.Parameter(torch.ones(n_actions))*0.1

        self.actor = nn.Sequential(
            nn.Linear(n_states, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, n_actions),
            nn.Sigmoid()
            )

    def forward(self, state):
        mu= self.actor(state)
        k=torch.Tensor([8,8,1])

        dist = MultivariateNormal(loc=mu*k,scale_tril=torch.diag(self.stds) )
        # print(self.actor(state))
        # print('look')
        # action = dist.sample()
        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

# critic:value network
class Critic(nn.Module):
    def __init__(self, n_states, cfg, chkpt_dir='/home/zhiyuan/notebook_script/cem_promp/ppo_model'):
        super(Critic, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(n_states, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, 1))

    def forward(self, state):
        value = self.critic(state)
        return value
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_states, n_actions, cfg):
        # 训练参数
        self.gamma = cfg.gamma  # 折扣因子
        self.n_epochs = cfg.n_epochs  # 每次更新重复次数
        self.gae_lambda = cfg.gae_lambda  # GAE参数
        self.policy_clip = cfg.policy_clip  # clip参数
        self.device = cfg.device  # 运行设备

        # AC网络及优化器
        self.actor = Actor(n_states, n_actions, cfg)
        self.critic = Critic(n_states, cfg)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # 经验池
        self.memory = PPOmemory(cfg.mini_batch_size)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)  # 数组变成张量
        dist = self.actor(state)  # action分布
        value = self.critic(state)  # state value值
        action = dist.sample()  # 随机选择action
        prob = torch.squeeze(dist.log_prob(action)).item() 
        # prob = dist.log_prob(action)
        value = torch.squeeze(value).item()
        return action, prob,value

    def learn(self):
        for _ in range(self.n_epochs):
            # memory中的trace以及处理后的mini_batches，mini_batches只是trace索引而非真正的数据
            states_arr, actions_arr, old_probs_arr, vals_arr,\
                rewards_arr, mini_batches = self.memory.sample()

            # 计算GAE
            advantage = rewards_arr-vals_arr
            advantage = torch.tensor(advantage).to(self.device)
            values = vals_arr[:]
            # mini batch 更新网络
            values = torch.tensor(values).to(self.device)
            for batch in mini_batches:
                states = torch.tensor(states_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_probs_arr[batch]).to(self.device)
                actions = torch.tensor(actions_arr[batch]).to(self.device)

                # mini batch 更新一次critic和actor的网络参数就会变化
                # 需要重新计算新的dist,values,probs得到ratio,即重要性采样中的新旧策略比值
                dist = self.actor(states)
                critic_value = torch.squeeze(self.critic(states))
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                # actor loss
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clip_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,\
                     1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clip_probs).mean()
                # critic loss
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()
                # total_loss
                total_loss = actor_loss + 0.5 * critic_loss

                # 更新
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optim.step()
                self.critic_optim.step()
        self.memory.clear()
