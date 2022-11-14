class Agent:
    def __init__(self, n_states, n_actions, cfg):
        # 训练参数
        self.gamma = cfg.gamma  # 折扣因子
        self.n_epochs = cfg.n_epochs  # 每次更新重复次数
        self.gae_lambda = cfg.gae_lambda  # GAE参数
        self.policy_clip = cfg.policy_clip  # clip参数
        self.device = cfg.device  # 运行设备

        # AC网络及优化器
        self.actor = Actor(n_states, n_actions, cfg.hidden_dim)
        self.critic = Critic(n_states, cfg.hidden_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # 经验池
        self.memory = PPOmemory(cfg.mini_batch_size)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)  # 数组变成张量
        dist = self.actor(state)  # action分布
        value = self.critic(state)  # state value值
        action = dist.sample()  # 随机选择action
        prob = torch.squeeze(dist.log_prob(action)).item()  # action对数概率

        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return action, prob, value

    def learn(self):
        for _ in range(self.n_epochs):
            # memory中的trace以及处理后的mini_batches，mini_batches只是trace索引而非真正的数据
            states_arr, actions_arr, old_probs_arr, vals_arr,\
                rewards_arr, dones_arr, mini_batches = self.memory.sample()

            # 计算GAE
            values = vals_arr[:]
            advantage = np.zeros(len(rewards_arr), dtype=np.float32)
            for t in range(len(rewards_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards_arr) - 1):
                    a_t += discount * (rewards_arr[k] + self.gamma * values[k+1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)

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
