def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []
    steps = 0
    for i_ep in range(cfg.train_eps):
        traj= 0
        while traj < 20:
            state = env.reset()
            action, prob, val = agent.choose_action(state)
            reward = get_reward(state,action)
            agent.memory.push(state, action, prob, val, reward, done)
            traj+=1
        agent.learn()
        
        if (i_ep + 1) % 10 == 0:
            print(f"episode: {i_ep + 1}/{cfg.train_eps},reward:{ep_reward:.2f}")

    agent.save_models()
    x = [i+1 for i in range(len(rewards))]
    plt.plot(x, rewards)
    plt.show()
    plt.savefig(figure_file)
    print('完成训练！')


        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, prob, val = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            steps += 1
            ep_reward += reward
            agent.memory.push(state, action, prob, val, reward, done)
            if steps % cfg.batch_size == 0:
                agent.learn()
            state = state_
        rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print(f"回合：{i_ep + 1}/{cfg.train_eps}，奖励：{ep_reward:.2f}")
    print('完成训练！')

def train(cfg, agent):
    print('开始训练！')
    print(f' 算法：{cfg.algo_name}, 设备：{cfg.device}')
    figure_file = 'ppo_via.png'
    rewards = []
    for i_ep in range(cfg.train_eps):
        state = env_reset()
        action, prob,val = agent.choose_action(state)
        reward = get_reward(state,action)
        ep_reward = reward
        agent.memory.push(state, action, prob,val, reward)
        agent.learn()
        rewards.append(ep_reward)
        # print(rewards)
        if (i_ep + 1) % 10 == 0:
            print(f"episode: {i_ep + 1}/{cfg.train_eps},reward:{ep_reward:.2f}")
    agent.save_models()
    x = [i+1 for i in range(len(rewards))]
    plt.plot(x, rewards)
    plt.show()
    plt.savefig(figure_file)
    print('完成训练！')
