import intprim
from intprim.probabilistic_movement_primitives import *
import matplotlib.pyplot as plt
from intprim.util.kinematics import BaseKinematicsClass
from cem_promp.cem import CEM
from cem_promp.utils import *
import time
from pytorch_fid.fid_score import calculate_frechet_distance
from ppo_via import  Agent
import argparse

dataset= np.load('/home/zhiyuan/notebook_script/0930/l_shape.npy')
num_joints =2
basis_model = intprim.basis.GaussianModel(8, 0.1, ["x","y"])
promp = ProMP(basis_model)
Q = dataset.transpose(0,2,1)

for i in range(len(Q)):
	promp.add_demonstration(Q[i])
n_samples = 30
domain = np.linspace(0,1,100)

mean_margs =  np.zeros([2,100])
sigma_s = np.zeros([2,2,100])
# stdqs =  np.zeros([2,100])
for i in range(len(domain)):
    mu_marg_q, Sigma_marg_q = promp.get_marginal(domain[i])
    sigma_s[:,:,i] = Sigma_marg_q
    mean_margs[:,i] = mu_marg_q

# prepared for fid score
mu_1= mean_margs.T.reshape(-1)
sigma_1 = np.zeros([200,200])
for i in range(100):
    sigma_1[i*2:(i+1)*2,i*2:(i+1)*2] = sigma_s[:,:,i]


# state: start, end , obstcle   长和宽为一的正方形 只要确定左下角的x和y

def get_reward(state, action):
       
    old_promp=promp
    action = action.reshape([-1,3])
    t_cond = np.zeros(2+action.shape[0])
    t_cond[0]=0
    t_cond[-1] = 1 
    for i in range(action.shape[0]):
        t_cond[i+1]=action[i,2]

    q_cond =np.zeros([2+action.shape[0],2])
    q_cond[0]= np.array([state[0],state[1]])
    q_cond[-1]=np.array([state[2],state[3]])
    for i in range(action.shape[0]):
        q_cond[i+1]=action[i,:2]
        
    mu_w_cond_rec, Sigma_w_cond_rec=old_promp.get_basis_weight_parameters()

    for i in range(t_cond.shape[0]):
        mu_w_cond_rec, Sigma_w_cond_rec = old_promp.get_conditioned_weights(t_cond[i], q_cond[i], mean_w=mu_w_cond_rec, var_w=Sigma_w_cond_rec)

    cond_traj = np.zeros([2,100])
    sigma_con_array =np.zeros([2,2,100]) 
    for i in range(len(domain)):
        mu_marg_q_con, Sigma_marg_q_con = promp.get_marginal(domain[i], mu_w_cond_rec, Sigma_w_cond_rec)
        sigma_con_array[:,:,i] = Sigma_marg_q_con
        cond_traj[:,i] = mu_marg_q_con

    limit = np.array([[state[4],state[4]+1],[state[5],state[5]+1]])   
    obs_dis= traj_rect_dist(cond_traj.T,  limit)

    # mu_1= mean_margs.T.reshape(-1)
    mu_2= cond_traj.T.reshape(-1)
    # sigma_1 = np.zeros([200,200])
    sigma_2 = np.zeros([200,200])
    for i in range(100):
        # sigma_1[i*2:(i+1)*2,i*2:(i+1)*2] = sigma_s[:,:,i]
        sigma_2[i*2:(i+1)*2,i*2:(i+1)*2] = sigma_con_array[:,:,i]
    fid_cost= calculate_frechet_distance(mu_1,sigma_1,mu_2,sigma_2) *0.01

    alif =0.5
    reward =  alif* fid_cost  -obs_dis*(1-alif)

    return reward


def get_args():
    parser = argparse.ArgumentParser(description="hyper parameters")
    parser.add_argument('--algo_name', default='PPO', type=str, help="name of algorithm")
    # parser.add_argument('--env_name', default='CartPole-v1', type=str, help="name of environment")
    parser.add_argument('--train_eps', default=100, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--mini_batch_size', default=5, type=int, help='mini batch size')
    parser.add_argument('--n_epochs', default=4, type=int, help='update number')
    parser.add_argument('--actor_lr', default=0.0003, type=float, help="learning rate of actor net")
    parser.add_argument('--critic_lr', default=0.0003, type=float, help="learning rate of critic net")
    parser.add_argument('--gae_lambda', default=0.95, type=float, help='GAE lambda')
    parser.add_argument('--policy_clip', default=0.2, type=float, help='policy clip')
    parser.add_argument('-batch_size', default=20, type=int, help='batch size')
    parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dim')
    parser.add_argument('--device', default='cpu', type=str, help="cpu or cuda")
    args = parser.parse_args()
    return args

cfg = get_args()


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.show()
    plt.savefig(figure_file)


n_states=6
n_actions=3

agent = Agent(n_states, n_actions, cfg)

def env_reset():
    states = np.zeros(6)
    mu= np.array([2.57696126, 2.34334736, 6.38525582, 3.0674252])
    for i in range(4):
        states[i] = np.random.normal(loc=mu[i], scale=1)
    states[4]=np.random.uniform(0,7)
    states[5]=np.random.uniform(0,7)
    states[:4]=  np.clip(states[:4],  0, 8)
    return states


def train(cfg, agent):
    print('开始训练！')
    print(f' 算法：{cfg.algo_name}, 设备：{cfg.device}')
    figure_file = 'ppo_via.png'
    rewards = []
    for i_ep in range(cfg.train_eps):
        state = env_reset()
        ep_reward = 0
            
        action, prob,val = agent.choose_action(state)
        reward = get_reward(state,action)

        ep_reward += reward
        agent.memory.push(state, action, prob,val, reward)
        agent.learn()
        rewards.append(ep_reward)
        print(rewards)
        if (i_ep + 1) % 10 == 0:
            print(f"回合：{i_ep + 1}/{cfg.train_eps}，奖励：{ep_reward:.2f}")

    x = [i+1 for i in range(len(rewards))]
    plot_learning_curve(x, rewards, figure_file)
    

train(cfg, agent)
