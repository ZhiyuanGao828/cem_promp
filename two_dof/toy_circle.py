import numpy as np

import intprim
from intprim.probabilistic_movement_primitives import *
import numpy as np
import matplotlib.pyplot as plt
from intprim.util.kinematics import BaseKinematicsClass
import sys

import matplotlib.pyplot as plt
import numpy as np

# Import the library.
import intprim
import toy_robot
import toy_obstacle

# Set a seed for reproducibility
np.random.seed(213413414)

# Define some parameters used when generating synthetic data.
num_train_trajectories = 30
train_translation_mean = 0.0
train_translation_std = 0.2
train_noise_std = 0.35
train_length_mean = 95
train_length_std = 30

# Generate some synthetic handwriting trajectories.
training_trajectories = intprim.examples.create_2d_handwriting_data(
    num_train_trajectories,
    train_translation_mean,
    train_translation_std,
    train_noise_std,
    train_length_mean,
    train_length_std)

toy_obstacle = toy_obstacle.ToyObstacle(
    center_x=5.5, center_y=6,
    width=2, height=3, safe_distance=1
)

# Plot the results.
fig, ax = plt.subplots()
for trajectory in training_trajectories:
    ax.plot(trajectory[0], trajectory[1], alpha=0.5)

toy_obstacle.visualize(ax)
ax.set_aspect(1)
ax.set_title('demonstration and obstacle')
plt.show()


toy_robo = toy_robot.ToyRobot(6., 8., 0., 0.)
joint_trajectories = []
for trajectory in training_trajectories:
    assert toy_robo.if_solvable(trajectory[0], trajectory[1])
    _j1, _j2 = toy_robo.ik(trajectory[0], trajectory[1])
    joint_trajectories.append(np.vstack([_j1, _j2]))


# promp part
basis_model = intprim.basis.GaussianModel(8, 0.1, ['joint'+str(i) for i in range(2)])
promp = ProMP(basis_model)

# Add Demonstrations to the ProMP, which in turn calculates the list of weights for each demonstration.
for i in range(len(joint_trajectories)):
    promp.add_demonstration(joint_trajectories[i])

# Plot samples from the learnt ProMP

n_samples = 20 # Number of trajectoies to sample
plot_dof = 0 # Degree of freedom to plot
domain = np.linspace(0,1,100)

fig, axes = plt.subplots(3, 1)
fig_eval, axes_eval = plt.subplots()

axes_eval.set_title('sampled from proMP')
toy_obstacle.visualize(axes_eval)

for i in range(n_samples):
    for plot_dof in range(2):
        axes[plot_dof].set_title(f'DOF {plot_dof}')

        samples, _ = promp.generate_probable_trajectory(domain)
        axes[plot_dof].plot(domain, samples[plot_dof,:], 'g--', alpha=0.3)
    _th1, _th2 = samples[0], samples[1]
    _x, _y = toy_robo.fk(_th1, _th2)
    axes_eval.plot(_x, _y, 'g--', alpha=0.3)

# mean
_, _var = promp.get_basis_weight_parameters()
samples, _ = promp.generate_probable_trajectory(domain, var=np.zeros_like(_var))
_th1, _th2 = samples[0], samples[1]
_x, _y = toy_robo.fk(_th1, _th2)
axes_eval.plot(_x, _y, )

for plot_dof in range(2):
    mean_margs = np.zeros(samples[plot_dof, :].shape)
    upper_bound = np.zeros(samples[plot_dof, :].shape)
    lower_bound = np.zeros(samples[plot_dof, :].shape)
    for i in range(len(domain)):
        mu_marg_q, Sigma_marg_q = promp.get_marginal(domain[i])
        std_q = Sigma_marg_q[plot_dof][plot_dof] ** 0.5

        mean_margs[i] = mu_marg_q[plot_dof]
        upper_bound[i] = mu_marg_q[plot_dof] + std_q
        lower_bound[i] = mu_marg_q[plot_dof] - std_q

    axes[plot_dof].fill_between(domain, upper_bound, lower_bound, color='g', alpha=0.2)
    axes[plot_dof].plot(domain, mean_margs, 'g-')
    axes[plot_dof].set_title('Samples for DoF {}'.format(plot_dof))

fig.tight_layout()
plt.show()


target_task_space = [6.5, 3.]
q_cond_init = toy_robo.ik(target_task_space[0], target_task_space[1])
mu_w_cond, Sigma_w_cond = promp.get_conditioned_weights(1., q_cond_init)

fig, axes = plt.subplots()
for i in range(n_samples):
    samples, _ = promp.generate_probable_trajectory(domain)
    # for plot_dof in range(2):
    #     axes[plot_dof].set_title(f'DOF {plot_dof}')
    #
    #
    #     axes[plot_dof].plot(domain, samples[plot_dof,:], 'g--', alpha=0.3)
    _th1, _th2 = samples[0], samples[1]
    _x, _y = toy_robo.fk(_th1, _th2)
    axes.plot(_x, _y, 'g--', alpha=0.3)

# mean
_, _var = promp.get_basis_weight_parameters()
samples, weights_conditioned = promp.generate_probable_trajectory(domain, var=np.zeros_like(_var))
_th1, _th2 = samples[0], samples[1]
_x, _y = toy_robo.fk(_th1, _th2)
axes.plot(_x, _y, )
toy_obstacle.visualize(axes)
axes.scatter(target_task_space[0], target_task_space[1])
axes.set_title('conditioned proMP')
plt.show()

for i in range(10):
    reward_list = []
    weights_list = []
    for i in range(100):
        samples, weights = promp.generate_probable_trajectory(domain, mean=weights_conditioned, var=_var)
        _x, _y = toy_robo.fk(samples[0], samples[1])
        reward = toy_obstacle.eval_distance(_x, _y).mean()
        reward_list.append(reward)
        weights_list.append(weights)
    print(np.mean(reward_list), "+-", np.std(reward_list), np.min(reward_list), np.max(reward_list))

    ind = np.argsort(reward_list)
    ind = ind[70:]
    weights_list= np.vstack(weights_list)
    weights_list = weights_list[ind]
    weights_list = np.mean(weights_list, axis=0)
    weights_conditioned = weights_list


fig, axes = plt.subplots()
for i in range(n_samples):
    samples, _ = promp.generate_probable_trajectory(domain, mean=weights_conditioned)
    # for plot_dof in range(2):
    #     axes[plot_dof].set_title(f'DOF {plot_dof}')
    #
    #
    #     axes[plot_dof].plot(domain, samples[plot_dof,:], 'g--', alpha=0.3)
    _th1, _th2 = samples[0], samples[1]
    _x, _y = toy_robo.fk(_th1, _th2)
    axes.plot(_x, _y, 'g--', alpha=0.3)

# mean
_, _var = promp.get_basis_weight_parameters()
samples, weights_conditioned = promp.generate_probable_trajectory(domain, mean=weights_conditioned,
                                                                  var=np.zeros_like(_var))
_th1, _th2 = samples[0], samples[1]
_x, _y = toy_robo.fk(_th1, _th2)
axes.plot(_x, _y, )
toy_obstacle.visualize(axes)
axes.scatter(target_task_space[0], target_task_space[1])
axes.set_title('optimized proMP')
plt.show()
print('finished')
