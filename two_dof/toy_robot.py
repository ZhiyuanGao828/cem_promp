import numpy as np
import scipy
import scipy.spatial.transform


class ToyRobot:
    def __init__(self, l1, l2, base_x, base_y):
        self.l1 = l1
        self.l2 = l2
        self.base_x = base_x
        self.base_y = base_y
        return

    def fk(self, theta_1, theta_2):
        x_1, y_1 = self.l1 * np.cos(theta_1), self.l1 * np.sin(theta_1)
        x_2, y_2 = self.l2 * np.cos(theta_1 + theta_2), self.l2 * np.sin(theta_1 + theta_2)

        return x_1 + x_2 + self.base_x, y_1 + y_2 + self.base_y

    def ik(self, x_in, y_in):
        x, y = x_in - self.base_x, y_in - self.base_y
        cos_aux = np.square(x) + np.square(y) + np.square(self.l1) - np.square(self.l2)
        cos_aux /= 2 * self.l1 * np.sqrt(np.square(x) + np.square(y))
        aux = np.arccos(cos_aux)
        th = np.arctan2(y, x)
        theta_1 = th - aux

        cos_aux_2 = - np.square(self.l1) - np.square(self.l2) + np.square(x) + np.square(y)
        cos_aux_2 /= 2 * self.l1 * self.l2
        theta_2 = np.arccos(cos_aux_2)
        return theta_1, theta_2

    def if_solvable(self, x, y):
        cond1 = np.square(x) + np.square(y) < np.square(self.l1 + self.l2)
        cond2 = np.square(x) + np.square(y) > np.square(self.l1 - self.l2)
        return np.all(cond1) and np.all(cond2)


if __name__ == '__main__':
    toy_robo = ToyRobot(6., 8., 0., 0.)
    th1, th2 = 0.1, 0.2
    print()
    print(th1, th2)
    x, y = toy_robo.fk(th1, th2)
    print(x, y)
    print(toy_robo.ik(x, y))
    print(toy_robo.fk(
        *toy_robo.ik(x, y)
    ))
    