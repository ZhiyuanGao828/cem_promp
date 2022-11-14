import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import Arc


def arc_patch(center, radius, theta1, theta2, ax=None, resolution=50, **kwargs):
    # make sure ax is not empty
    if ax is None:
        ax = plt.gca()
    # generate the points
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    points = np.vstack((radius*np.cos(theta) + center[0],
                        radius*np.sin(theta) + center[1]))
    # build the polygon and add it to the axes
    poly = patches.Polygon(points.T, closed=True, **kwargs)
    ax.add_patch(poly)
    return poly


class ToyObstacle:
    def __init__(self, center_x, center_y, width, height, safe_distance=1):
        self.center_x = center_x
        self.center_y = center_y
        self.half_width = width/2.
        self.half_height = height/2.
        self.safe_distance = safe_distance
        return

    def visualize(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()


        rect = patches.Rectangle(
            (self.center_x - self.half_width, self.center_y-self.half_height - self.safe_distance),
            height=self.half_height*2 + 2 * self.safe_distance,
            width=self.half_width*2,
            rotation_point='center',
            linewidth=1, edgecolor='blue', facecolor='blue',
            alpha=0.2,
        )
        ax.add_patch(rect)

        rect = patches.Rectangle(
            (self.center_x - self.half_width - self.safe_distance, self.center_y-self.half_height),
            height=self.half_height*2,
            width=self.half_width*2 + 2 * self.safe_distance,
            rotation_point='center',
            linewidth=1, edgecolor='blue', facecolor='blue',
            alpha=0.2,
        )
        ax.add_patch(rect)

        centers = [
            [self.center_x + self.half_width, self.center_y + self.half_height],
            [self.center_x - self.half_width, self.center_y + self.half_height],
            [self.center_x - self.half_width, self.center_y - self.half_height],
            [self.center_x + self.half_width, self.center_y - self.half_height],
        ]

        thetas = [
            [0, 90],
            [90, 180],
            [180, 270],
            [270, 360]
        ]
        for _center, _theta in zip(centers, thetas):
            _c_x, _c_y = _center
            _th1, _th2 = _theta
            ax.add_patch(
                # arc_patch(center=(_c_x, _c_y), radius=self.safe_distance,
                #           theta1=_th1, theta2=_th2,
                #           color='blue',
                #           linewidth=1, edgecolor='blue', facecolor='blue',
                #           alpha=0.2, fill=True,
                #           )
                patches.Wedge((_c_x, _c_y), self.safe_distance,
                            theta1=_th1, theta2=_th2,
                            color='blue',
                            linewidth=1, edgecolor ='blue', facecolor = 'blue',
                            alpha=0.2, fill=True,
                            )
            )

        return ax

    def eval_distance(self, x, y, outside_scaling=5, inside_scaling=100):
        # distance = np.zeros_like(x)

        # TODO, not right, check again

        dis_1 = np.abs(self.center_x + self.half_width - x)
        dis_2 = np.abs(self.center_x - self.half_width - x)
        dis_3 = np.abs(self.center_y + self.half_height - y)
        dis_4 = np.abs(self.center_y - self.half_height - y)

        distance = np.min([dis_1, dis_2, dis_3, dis_4], axis=0) * outside_scaling / self.safe_distance

        distance[self.inside_obstacle(x, y)] = -inside_scaling
        distance[self.outside_obstacle(x, y)] = outside_scaling
        return distance

    def inside_obstacle(self, x, y):
        cond1 = x > self.center_x - self.half_width
        cond2 = x < self.center_y + self.half_width
        cond3 = y > self.center_y - self.half_height
        cond4 = y < self.center_y + self.half_height
        cond = np.logical_and.reduce([cond1, cond2, cond3, cond4])
        return cond

    def outside_obstacle(self, x, y):
        cond1 = x > self.center_x - self.half_width - self.safe_distance
        cond2 = x < self.center_y + self.half_width + self.safe_distance
        cond3 = y > self.center_y - self.half_height - self.safe_distance
        cond4 = y < self.center_y + self.half_height + self.safe_distance
        cond = np.logical_and.reduce([cond1, cond2, cond3, cond4])
        return np.logical_not(cond)


if __name__ == '__main__':
    toy_obstacle = ToyObstacle(3, 4, 2, 3)
    ax = toy_obstacle.visualize(ax=None)
    ax.plot([1, 2], [3, 4])
    plt.show()

    print(
        toy_obstacle.eval_distance(
            np.array([3,4,5,6]),
            np.array([3,4,5,6])
        )
    )