import numpy as np
import pygame
from matplotlib import pyplot as plt
from sklearn.datasets import *
from sklearn.metrics import accuracy_score
from nn import NeuralNet

PI = np.pi

class RoboticArm:
    def __init__(self):
        self.joints_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.link_size = [2, 1, 0.5]
        self.joints_angles = [0, 0, 0]
        self.forward_kinematic()

    def forward_kinematic(self):
        last_angle = 0
        self.joints_pos[1] = [np.cos(self.joints_angles[0]) * self.link_size[0], np.sin(self.joints_angles[0]) * self.link_size[0]]
        for i in range(1,  len(self.joints_pos)):
            last_pos = self.joints_pos[i - 1]
            angle = self.joints_angles[i - 1] + last_angle
            self.joints_pos[i] = [last_pos[0] + np.cos(angle) * self.link_size[i - 1], last_pos[1] + np.sin(angle) * self.link_size[i - 1]]
            last_angle = angle

    def set_angles(self, index, value):
        self.joints_angles[index] = value
        self.forward_kinematic()


def main():
    X, y = make_moons(n_samples=100, noise=0.1, random_state=21)
    y = y.reshape((y.shape[0], 1))  # from a big array to a multiples little arrays

    nn = NeuralNet([2, 6, 6, 1])
    nn.setup_training(X, y)
    nn.iteration_training(10)

    arm = RoboticArm()

    width = 800
    height = 800

    pygame.init()

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("NN experimentation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)


    # robotic arm
    values = [[5 for _ in range(80)] for _
            in range(80)]
    angles = [0, 0, 0]

    for _ in range(100000):
        dt = 0.005
        for i in range(3):
            angles[i] += (5 + 4.3 * i + np.random.uniform(0, 1)) * dt
            angles[i] %= 2 * PI

        arm.set_angles(0, angles[0])
        arm.set_angles(1, angles[1])
        arm.set_angles(2, angles[2])
        arm.forward_kinematic()

        new_value = min(angles[0], abs(2 * PI - angles[0])) + min(angles[1], abs(2 * PI - angles[1])) + min(angles[2],
                                                                                                            abs(2 * PI -
                                                                                                                angles[
                                                                                                                    2]))
        end_pos = arm.joints_pos[-1]
        grid_x = int((end_pos[0] + 4) * 10)
        grid_y = int((end_pos[1] + 4) * 10)

        if grid_x >= 0 and grid_x < 80 and grid_y >= 0 and grid_y < 80:
            if new_value < values[grid_y][grid_x]:
                values[grid_y][grid_x] = new_value

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # update
        dt = clock.tick(60) / 1000.0  # Convert milliseconds to seconds
        for i in range(3):
            angles[i] += (5 + 4.3 * i + np.random.uniform(0, 3)) * dt
            angles[i] %= 2 * PI

        arm.set_angles(0, angles[0])
        arm.set_angles(1, angles[1])
        arm.set_angles(2, angles[2])
        arm.forward_kinematic()

        new_value = min(angles[0], abs(2*PI - angles[0])) + min(angles[1], abs(2*PI - angles[1])) + min(angles[2], abs(2*PI - angles[2]))
        end_pos = arm.joints_pos[-1]
        grid_x = int((end_pos[0] + 4) * 10)
        grid_y = int((end_pos[1] + 4) * 10)

        if grid_x >= 0 and grid_x < 80 and grid_y >= 0 and grid_y < 80:
            if new_value < values[grid_y][grid_x]:
                values[grid_y][grid_x] = new_value

        screen.fill((100, 100, 100))

        for y in range(80):
            for x in range(80):
                col = (5 - values[y][x]) / 5  * 255
                pygame.draw.rect(screen, (col, col, col), (x * 10, y * 10, 10, 10))

        for i in range(len(arm.joints_pos) - 1):
            spos1 = (int(arm.joints_pos[i][0] * 100 + width / 2), int(arm.joints_pos[i][1] * 100 + height / 2))
            spos2 = (int(arm.joints_pos[i + 1][0] * 100 + width / 2), int(arm.joints_pos[i + 1][1] * 100 + height / 2))

            pygame.draw.line(screen, (255, 255, 255), spos1, spos2, 5)

        for i in range(len(arm.joints_pos)):
            pos = (int(arm.joints_pos[i][0] * 100 + width / 2), int(arm.joints_pos[i][1] * 100 + height / 2))
            pygame.draw.circle(screen, (255, 0, 0), (pos[0], pos[1]), 10)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()
