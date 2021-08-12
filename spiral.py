import math
import tempfile
import imageio
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from os import path
from tqdm import tqdm


def spiral(n_points, upto=5, std=0.2, revert=False):
    angles = np.linspace(0, math.pi * 4, n_points) * 1j
    points = math.e ** angles
    points *= np.linspace(upto, 1, num=n_points)
    points.real += np.random.normal(0, std, size=n_points)
    points.imag += np.random.normal(0, std, size=n_points)
    return points if revert is False else points * -1


def prepare_train_data(class1, class2):
    train_x = np.empty((len(class1) + len(class2), 2))
    train_x[:, 0] = np.concatenate([class1.real, class2.real])
    train_x[:, 1] = np.concatenate([class1.imag, class2.imag])
    train_x = torch.Tensor(train_x)
    train_y = torch.cat(
            [torch.zeros(len(class1)), torch.ones(len(class2))]
            ).unsqueeze(1)
    return train_x, train_y


class MLP(nn.Module):
    def __init__(self, input_shape, depth=64):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
                nn.Linear(input_shape, depth),
                nn.ReLU(),
                nn.Linear(depth, depth),
                nn.ReLU(),
                nn.Linear(depth, depth),
                nn.ReLU(),
                nn.Linear(depth, 1),
                nn.Sigmoid()
                )
        self.mse = nn.MSELoss()
        self.optim = torch.optim.RMSprop(params=self.mlp.parameters(),
                                         lr=0.002)

    def forward(self, x):
        return self.mlp(x)

    def train(self, x, y):
        output = self.mlp(x)
        loss = self.mse(output, y)
        loss.backward()
        self.optim.step()
        return loss.item()


if __name__ == "__main__":
    n_points = 256
    epochs = 256

    mlp = MLP(input_shape=2)

    # create training data
    class1 = spiral(n_points, revert=False)
    class2 = spiral(n_points, revert=True)
    train_x, train_y = prepare_train_data(class1, class2)

    # create test data
    grid_points = np.linspace(train_x.min(), train_x.max(), n_points)
    xa, xb = np.meshgrid(grid_points, grid_points)
    test_x = torch.Tensor([
            [xa[i][j], xb[i][j]]
            for i in range(n_points)
            for j in range(n_points)
            ])

    images = []
    pbar = tqdm(total=epochs)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for e in range(epochs):
            # train epoch
            loss = mlp.train(train_x, train_y)

            # plot epoch result
            xy = mlp(test_x).detach().reshape((n_points, n_points))
            plt.contourf(xa, xb, xy)
            plt.scatter(class1.real, class1.imag,
                        linewidths=2, edgecolors='black')
            plt.scatter(class2.real, class2.imag,
                        linewidths=2, edgecolors='black')
            plt.gca().set_aspect('equal')
            plt.colorbar()

            # save plot
            file_path = path.join(tmp_dir, f"{e}.png")
            plt.savefig(file_path)
            plt.clf()
            images.append(imageio.imread(file_path))

            # log
            pbar.update(1)
            pbar.set_description_str(f"loss: {round(loss, 5)}")

        print("creating gif...")
        imageio.mimsave("./training.gif", images)
        print("done")
