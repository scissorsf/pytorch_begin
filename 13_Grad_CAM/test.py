
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn.functional as F
from skimage.transform import resize
import matplotlib.pyplot as plt

import random

batch_size = 16
learning_rate = 0.0001
num_epoch = 1


def one_sided_padding(x):
    rand1 = random.randrange(0, 15, 3)
    rand2 = random.randrange(0, 15, 3)

    zero = np.zeros(shape=[28, 28, 1])
    zero[rand1:rand1 + 12, rand2:rand2 + 12,
         :] = np.asarray(x).reshape(12, 12, 1)
    return zero


mnist_train = dset.MNIST("./data/mnist", train=True,
                         transform=transforms.Compose([
                             transforms.RandomCrop(22),
                             transforms.Resize(12),
                             transforms.Lambda(one_sided_padding),
                             transforms.ToTensor(),
                         ]),
                         target_transform=None,
                         download=False)

mnist_test = dset.MNIST("./data/mnist/", train=False,
                        transform=transforms.Compose([
                            transforms.RandomCrop(22),
                            transforms.Resize(12),
                            transforms.Lambda(one_sided_padding),
                            transforms.ToTensor(),
                        ]),
                        target_transform=None,
                        download=False)

train_loader = torch.utils.data.DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
test_loader = torch.utils.data.DataLoader(
    mnist_test, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


class CNN(nn.Module):
    def __init__(self, num_feature=32):
        super(CNN, self).__init__()
        self.num_feature = num_feature

        self.layer = nn.Sequential(
            nn.Conv2d(1, self.num_feature, 3, 1, 1),
            nn.BatchNorm2d(self.num_feature),
            nn.ReLU(),
            nn.Conv2d(self.num_feature, self.num_feature * 2, 3, 1, 1),
            nn.BatchNorm2d(self.num_feature * 2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(self.num_feature * 2, self.num_feature * 4, 3, 1, 1),
            nn.BatchNorm2d(self.num_feature * 4),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            # nn.Conv2d(self.num_feature * 4, self.num_feature * 8, 3, 1, 1),
            # nn.BatchNorm2d(self.num_feature * 8),
            # nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.num_feature *4 * 7 * 7, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):

                # Kaming Initialization
                init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.Linear):

                # Kaming Initialization
                init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        out = self.layer(x)
        out = out.view(x.size()[0], -1)
        out = self.fc_layer(out)

        return out

def forward_hook(*args):
   
    print("\n This is Forward Hook. \n")
    for idx, ctx in enumerate(args):
        print("\n 第{}个参数\n".format(idx))
        if isinstance(ctx, Variable):
            print(ctx.size())
        elif isinstance(ctx, tuple):
            for jxx, c in enumerate(ctx):
                print("\n \t 第{}个参数的第{}个参数 \n".format(idx,jxx))
                print(c.size())
        else:
            print(ctx)


def backward_hook(*args):

    print("\n This is Backward Hook. \n")
    for idx, ctx in enumerate(args):
        print("\n 第{}个参数\n".format(idx))
        if isinstance(ctx, Variable):
            print(ctx.size())
        elif isinstance(ctx, tuple):
            for jxx, c in enumerate(ctx):
                print("\n \t 第{}个参数的第{}个参数 \n".format(idx, jxx))
                print(c.size())
        else:
            print(ctx)


def main():

    model = CNN().cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    image,label = next(iter(train_loader))
    y_ = Variable(label).cuda()
    #h1 = model._modules['layer'][6].register_forward_hook(forward_hook)
    h2 = model._modules['layer'][3].register_backward_hook(backward_hook)

    # print(model._modules['layer'][6])

    o_ = model(Variable(image, requires_grad=True).cuda())
    loss = loss_func(o_,y_)
    loss.backward()

    h2.remove()
    # for i in range(num_epoch):
    #     model.train()
    #     for j, [image, label] in enumerate(train_loader):
    #         x = Variable(image).cuda()
    #         y_ = Variable(label).cuda()

    #         optimizer.zero_grad()
    #         output = model.forward(x)
    #         loss = loss_func(output, y_)
    #         loss.backward()
    #         optimizer.step()

    #     top_1_count = torch.FloatTensor([0])
    #     total = torch.FloatTensor([0])
    #     model.eval()
    #     for image, label in test_loader:
    #         x = Variable(image, volatile=True).cuda()
    #         y_ = Variable(label).cuda()

    #         output = model.forward(x)

    #         values, idx = output.max(dim=1)
    #         top_1_count += torch.sum(y_ == idx).float().cpu().data

    #         total += label.size(0)

    #     print("Test Data Accuracy: {}%".format(
    #         100 * (top_1_count / total).numpy()))
    #     if (top_1_count / total).numpy() > 0.98:
    #         break


if __name__ == '__main__':
    main()
