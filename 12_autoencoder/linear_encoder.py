import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

if not os.path.exists('./12_autoencoder/mlp_img'):
    os.mkdir('./12_autoencoder/mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 1
batch_size = 64
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./data/mnist/', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def plot_gallery(images, h=28, w=28, n_row=4, n_col=8):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    print(images.ndim)
    if images.ndim == 2:
        plt.imshow(images, cmap=plt.cm.gray)
    else:
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            #plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        intermediate_x = self.encoder(x)
        x = self.decoder(intermediate_x)
        return intermediate_x,x


model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img)
        # ===================forward=====================
        imoutput, output = model(img)
        loss = criterion(output, img)

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data[0]))
    if (epoch+1) % 1 == 0:
        print(imoutput.size())

        pic = torch.squeeze(to_img(output.data))
        
        plot_gallery(imoutput.data.numpy())
        plt.show()


        #save_image(pic, './12_autoencoder/mlp_img/image_{}.png'.format(epoch))

#torch.save(model.state_dict(), './sim_autoencoder.pth')
