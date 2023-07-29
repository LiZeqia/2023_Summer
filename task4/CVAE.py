import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import logging
from PIL import Image

# 定义 CVAE 模型
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 14 * 14 + n_classes, 512)
        self.fc21 = nn.Linear(512, latent_dim)
        self.fc22 = nn.Linear(512, latent_dim)
        self.fc3 = nn.Linear(latent_dim + n_classes, 512)
        self.conv4 = nn.ConvTranspose2d(512, 64, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.ConvTranspose2d(32, 8, kernel_size=4, stride=1, padding=1)
        self.fc4 = nn.Linear(200, 784)

    def encode(self, x, c):
        x = x.view(-1, 1, 28, 28)
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h2 = h2.view(-1, 64 * 14 * 14)
        hc = torch.cat((h2, c), 1)
        h3 = F.relu(self.fc1(hc))
        return self.fc21(h3), self.fc22(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c):
        zc = torch.cat((z, c), 1)
        h3 = F.relu(self.fc3(zc))
        h4 = F.relu(self.conv4(h3.view(-1, 512, 1, 1)))
        h5 = F.relu(self.conv5(h4))
        h6 = F.relu(self.conv6(h5))
        return torch.sigmoid(self.fc4(h6.view(-1,200)))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# 定义损失函数
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 训练和测试函数
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = torch.eye(n_classes)[labels].to(device)  #128*10

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, labels)
        #print(recon_batch.size())

        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    loss_mean=train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, loss_mean))
    return loss_mean

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = torch.eye(n_classes)[labels].to(device)
            recon_batch, mu, logvar = model(data, labels)
            test_loss +=loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

def train_model(epochs):
    logging.basicConfig(filename='CVAE.log', level=logging.INFO)
    for epoch in range(1, epochs + 1):
        train_loss=train(epoch)
        test_loss=test(epoch)
        logging.info('Epoch %d: Train_Loss:%.3f Test_loss:%.3f ', epoch,float(train_loss), float(test_loss))
    torch.save(model.state_dict(), 'model.pth')

def draw_loss():

    with open('CVAE.log', 'r') as f:
        lines = f.readlines()
        loss_train_with = [float(line.split(' ')[-3].split(":")[-1]) for line in lines]
        loss_test_with = [float(line.split(' ')[-2].split(":")[-1]) for line in lines]


    x = [int(i) for i in range(0, 50)]

    plt.subplot(1, 1, 1)
    plt.plot(x, loss_train_with, color="r", label="train_loss")
    plt.plot(x, loss_test_with, color="b", label="test_loss")
    plt.legend()
    plt.title("loss")

    plt.show()


def generate_image():
    model.load_state_dict(torch.load('model.pth'))

    images = []

    for i in range(10):
        label = i
        # Convert the label to a one-hot vector
        y = torch.zeros(1, 10).to(device)
        y[0, label] = 1

        for j in range(0, 10):
            z = torch.randn(1, latent_dim).to(device)
            x_hat = model.decode(z, y)
            x_hat = x_hat.cpu().detach().numpy().reshape(28, 28)

            images.append(x_hat)

    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)  # 调整子图之间的间距

    for row in range(10):
        for col in range(10):
            ax = axes[row, col]
            ax.imshow(images[row * 10 + col], cmap='gray')
            ax.axis('off')

    plt.show()



if __name__=="__main__":
    # 设定超参数
    batch_size = 128
    epochs = 20
    seed = 1
    log_interval = 10
    latent_dim = 20
    n_classes = 10

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # 加载数据集
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,transform=transforms.ToTensor()),batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),batch_size=batch_size, shuffle=True, **kwargs)

    model = CVAE().to(device)
    #
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    #训练模型
    #train_model(epochs)
    #绘制训练过程中图像
    #draw_loss()
    #验证训练结果
    generate_image()


