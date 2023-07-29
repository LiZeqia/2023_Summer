
import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import gradio as gr
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import logging
import glob

#建立DNN模型
class DNN_Net(nn.Module):
    def __init__(self):
        super(DNN_Net, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

#建立DNN_8模型 模型有八层
class DNN_8_Net(nn.Module):
    def __init__(self):
        super(DNN_8_Net, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)
        return x

#建立CNN模型
class CNN_Net_10(nn.Module):
    def __init__(self):
        super(CNN_Net_10, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        #print(x.size())
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        #print(x.size())

        x=self.relu(self.bn3(self.conv3(x)))
        x=self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        #print(x.size())

        x=self.relu(self.bn5(self.conv5(x)))
        x=self.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        #print(x.size())

        x=self.relu(self.bn7(self.conv7(x)))
        x=self.relu(self.bn8(self.conv8(x)))
        x = self.pool(x)
        #print(x.size())

        x = x.view(-1, 512 * 2 * 2)
        #print(x.size())
        x = self.fc1(x)
        #print(x.size())
        x = self.relu(x)
        #print(x.size())
        x = self.fc2(x)
        #print(x.size())
        x = self.relu(x)
        #print(x.size())
        x = self.fc3(x)
        #print(x.size())
        return x


def train():
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    meal_loss=running_loss/len(train_loader)
    return meal_loss

def test():
    model.eval()
    correct = 0
    total = 0
    run_loss=0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            run_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100*correct/total,run_loss/len(train_loader)

def train_model():
    logging.basicConfig(filename='bs_512.log', level=logging.INFO)
    for i in range(0,20):
        train_loss=train()
        accu,test_loss=test()
        print('Epoch {},  Train_Loss:{:.3f},   Test_loss:{:.3f},   Accuracy:{:.3f}'.format( i + 1,float(train_loss), float(test_loss), accu))
        logging.info('Epoch %d,  Train_Loss:%.3f,   Test_loss:%.3f,   Accuracy:%.3f', i + 1,float(train_loss), float(test_loss), accu)
    #torch.save(model.state_dict(), 'CNN.pth')

def draw_loss():
    with open('./train_loss/Train_DNN.log', 'r') as f:
        lines = f.readlines()
        loss_train_DNN = [float(line.split(',')[-3].split(":")[-1]) for line in lines]
        loss_test_DNN = [float(line.split(',')[-2].split(":")[-1]) for line in lines]
        accu_DNN = [float(line.split(',')[-1].split(":")[-1]) for line in lines]

    with open('./train_loss/Train_DNN_8.log', 'r') as f:
        lines = f.readlines()
        loss_train_DNN_8 = [float(line.split(',')[-3].split(":")[-1]) for line in lines]
        loss_test_DNN_8 = [float(line.split(',')[-2].split(":")[-1]) for line in lines]
        accu_DNN_8 = [float(line.split(',')[-1].split(":")[-1]) for line in lines]
    with open('./train_loss/Train_CNN.log', 'r') as f:
        lines = f.readlines()
        loss_train_CNN = [float(line.split(',')[-3].split(":")[-1]) for line in lines]
        loss_test_CNN = [float(line.split(',')[-2].split(":")[-1]) for line in lines]
        accu_CNN = [float(line.split(',')[-1].split(":")[-1]) for line in lines]
    x = [int(i) for i in range(0, 100)]

    plt.subplot(5, 1, 1)
    plt.plot(x, loss_train_DNN, color="r", label="DNN")
    plt.plot(x, loss_train_DNN_8, color="b", label="DNN_8")
    plt.plot(x,loss_train_CNN,color="g",label="CNN")
    plt.legend()
    plt.title("Train_loss")

    plt.subplot(5, 1, 3)
    plt.plot(x, loss_test_DNN, color="r", label="DNN")
    plt.plot(x, loss_test_DNN_8, color="b", label="DNN_8")
    plt.plot(x, loss_test_CNN, color="g", label="CNN")
    plt.legend()
    plt.title("Test_loss")

    plt.subplot(5, 1, 5)
    plt.plot(x, accu_DNN, color="r", label="DNN")
    plt.plot(x, accu_DNN_8, color="b", label="DNN_8")
    plt.plot(x, accu_CNN, color="g", label="CNN")
    plt.legend()
    plt.title("Accuracy")

    plt.show()


def draw_test(path):
    file_all=glob.glob(os.path.join(path,"*.log"))

    x = [int(i) for i in range(0, 20)]
    loss=[]
    accu=[]
    name=[]
    for file_path in file_all:
        file_name = os.path.basename(file_path)[0:-4]
        with open(file_path, 'r') as f:
            lines = f.readlines()
            loss_tr = [float(line.split(',')[-3].split(":")[-1]) for line in lines]
            accu_tr = [float(line.split(',')[-1].split(":")[-1]) for line in lines]
            loss.append(loss_tr)
            accu.append(accu_tr)
            name.append(file_name)

    plt.subplot(2, 1, 1)
    for i in range(0,len(loss)):
        plt.plot(x, loss[i], label=name[i])
    plt.legend()
    plt.title("Train_loss")

    plt.subplot(2, 1, 2)
    for i in range(0, len(accu)):
        plt.plot(x, accu[i], label=name[i])
    plt.legend()
    plt.title("Accuracy")

    plt.show()

# 定义预测函数，用于预测图片分类结果
def predict(choose):
    cifar10_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    img_now, label_now = test_dataset[int(choose)]

    model.eval()
    image = img_now.unsqueeze(0).to(device)  # 添加一个维度以匹模型的输入形状
    with torch.no_grad():
        output = model(image)  # 使用模型进行预测img_now=img_now.numpy()
    predicted_class = torch.argmax(output, dim=1).item()  # 获取预测结果的类索引

    img_now=img_now.numpy()
    img_now = np.transpose(img_now, (1, 2, 0))

    print(type(cifar10_labels[label_now]))
    print(type(cifar10_labels[predicted_class]))
    print(type(img_now))
    return cifar10_labels[label_now], cifar10_labels[predicted_class],img_now


def gradio_show():
    model.load_state_dict(torch.load('CNN.pth'))

    input_text = gr.inputs.Textbox()
    output_text_1 = gr.outputs.Textbox()
    output_text_2 = gr.outputs.Textbox()
    output_image = gr.outputs.Image(type='pil')
    gr.Interface(fn=predict, inputs=input_text, outputs=[output_text_1, output_text_2, output_image]).launch()


if __name__=="__main__":
    # 加载 CIFAR-10 数据集
    data_dir = './data'
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #选择模型
    #model=DNN_Net().to(device)
    #model=DNN_8_Net().to(device)
    model = CNN_Net_10().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    #训练模型
    train_model()

    #绘图--损失值
    #draw_loss()

    #绘图--调试参数
    #draw_test("./lr")

    #gradio可视化
    #gradio_show()
