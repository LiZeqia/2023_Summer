import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from itertools import chain
import statistics
import itertools
import logging
import matplotlib.pyplot as plt


def load_imdb_data(data_dir):
    train_texts, train_labels = [], []
    for category in ['pos', 'neg']:
        train_dir = os.path.join(data_dir, 'train', category)
        for filename in os.listdir(train_dir):
            with open(os.path.join(train_dir, filename), 'r', encoding='utf-8') as file:
                text = file.read().strip()
                train_texts.append(text)
                train_labels.append(1 if category == 'pos' else 0)

    test_texts, test_labels = [], []
    for category in ['pos', 'neg']:
        test_dir = os.path.join(data_dir, 'test', category)
        for filename in os.listdir(test_dir):
            with open(os.path.join(test_dir, filename), 'r', encoding='utf-8') as file:
                text = file.read().strip()
                test_texts.append(text)
                test_labels.append(1 if category == 'pos' else 0)

    return train_texts, train_labels, test_texts, test_labels


def build_vocab(texts, max_vocab_size=None):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    if max_vocab_size is not None:
        counter = dict(counter.most_common(max_vocab_size))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for i, (word, freq) in enumerate(counter.items(), 2):
        vocab[word] = i
    return vocab, len(vocab)


def convert_texts_to_ids(texts, vocab):
    unk_id = vocab['<UNK>']
    return [[vocab.get(word, unk_id) for word in text.split()] for text in texts]


class IMDBDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label = self.data[index]
        text = torch.tensor(text)
        label = torch.tensor(label)
        return text, label
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_prob):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h_n = h_n[-1, :, :]
        out = self.fc(h_n)
        return out

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_prob):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.gru(x)
        h_n = h_n[-1, :, :]
        out = self.fc(h_n)
        return out

def same_len(text,size):
    text_1=[]
    for i in text:
        if(len(i)>size):
            i=i[:size]
        while(len(i)<size):
            i.append(0)
        text_1.append((i))
    return text_1

def train_model(num_epochs):
    # logging.basicConfig(filename='LSTM.log', level=logging.INFO)
    logging.basicConfig(filename='GRU.log', level=logging.INFO)
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        loss_train = 0.0
        loss_test = 0.0
        accuracy = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_train = loss_train + loss.item()
            if i % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 在测试集上评估模型
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss_test = loss_test + loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f'Test Accuracy: {accuracy:.2f}%')
        logging.info('Epoch %d: Train_Loss:%.3f Test_loss:%.3f Accuracy:%.3f', epoch + 1,
                     float(loss_train / len(train_loader)), float(loss_test / len(test_loader)), accuracy)
    # torch.save(model.state_dict(), 'LSTM.pth')
    torch.save(model.state_dict(), 'GRU.pth')

def draw_image():
    with open('LSTM.log', 'r') as f:
        lines = f.readlines()
        loss_train_with = [float(line.split(' ')[-3].split(":")[-1]) for line in lines]
        loss_test_with = [float(line.split(' ')[-2].split(":")[-1]) for line in lines]
        accu_with = [float(line.split(' ')[-1].split(":")[-1]) for line in lines]

    with open('GRU.log', 'r') as f:
        lines = f.readlines()
        loss_train_without = [float(line.split(' ')[-3].split(":")[-1]) for line in lines]
        loss_test_without = [float(line.split(' ')[-2].split(":")[-1]) for line in lines]
        accu_without = [float(line.split(' ')[-1].split(":")[-1]) for line in lines]

    x = [int(i) for i in range(0, 20)]

    plt.subplot(5, 1, 1)
    plt.plot(x, loss_train_with, color="r", label="LSTM")
    plt.plot(x, loss_train_without, color="b", label="GRU")
    plt.legend()
    plt.title("Train_loss")

    plt.subplot(5, 1, 3)
    plt.plot(x, loss_test_with, color="r", label="LSTM")
    plt.plot(x, loss_test_without, color="b", label="GRU")
    plt.legend()
    plt.title("Test_loss")

    plt.subplot(5, 1, 5)
    plt.plot(x, accu_with, color="r", label="LSTM")
    plt.plot(x, accu_without, color="b", label="GRU")
    plt.legend()
    plt.title("Accuracy")

    plt.show()

def model_test():
    #这里需要将batch_size设置为1

    model.load_state_dict(torch.load('GRU.pth'))
    # model.load_state_dict(torch.load('LSTM.pth'))

    model.eval()
    with torch.no_grad():
        correct = [0, 0, 0]
        total = [0, 0, 0]
        for inputs, labels in test_loader:
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            if (labels == 0):
                total[0] = total[0] + 1
            else:
                total[1] = total[1] + 1
            total[2] = total[2] + 1

            if (predicted == labels):
                correct[labels] = correct[labels] + 1
                correct[2] = correct[2] + 1
        print(float(correct[0] / total[0]))
        print(float(correct[1] / total[1]))
        print(float(correct[2] / total[2]))
if __name__ =="__main__":
    # 设置超参数
    max_vocab_size = 10000
    embedding_size = 128
    hidden_size = 64
    num_layers = 2
    num_classes = 2
    dropout_prob = 0.5
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20

    # 加载 IMDB 数据集并构建词汇表
    data_dir = './data/aclImdb'
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(data_dir)

    vocab, vocab_size = build_vocab(train_texts, max_vocab_size)

    train_ids = convert_texts_to_ids(train_texts, vocab)
    test_ids = convert_texts_to_ids(test_texts, vocab)


    text_len=[len(i) for i in train_ids]
    cut_len=int(statistics.mean(text_len))

    train_ids=same_len(train_ids,cut_len)
    test_ids=same_len(test_ids,cut_len)

    # 将 ID 列表转换为张量并分批
    train_data = [(ids, label) for ids,label in zip(train_ids, train_labels)]
    test_data = [(ids, label) for ids, label in zip(test_ids, test_labels)]
    train_dataset = IMDBDataset(train_data)
    test_dataset = IMDBDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型和优化器
    #model = LSTMClassifier(vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_prob)
    model = GRUClassifier(vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_prob)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #训练模型
    #train_model(num_epochs)
    #绘制训练过程图
    #draw_image()
    #测试模型
    model_test()
