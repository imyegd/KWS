import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import *
class CNN(nn.Module):
    def __init__(self, num_classes=12):  # 假设有10个类别
        super(CNN, self).__init__()
        # 第一个卷积层，输入通道为1，输出通道为32，卷积核大小为3x3，使用padding=1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 池化层，窗口大小为2x2，步长为2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个卷积层，输入通道为32，输出通道为64，卷积核大小为3x3，使用padding=1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 第三个卷积层，输入通道为64，输出通道为128，卷积核大小为3x3，使用padding=1
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # 全连接层，输入特征数量为128 * 25 * 5（假设经过两次池化后尺寸变为25x5），输出128
        self.fc1 = nn.Linear(128 * 24 * 3, 128)
        # 最后的全连接层，输出为类别数
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        # 通过第一个卷积层和池化层
        x = self.pool(F.relu(self.conv1(x)))
        # 通过第二个卷积层
        x = self.pool(F.relu(self.conv2(x)))
        # 通过第三个卷积层
        x = F.relu(self.conv3(x))
        # 展平层
        x = x.view(-1, 128 * 24 * 3)  # 根据实际输出尺寸调整
        # 通过第一个全连接层
        x = F.relu(self.fc1(x))
        # 通过第二个全连接层，输出层
        x = self.fc2(x)
        return x
    

class LSTMModel(nn.Module):
    def __init__(self, ):
        super(LSTMModel, self).__init__()

        with open('config.json', 'r') as config_file:
            config = json.load(config_file)

        self.hidden_layer_size = config_file["lstm_para"]["hidden_layer_size"]
        self.num_layers = config_file["lstm_para"]["num_layers"]
        input_size = config_file["lstm_para"]["input_size"]
        output_size = config_file["lstm_para"]["output_size"]
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        # 全连接层，用于输出
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)

        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出用于预测
        out = self.fc(out[:, -1, :])
        return out
    
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path):
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()  # 清空梯度
            # images = images.float()
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            running_loss += loss.item()

            # 转换输出和标签到相同的类型
            preds = outputs.argmax(dim=1)
            # 更新正确预测的数量
            train_correct += preds.eq(labels.argmax(dim=1)).sum().item()
            train_total += labels.size(0)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
        print(f'Train Acc: {train_correct/train_total}')

        # 验证模型
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # 在验证阶段不计算梯度
            for images, labels in val_loader:
                outputs = model(images)
                # 计算损失
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # 转换输出和标签到相同的类型
                preds = outputs.argmax(dim=1)
                
                # 更新正确预测的数量
                val_correct += preds.eq(labels.argmax(dim=1)).sum().item()
                val_total += labels.size(0)
        print(f'Validation Loss: {val_loss/len(val_loader)}')
        print(f'Validation Acc: {val_correct/val_total}')
        torch.save(model.state_dict(), os.path.join(save_path, f'{epoch:02}_model.pth'))
        with open(os.path.join(save_path, 'checkpoint.txt'), 'a', encoding='utf-8') as f:
            lines = []
            lines.append(f'Epoch {epoch+1}/{num_epochs}')
            lines.append(f'Train Loss: {running_loss/len(train_loader)}')
            lines.append(f'Train Acc: {train_correct/train_total}')
            lines.append(f'Validation Loss: {val_loss/len(val_loader)}')
            lines.append(f'Validation Acc: {val_correct/val_total}')
            lines.append('\n\n')
            f.write('\n'.join(lines))


def test_model(model, test_loader):
        
        model.eval()  # 设置模型为评估模式
        
        test_correct = 0
        test_total = 0
        with torch.no_grad():  # 在验证阶段不计算梯度
            for images, labels in test_loader:
                outputs = model(images)

                
                # 转换输出和标签到相同的类型
                preds = outputs.argmax(dim=1)
                
                # 更新正确预测的数量
                test_correct += preds.eq(labels.argmax(dim=1)).sum().item()
                test_total += labels.size(0)
        print(f'Test Acc: {test_correct/test_total}')


if __name__=="__main__":

    train_loader = getloader('train')

    val_loader = getloader('test')

    test_loader = getloader('val')


    # model = CNN()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # num_epochs = 20
    # save_path = 'model/CNN'
    # train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path)

    model = CNN()
    model.load_state_dict(torch.load('model\CNN\\19_model.pth'))
    test_model(model, test_loader)

