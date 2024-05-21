import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import *

class ResidualBlock(nn.Module):  # 定义ResidualBlock类
    """实现子modual：residualblock"""

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):  # 初始化，自动执行
        super(ResidualBlock, self).__init__()  # 继承nn.Module
        self.left = nn.Sequential(  # 左网络，构建Sequential，属于特殊的module，类似于forward前向传播函数，同样的方式调用执行
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut  # 右网络，也属于Sequential，充当残差和非残差的判断标志。

    def forward(self, x):  # ResidualBlock的前向传播函数
        out = self.left(x)  # # 和调用forward一样如此调用left这个Sequential
        if self.right is None:  # 残差（ResidualBlock）
            residual = x  #
        else:  # 非残差（非ResidualBlock）
            residual = self.right(x)  #
        out += residual  # 结果相加
        return F.relu(out)  # 返回激活函数执行后的结果作为下个单元的输入


class Resnet(nn.Module):
    def __init__(self, numclasses):  # 创建实例时直接初始化 （3）
        super(Resnet, self).__init__()  # 表示ResNet继承nn.Module （4）
        self.pre = nn.Sequential(  # 构建Sequential，属于特殊的module，类似于forward前向传播函数，同样的方式调用执行 （5）（26）
            nn.Conv2d(1, 32, 3, padding=1),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer1 = self.make_layer(64, 128, 1)
        # 输入通道数为64，输出为128，根据残差网络结构将一个非Residual Block加上多个Residual Block构造成一层layer（6）
        self.layer2 = self.make_layer(128, 128, 1)
        # 输入通道数为128，输出为256
        self.layer3 = self.make_layer(128, 256, 1, stride=2)
        # 输入通道数为256，输出为256
        self.layer4 = self.make_layer(256, 256, 1)
        # 输入通道数为256，输出为512
        self.layer5 = self.make_layer(256, 512, 1, stride=2)
        # 输入通道数为256，输出为512
        self.layer6 = self.make_layer(512, 512, 1, stride=2)
        # 输入通道数为256，输出为512
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(3584, 2048)
        self.fc2 = nn.Linear(2048, numclasses)
        

    def make_layer(self, inchannel, outchannel, block_num, stride=1):
        # 创建layer层，（block_num-1）表示此层中Residual Block的个数 （7）
        """构建layer，包含多个residualblock"""
        shortcut = nn.Sequential(  # 构建Sequential，属于特殊的module，类似于forward前向传播函数，同样的方式调用执行 （8）
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []  # 创建一个列表，将非Residual Block和多个Residual Block装进去 （9）
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))  # 非残差也就是非Residual Block创建及入列表 （10）

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))  # 残差也就是Residual Block创建及入列表 （16）

        return nn.Sequential(
            *layers)

    def forward(self, x):  # ResNet类的前向传播函数 （24）
        x = x.unsqueeze(1)
        x = self.pre(x)  # 和调用forward一样如此调用pre这个Sequential（25）
        # print(x.shape)
        x = self.layer1(x)  # 和调用forward一样如此调用layer1这个Sequential
        # print(x.shape)
        x = self.layer2(x)  # 和调用forward一样如此调用layer2这个Sequential
        # print(x.shape)
        x = self.layer3(x)  # 和调用forward一样如此调用layer3这个Sequential
        # print(x.shape)
        x = self.layer4(x)  # 和调用forward一样如此调用layer4这个Sequential
        # print(x.shape)
        x = self.layer5(x)  # 和调用forward一样如此调用layer5这个Sequential
        # print(x.shape)
        x = self.layer6(x)  # 和调用forward一样如此调用layer6这个Sequential
        # print(x.shape)
        x = x.view(x.size(0), -1)  # 设置返回结果的尺度 （43）
        x = self.flat(x)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return x
    


class CNN(nn.Module):
    def __init__(self):  # 假设有10个类别
        super(CNN, self).__init__()
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)

            input_dim = config["cnn_para"]["input_dim"]
            hidden_dim = config["cnn_para"]["hidden_dim"]
            num_classes = config["cnn_para"]["output_size"]
            self.hidden_dim = config["cnn_para"]["hidden_dim"]

        # 第一个卷积层，输入通道为1，输出通道为32，卷积核大小为3x3，使用padding=1
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        # 池化层，窗口大小为2x2，步长为2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个卷积层，输入通道为32，输出通道为64，卷积核大小为3x3，使用padding=1
        self.conv2 = nn.Conv2d(hidden_dim, 2 * hidden_dim, kernel_size=3, stride=1, padding=1)
        # 第三个卷积层，输入通道为64，输出通道为128，卷积核大小为3x3，使用padding=1
        self.conv3 = nn.Conv2d(2 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1)
        # 全连接层，输入特征数量为128 * 24 * 3（假设经过两次池化后尺寸变为25x5），输出128
        self.fc1 = nn.Linear(4 * hidden_dim * 24 * 3, 4 * hidden_dim)
        # 最后的全连接层，输出为类别数
        self.fc2 = nn.Linear(4 * hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        # 通过第一个卷积层和池化层
        x = self.pool(F.relu(self.conv1(x)))
        # 通过第二个卷积层
        x = self.pool(F.relu(self.conv2(x)))
        # 通过第三个卷积层
        x = F.relu(self.conv3(x))
        # 展平层
        x = x.view(-1, 4 * self.hidden_dim * 24 * 3)  # 根据实际输出尺寸调整
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

            self.hidden_layer_size = config["lstm_para"]["hidden_layer_size"]
            self.num_layers = config["lstm_para"]["num_layers"]
            input_size = config["lstm_para"]["input_size"]
            output_size = config["lstm_para"]["output_size"]
        # LSTM层
        self.lstm = nn.LSTM(input_size, self.hidden_layer_size, self.num_layers, batch_first=True)
        # 全连接层，用于输出
        self.fc = nn.Linear(self.hidden_layer_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)

        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出用于预测
        out = self.fc(out[:, -1, :])
        return out
    

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()

        with open('config.json', 'r') as config_file:
            config = json.load(config_file)

            self.input_dim = config["cnnlstm"]["input_dim"]
            self.hidden_dim = config["cnnlstm"]["hidden_dim"]
            self.output_dim = config["cnnlstm"]["output_dim"]
            self.hidden_layer_size = config["cnnlstm"]["hidden_layer_size"]
            self.num_layers = config["cnnlstm"]["num_layers"]
            self.lstm_input = config["cnnlstm"]["lstm_last_dim"]
        
        self.conv1 = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)  # 卷积层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个卷积层，输入通道为32，输出通道为64，卷积核大小为3x3，使用padding=1
        self.conv2 = nn.Conv2d(self.hidden_dim, 1, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(self.lstm_input, self.lstm_input)
        self.lstm = nn.LSTM(self.lstm_input, self.hidden_layer_size, self.num_layers, batch_first=True)
        self.fc2 = nn.Linear(self.hidden_layer_size, self.output_dim)  # 全连接层
 
    def forward(self, x):
        x = x.unsqueeze(1) # [16, 1, 99, 13]
        x = self.pool(self.conv1(x))  # 卷积层激活函数[16, 16, 49, 6]
        x = F.relu(self.conv2(x))  # 池化层[16, 1, 49, 6]
        x = x.squeeze(dim=1)    # [16, 49, 6]
        x = self.fc1(x)
        # x = x.permute(0, 2, 1)  # 调整维度顺序以适应LSTM输入
        # x = x.view(x.shape[0], x.shape[1], -1)  # [16, 49, 6]

        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        # 前向传播LSTM
        x, _ = self.lstm(x, (h0, c0))   # [16, 49, 200]

        # 取最后一个时间步的输出用于预测
        x = self.fc2(x[:, -1, :])# [16, 12]
        # _, (h_n, _) = self.lstm(x)  # LSTM层
        # x = h_n.squeeze(0)  # 去除LSTM输出的第一维（batch_size=1）
        # x = F.relu(x)  # LSTM层激活函数
        # x = self.fc(x)  # 全连接层
        return x

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)

            self.input_dim = config["transformer"]["input_dim"]
            self.num_heads = config["transformer"]["num_heads"]
            self.num_encoder_layers = config["transformer"]["num_encoder_layers"]
            self.num_classes = config["transformer"]["num_classes"]

        
        # Transformer编码器层
        encoder_layer = TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=self.num_heads,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, self.num_encoder_layers)
        
        # 分类头
        self.classifier = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, x):
        # 调整输入数据形状以匹配Transformer期望的形状 [sequence_length, batch_size, feature_size]
        # x = x.permute(1, 0, 2)  # 从 [batch_size, sequence_length, feature_size] 到 [sequence_length, batch_size, feature_size]
        
        # 通过Transformer编码器
        x = self.transformer_encoder(x)
        
        # 取序列中最后一个时间步的特征用于分类
        x = x[:, -1, :]
        
        # 通过分类头
        out = self.classifier(x)
        
        return out

# class CNNLSTM(nn.Module):
#     def __init__(self):
#         super(CNNLSTM, self).__init__()

#         with open('config.json', 'r') as config_file:
#             config = json.load(config_file)

#             self.input_dim = config["cnnlstm"]["input_dim"]
#             self.hidden_dim = config["cnnlstm"]["hidden_dim"]
#             self.output_dim = config["cnnlstm"]["output_dim"]

        
#         self.conv1 = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)  # 卷积层
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层
#         self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)  # LSTM层
#         self.fc = nn.Linear(self.hidden_dim, self.output_dim)  # 全连接层
 
#     def forward(self, x):
#         x = x.unsqueeze(1) # [16, 1, 99, 13]
#         x = F.relu(self.conv1(x))  # 卷积层激活函数[16, 16, 99, 13]
#         x = self.pool(x)  # 池化层[16, 16, 49, 6]
#         x = x.squeeze(dim=1)
#         # x = x.permute(0, 2, 1)  # 调整维度顺序以适应LSTM输入
#         x = x.view(x.shape[0], x.shape[1], -1)  # [16, 16, 294]
#         _, (h_n, _) = self.lstm(x)  # LSTM层
#         x = h_n.squeeze(0)  # 去除LSTM输出的第一维（batch_size=1）
#         x = F.relu(x)  # LSTM层激活函数
#         x = self.fc(x)  # 全连接层
#         return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path):
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for data, labels in train_loader:
            optimizer.zero_grad()  # 清空梯度
            # images = images.float()
            outputs = model(data)  # 前向传播
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

    val_loader = getloader('val')

    test_loader = getloader('test')


    # # training
    # model = CNN()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # num_epochs = 20
    # save_path = 'model/CNN'
    # train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path)

    # model = LSTMModel()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # num_epochs = 20
    # save_path = 'model/LSTM'
    # train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path)

    # model = Resnet(12)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # num_epochs = 20
    # save_path = 'model/Restnet'
    # train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path)

    # model = CNNLSTM()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # num_epochs = 20
    # save_path = 'model/CNNLSTM'
    # train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path)

    model = Transformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    save_path = 'model/Transformer'
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path)


    # testing
    # model = LSTMModel()
    # model.load_state_dict(torch.load('model\LSTM\\19_model.pth'))
    # test_model(model, test_loader)

