import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class LSTM_model(nn.Module):
    def __init__(self,parameter):
        super(LSTM_model,self).__init__()
        self.parameter=parameter
        self.embed_num = self.parameter.embed_num  # dict为字典类型 其长度比编号大一
        self.embed_dim = self.parameter.embed_dim
        self.class_num = self.parameter.label_size  # 几分类
        self.hidden_dim = self.parameter.LSTM_hidden_dim
        self.num_layers = self.parameter.num_layers

        print("embed_dim", self.embed_dim)
        print("使用LSTM模型")

        self.embedding = nn.Embedding(self.embed_num, self.embed_dim)
        # 预训练词向量
        if self.parameter.word_Embedding:
            pretrained_weight = np.array(self.parameter.pretrained_weight)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.embedding.weight.requires_grad = True

        self.lstm=nn.LSTM(self.embed_dim, self.hidden_dim,dropout=self.parameter.dropout,num_layers=self.parameter.num_layers)
        self.fc1 = nn.Linear(self.hidden_dim, self.class_num)
        self.hidden=self.init_hidden(self.num_layers, self.parameter.batch)
        # print("hidden:",self.hidden)
        self.dropout = nn.Dropout(self.parameter.dropout)
        # self.fc1 = nn.Linear(parameter.embed_dim, parameter.label_size)    #把两矩阵进行合并需求矩阵，1*50与50*5向量相乘得到1*5
        self.dropout_embed = nn.Dropout(self.parameter.dropout_embed)

    def init_hidden(self,num_layers,batch):

        return (Variable(torch.zeros(1 * num_layers, batch, self.hidden_dim)),
                Variable(torch.zeros(1 * num_layers ,batch, self.hidden_dim)))


    def forward(self, x):
        x = self.embedding(x)  # 把variable转为一段一段list集合  (N,W,embed_dim)
        x = self.dropout_embed(x)
        x = x.view(len(x), x.size(1), -1)         #batch *N * embed_num
        #lstm
        # print("x:",x.size())
        lstm_out,self.hidden=self.lstm(x.permute(1, 0, 2),self.hidden)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)

        #pooling
        lstm_out=F.tanh(lstm_out)
        # print("lstm_out:",lstm_out.size())      #batch * embed_num * N
        lstm_out=F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)    #[N,hidden_dim]
        lstm_out=F.tanh(lstm_out)

        # Linear
        logit=self.fc1(lstm_out)    #[batch , hidden_dim]
        # print("lstm_out",lstm_out.size())

        return logit



