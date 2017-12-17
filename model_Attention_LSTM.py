import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np

class Attention_LSTM_model(nn.Module):
    def __init__(self,parameter):
        super(Attention_LSTM_model,self).__init__()
        self.parameter=parameter
        self.embed_num = self.parameter.embed_num  # dict为字典类型 其长度比编号大一
        self.embed_dim = self.parameter.embed_dim
        self.class_num = self.parameter.label_size  # 几分类
        self.hidden_dim = self.parameter.LSTM_hidden_dim
        self.num_layers = self.parameter.num_layers

        self.attention_size=self.parameter.attention_size   #这个维度可以随意设置，这里我设置为50

        # print("attention_size",self.attention_size)
        self.batch_size = self.parameter.batch
        self.U=Parameter(torch.randn(self.attention_size,1))       #随机矩阵 attention_size *1
        # print("attention",self.U.size())
        print("attention_size:",self.attention_size)
        print("batch_size:",self.batch_size)
        print("embed_dim", self.embed_dim)
        print("使用Attention_LSTM_model模型")

        self.embedding = nn.Embedding(self.embed_num, self.embed_dim)
        # 预训练词向量
        if self.parameter.word_Embedding:
            pretrained_weight = np.array(self.parameter.pretrained_weight)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.embedding.weight.requires_grad = True

        self.lstm=nn.LSTM(self.embed_dim, self.hidden_dim,dropout=self.parameter.dropout,num_layers=self.parameter.num_layers,batch_first=True)

        self.hidden_label1 = nn.Linear(self.hidden_dim, self.attention_size)

        self.hidden_label2 = nn.Linear(self.hidden_dim, self.class_num)

        self.hidden=self.init_hidden(self.num_layers, self.parameter.batch)
        # print("hidden:",self.hidden)
        self.dropout = nn.Dropout(self.parameter.dropout)
        # self.fc1 = nn.Linear(parameter.embed_dim, parameter.label_size)    #把两矩阵进行合并需求矩阵，1*50与50*5向量相乘得到1*5
        self.dropout_embed = nn.Dropout(self.parameter.dropout_embed)

    def init_hidden(self,num_layers,batch):

        return (Variable(torch.zeros(1 * num_layers, batch, self.hidden_dim)),
                Variable(torch.zeros(1 * num_layers ,batch, self.hidden_dim)))


    def forward(self, x):
        x = self.embedding(x)  # [batch , N ，embedding_dim] = [16, 48, 300]
        # print("x:",x.size())
        x = self.dropout_embed(x)
        x,self.hidden= self.lstm(x,self.hidden)   # [batch , N ，embedding_dim] = [16, 48, 300]
        # print("ather lstm:",x.size())
        #linear
        lstm_out=self.hidden_label1(x)   #[batch ，N ，attention_size] = [16, 48, 50]
        # print("after linear1: ",lstm_out.size())
        #tanh
        lstm_out = F.tanh(lstm_out)
        # print("after tanh:",lstm_out.size())    #[batch ，N ，attention_size] = [16, 48, 50]
        #加入矩阵过程中，先降维度，然后矩阵相乘，然后在升维度
        lstm_out1=lstm_out.view((lstm_out.size(0)*lstm_out.size(1)),lstm_out.size(2))#[(batch*N),attention_size]=[(16*48), 50]
        # print("after view",lstm_out1.size())

        lstm_out2=torch.mm(lstm_out1, self.U)     #[(batch*N),1]
        # print("after multiply:",lstm_out2.size())

        lstm_out3=lstm_out2.view(lstm_out.size(0),lstm_out.size(1)) #[batch , N]= [16 * 48]
       # print("after view", lstm_out3.size())
        lstm_out=F.softmax(lstm_out3)

        # print(lstm_out.size(0))

        for i in range(x.size(0)):
            if i == 0:
                c = torch.mm((torch.t(x[i])), lstm_out[i].unsqueeze(1))     #[hidden_size ,1]
                # print("cccc:",c.size())
            else:
                c = torch.cat([c,torch.mm((torch.t(x[i])), lstm_out[i].unsqueeze(1))], 1) #[hidden_size,batch]
                #print(lstm_out.size())
        # Linear
        lstm_out=torch.t(c)     #矩阵转置
        logit=self.hidden_label2(lstm_out)
        # print("logit",logit.size())
        return logit



