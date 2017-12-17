import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
from torch.autograd import Variable

class Attention_BiLSTM(nn.Module):
    def __init__(self,parameter):
        super(Attention_BiLSTM,self).__init__()
        self.parameter=parameter
        self.embedding_num=self.parameter.embed_num
        self.embedding_dim=self.parameter.embed_dim
        self.class_num=self.parameter.label_size
        self.num_layers=self.parameter.num_layers
        self.hidden_dim=self.parameter.LSTM_hidden_dim

        self.attention_size=self.parameter.attention_size
        self.embedding = nn.Embedding(self.embedding_num, self.embedding_dim)
        self.batch_size=self.parameter.batch
        self.learn_rate=self.parameter.learnRate
        self.U=Parameter(torch.randn(self.attention_size, 1))   #随机矩阵大小
        print("U",self.U.size())
        print("learn_rate:",self.learn_rate)
        print("batch_size:",self.batch_size)
        print("attention_size:",self.attention_size)
        print("embedding_dim", self.embedding_dim)
        print("使用Attention BiLSTM模型")
        # 预训练词向量
        if self.parameter.word_Embedding:
            pretrained_weight = np.array(self.parameter.pretrained_weight)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.embedding.weight.requires_grad = True
        self.bilstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, dropout=self.parameter.dropout,
                              num_layers=self.parameter.num_layers, batch_first=True, bidirectional=True)
        self.hidden1label1= nn.Linear(self.hidden_dim,self.attention_size)
        self.hidden2label1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        # print("hidden2lable1=",self.hidden2label1)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, self.class_num)
        self.hidden = self.init_hidden(self.num_layers, self.parameter.batch)

    def init_hidden(self, num_layers, batch):

        return (Variable(torch.zeros(2 * num_layers, batch, self.hidden_dim // 2)),
                Variable(torch.zeros(2 * num_layers, batch, self.hidden_dim // 2)))

    def forward(self,x):
        x=self.embedding(x)     #[batch, N，embedding_dim]
        # print("after embedding:",x.size())
        bilstm_out,self.hidden=self.bilstm(x, self.hidden)  #[batch ,N, hidden_dim]
        # print("after bilstm:",bilstm_out.size())

        linearout=self.hidden1label1(bilstm_out)
        # print("linearout:",linearout.size())    #[batch,N,atttention_size]

        tanh_out=F.tanh(linearout)

        view_out=tanh_out.view((tanh_out.size(0)*tanh_out.size(1)),tanh_out.size(2))
        # print("view_out:",view_out.size())
        #矩阵相乘
        bilstm_out1=torch.mm(view_out, self.U)   #[(batch*N),1]
        # print("bilstm_out:",bilstm_out1.size())

        view_out1=bilstm_out1.view(tanh_out.size(0),tanh_out.size(1))#[batch * N]
        # print("view_out1:",view_out1.size())

        softmax_out=F.softmax(view_out1)        #[batch * N]
        # print("after softmax:",softmax_out.size())
         #矩阵#[batch ,N, hidden_dim] 与矩阵 #[batch * N]相乘  结果是[hidden_dim , batch]
        for idx in range (softmax_out.size(0)):
            if idx == 0:
                c = torch.mm((torch.t(bilstm_out[idx])),softmax_out[idx].unsqueeze(1))
            else :
                c = torch.cat([c,torch.mm((torch.t(bilstm_out[idx])),softmax_out[idx].unsqueeze(1))],1)
            # print("after cat:",c.size())
        logit=self.hidden2label1(torch.t(c))
        # print("after linear",logit.size())
        logit=self.hidden2label2(logit)

        return logit


