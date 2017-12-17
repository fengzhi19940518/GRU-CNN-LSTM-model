import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class BiLSTM_model(nn.Module):
    def __init__(self,parameter):
        super(BiLSTM_model,self).__init__()
        self.parameter=parameter
        self.embed_num = self.parameter.embed_num  # dict为字典类型 其长度比编号大一
        self.embed_dim = self.parameter.embed_dim
        self.class_num = self.parameter.label_size  # 几分类
        self.hidden_dim = self.parameter.LSTM_hidden_dim
        self.num_layers = self.parameter.num_layers

        print("embed_dim", self.embed_dim)
        print("使用BiLSTM模型")

        self.embedding = nn.Embedding(self.embed_num, self.embed_dim)
        # 预训练词向量
        if self.parameter.word_Embedding:
            pretrained_weight = np.array(self.parameter.pretrained_weight)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.embedding.weight.requires_grad = True

        self.bilstm=nn.LSTM(self.embed_dim, self.hidden_dim//2, dropout=self.parameter.dropout, num_layers=self.parameter.num_layers, batch_first=True, bidirectional=True)
        self.hidden2label1=nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        # print("hidden2lable1=",self.hidden2label1)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, self.class_num)
        # print("hidden2label2=",self.hidden2label2)
        self.hidden=self.init_hidden(self.num_layers, self.parameter.batch)
        # print("hidden:",self.hidden)
        self.dropout = nn.Dropout(self.parameter.dropout)
        # self.fc1 = nn.Linear(parameter.embed_dim, parameter.label_size)    #把两矩阵进行合并需求矩阵，1*50与50*5向量相乘得到1*5
        self.dropout_embed = nn.Dropout(self.parameter.dropout_embed)

    def init_hidden(self,num_layers,batch):

        return (Variable(torch.zeros(2 * num_layers, batch, self.hidden_dim // 2)),
                Variable(torch.zeros(2 * num_layers ,batch, self.hidden_dim // 2)))


    def forward(self, x):
        x = self.embedding(x)  # 把variable转为一段一段list集合  (batch,N,embed_dim)
        # print("x:",x.size())
        x = self.dropout_embed(x)
        # x = x.view(len(x), x.size(1), -1)         #batch *N * embed_num
        # lstm
        # x = x.permute(1, 0, 2)        #[48, 16, 300]
        # print("x:",x.size())
        bilstm_out,self.hidden=self.bilstm(x, self.hidden)   #[48, 16, 300]
        # print("yyyy:", bilstm_out.size())
        # bilstm_out = torch.transpose(bilstm_out, 0, 1)
        # bilstm_out = torch.transpose(bilstm_out, 1, 2)  #[16, 150, 48]
        # print("xxxx=",bilstm_out.size())
        #pooling
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.tanh(bilstm_out)
        # print("lstm_out:",bilstm_out.size())      #batch * embed_num * N  =[16, 300, 48]
        bilstm_out=F.max_pool1d(bilstm_out, bilstm_out.size(2))    #[N,hidden_dim]=[16, 300,1]
        # print("bilstm:",bilstm_out.size())
        bilstm_out=bilstm_out.squeeze(2)        #[16, 300]
        # print("bilstm:bilstm",bilstm_out.size())
        bilstm_out=F.tanh(bilstm_out)

        # Linear
        logit = self.hidden2label1(bilstm_out)
        # logit=torch.transpose(logit,0,1)
        logit=self.hidden2label2(logit)

        return logit



