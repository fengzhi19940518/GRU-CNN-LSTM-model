import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class CNN_text(nn.Module):

    def __init__(self,parameter):
        super(CNN_text, self).__init__()
        self.parameter = parameter
        self.embed_num = parameter.embed_num
        self.embed_dim= parameter.embed_dim
        print("使用CNN模型训练")
        print("embed_num",parameter.embed_num)
        print("embed_dim",self.embed_dim)

        self.embedding=nn.Embedding(self.embed_num,self.embed_dim)
        #预训练词向量
        if self.parameter.word_Embedding:
            pretrained_weight=np.array(self.parameter.pretrained_weight)
            # print(len(pretrained_weight))
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.embedding.weight.requires_grad=True

        self.convs=[nn.Conv2d(in_channels=1,out_channels=self.parameter.kernel_num,
                              kernel_size=(k,self.embed_dim),bias=True)
                    for k in self.parameter.kernel_sizes]

        #print(self.embed_dim, self.label_size)
        self.dropout=nn.Dropout(self.parameter.dropout)
        self.fc1=nn.Linear(self.parameter.kernel_num *
                           len(self.parameter.kernel_sizes),self.parameter.label_size)
        self.dropout_embed = nn.Dropout(self.parameter.dropout_embed)
       # self.fc1 = nn.Linear(parameter.embed_dim, parameter.label_size)    #把两矩阵进行合并需求矩阵，1*50与50*5向量相乘得到1*5
        self.dropout_embed = nn.Dropout(self.parameter.dropout_embed)

    def forward(self,x):
        #print("x: ", x)
        x = self.embedding(x)   #把variable转为一段一段list集合
        x = self.dropout_embed(x)
        #print("x: ", x)
        #x=x.permute(0,2,1)      #表示你想得到的维数,调维度
        # x = F.max_pool1d(x.permute(0, 2, 1), x.size()[1]).squeeze(2)
        x=x.unsqueeze(1)
        x=[F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x=[F.max_pool1d(i,i.size(2)).squeeze(2) for i in x ]
        x=torch.cat(x, 1)
        #print('维数:',x.size())         #打印维数
        #print('v2:', x.squeeze(2).size())
        #logit=self.fc1(x.view(1, self.parameter.embed_dim))
        x=self.dropout(x)
        logit=self.fc1(x)

        return logit









