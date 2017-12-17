import collections
import random
import re
import os
import numpy as np
import torch
import torch.nn.functional as F
from myclassifier.model_CNN import CNN_text
from myclassifier.model_LSTM import LSTM_model
from myclassifier.model_GRU import GRU_model
from myclassifier.model_BiLSTM import BiLSTM_model
from myclassifier.model_Attention_LSTM import Attention_LSTM_model
from myclassifier.model_Attention_BiLSTM import Attention_BiLSTM

class Instance:
    def __init__(self):
        """
        m_word:生成一句话的单词list
        m_label:生成一句话对应的一个标签
        """
        self.m_word=[]
        self.m_label=0

class Example:
    def __init__(self):
        """
        word_indexes:一个句子所有单词的下标
        label_index：一个句子的标签
        """
        self.word_indexes=[]
        self.label_index=[]

class GenerateDic:
    """
    这个类主要是生成字典
    """
    def __init__(self):
        self.v_list=[]
        self.v_dict=collections.OrderedDict()        #固定字典的生成

    def produceDic(self,listName):
        for i in range(len(listName)):
            if listName[i] not in self.v_list:
                self.v_list.append(listName[i])
        self.v_list.append("unknow")
        # print("v_list",self.v_list)

        for j in range(len(self.v_list)):
            self.v_dict[self.v_list[j]]=j

        return self.v_dict


class Hyperparmeter:
    def __init__(self):
        self.learnRate = 0.001
        self.epochs = 0
        self.hidden_dim = 0     #隐层给的窗口数目
        self.label_size = 0
        self.embed_num = 0
        self.embed_dim = 300
        self.class_num = 2
        self.batch = 32
        self.unknow = None
        self.kernel_num = 200
        self.kernel_sizes = [3, 4, 5]
        self.word_Embedding = True
        self.word_Embedding_path = "./data/converted_word_Subj.txt"
        self.save_dir="snapshot"
        self.snapshot=None
        self.save_interval=100
        self.pretrained_weight = None
        self.dropout = 0.2  #
        self.dropout_embed = 0   #
        self.LSTM_hidden_dim = 300
        self.num_layers = 1 #单层的lstm
        self.CNN_model=False
        self.LSTM_model = False
        self.GRU_model=False
        self.BiLSTM_model=False
        self.Attention_LSTM_model=True
        self.Attention_BiLSTM_model=False
        self.attention_size= 100  #attention模型中这个维度可以自己随意设置

class LoadDoc:
    def readFile(self, path):
        f = open(path, 'r')
        newList=[]
        count = 0
        for line in f.readlines():
            count += 1
            instance=Instance()
            label,seq,sentence=line.partition(" ")
            sentence=LoadDoc.clean_str(sentence)
            instance.m_word.append(sentence.split(" "))
            instance.m_label=label
            #print("instance:",instance)
            newList.append(instance)
            if count == -1:
                break
        random.shuffle(newList)
       # print("newlist：",newList[1])
        return newList[:int(len(newList)*0.7)],\
               newList[int(len(newList)*0.7) : int(len(newList)*0.8)],\
               newList[int(len(newList) * 0.8):],
        #切片分成把分割后的数据集 7:2:1作为train dev test数据
    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip()
# loadDoc=LoadDoc()
# loadDoc.readFile("./data/sample.txt")

class Classifier:
    def __init__(self):
        self.param=Hyperparmeter()
        self.lableDic = GenerateDic()
        self.wordDic = GenerateDic()

    def ToList(self,InitList):
        """
        :param InitList: 相当于很多个instance集合，instance是一个句子一个标签的集合
        :return:
        """
        wordList=[]
        labelList=[]
        for i in range(len(InitList)):
            for j in InitList[i].m_word[0]:
                wordList.append(j)
            labelList.append(InitList[i].m_label)
        return wordList,labelList

    def change_Sentence_in_Num(self,dic_List,word_dic,label_dic):
        """
        这个函数就是一个查字典的过程
        :param dic_List: 传入的是一个字典类型的list[(key,value),(key,value),(key,value)...]
        :param word_dic:
        :param label_dic:
        :return:
        """
        example_list_id=[]
        for i in range(len(dic_List)):
            exampleId=Example()
            for j in dic_List[i].m_word[0]:
                if j in word_dic:
                    id=word_dic[j]
                else:
                    id=word_dic["unknow"]
                exampleId.word_indexes.append(id)
            num=label_dic[dic_List[i].m_label]
            exampleId.label_index.append(num)
            example_list_id.append(exampleId)
        return example_list_id

    def load_my_vec(self, path, vocab, freqs, k=None):
        word_vecs = {}
        with open(path, encoding="utf-8") as f:
            count = 0
            lines = f.readlines()[1:]
            for line in lines:
                values = line.split(" ")
                word = values[0]
                count += 1
                if word in vocab:  # whether to judge if in vocab
                    vector = []
                    for count, val in enumerate(values):
                        if count == 0:
                            continue
                        if count <= k:
                            vector.append(float(val))
                    word_vecs[word] = vector
        return word_vecs

    def add_unknow_words_by_uniform(self,word_vec,vocab,k=100):
        list_word2vec=[]
        outvec=0            #不在此向量中
        invec=0             #在词向量中的
        for word in vocab:
            if word not in word_vec:
                outvec+=1
                word_vec[word]=np.random.uniform(-0.25, 0.25, k).round(6).tolist()
                list_word2vec.append(word_vec[word])
            else:
                invec+=1
                list_word2vec.append(word_vec[word])

        return list_word2vec

    def train(self,dataSet):
        random.seed(100)
        torch.manual_seed(100)
        loadDocument=LoadDoc()      #加载数据集
        InitRst_train ,InitRst_test, InitRst_dev =loadDocument.readFile(dataSet)#得到train,test,dev数据集
        # InitRst_test=loadDocument.readFile(test_path)
        # InitRst_dev=loadDocument.readFile(dev_path)
        # InitRst_train=loadDocument.readFile(train_path)
        (word_train,label_train) = self.ToList(InitRst_train)   #得到句子集合，标签集合
        words_dict = self.wordDic.produceDic(word_train)
        labels_dict = self.lableDic.produceDic(label_train)

        labels_dict.pop("unknow")   #去掉unknow这个标签

        word2vec = self.load_my_vec(path=self.param.word_Embedding_path,
                                    vocab=words_dict, freqs=None, k=300)
        self.param.pretrained_weight = self.add_unknow_words_by_uniform(word_vec=word2vec,
                                                                            vocab=words_dict,
                                                                         k=300)
        #把所有的词，标签转化为字典的index
        Example_list_train = self.change_Sentence_in_Num(InitRst_train,words_dict,labels_dict)#dic_List,word_dic,label_dic
        Example_list_dev = self.change_Sentence_in_Num(InitRst_dev,words_dict,labels_dict)
        Example_list_test = self.change_Sentence_in_Num(InitRst_test,words_dict,labels_dict)

        self.param.unknow=words_dict["unknow"]
        self.param.embed_num=self.param.unknow + 1
        self.param.label_size=len(labels_dict)
        # 优化器
        #self.model = CNN_text(self.param)
        if self.param.LSTM_model:
            self.model=LSTM_model(self.param)
        elif self.param.GRU_model:
            self.model = GRU_model(self.param)
        elif self.param.BiLSTM_model:
            self.model=BiLSTM_model(self.param)
        elif self.param.Attention_LSTM_model:
            self.model=Attention_LSTM_model(self.param)
        elif self.param.Attention_BiLSTM_model:
            self.model=Attention_BiLSTM(self.param)
        else:
            self.model=CNN_text(self.param)
        print(self.model)

        # optimizer=torch.optim.Adagrad(self.model.parameters(),lr=self.parameter.learnRate)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param.learnRate)
        steps=0
        total_num = len(Example_list_train)
        print('train num:', total_num)
        batch = self.param.batch
        if total_num % batch == 0:
            num_batch = total_num // batch
        else:
            num_batch = total_num // batch + 1

        self.model.train()

        for x in range(1, 20):
            random.shuffle(Example_list_train)
            print('这是第%d次迭代' % x)
            correct = 0
            sum = 0
            for i in range(num_batch):
                batch_list = []
                # print("Running create batch {}".format(i))
                for j in range(i * batch,
                               (i + 1) * batch if (i + 1) * batch < len(Example_list_train) else len(Example_list_train)):
                    batch_list.append(Example_list_train[j])
                random.shuffle(batch_list)  # 进行重新洗牌
                feature, target = self.toVariable(batch_list)
                optimizer.zero_grad()  #
                self.model.zero_grad()

                # self.model.hidden=self.model.init_hidden(self.param.num_layers, self.param.batch)
                if self.param.LSTM_model:
                    if feature.size(0) == self.param.batch:
                        self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                                   self.param.batch)
                    else:
                        self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                                   feature.size(0))
                elif self.param.GRU_model:
                    if feature.size(0) == self.param.batch:
                        self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                                   self.param.batch)
                    else:
                        self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                                   feature.size(0))
                elif self.param.BiLSTM_model:
                    if feature.size(0) == self.param.batch:
                        self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                                   self.param.batch)
                    else:
                        self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                                   feature.size(0))
                elif self.param.Attention_LSTM_model:
                    if feature.size(0) == self.param.batch:
                        self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                                   self.param.batch)
                    else:
                        self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                                   feature.size(0))
                elif self.param.Attention_BiLSTM_model:
                    if feature.size(0) == self.param.batch:
                        self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                                   self.param.batch)
                    else:
                        self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                                   feature.size(0))

                logit = self.model(feature)
                loss = F.cross_entropy(logit, target)  # 目标函数的求导
                # print('loss:',loss.  data[0])
                loss.backward()
                optimizer.step()
                steps+=1
                for i in range(len(target)):
                    if (target.data[i] == self.getMaxIndex(logit[i].view(1, self.param.label_size))):
                        correct += 1
                    sum += 1
            print('train acc:{} correct / sum {} / {}'.format(correct / sum, correct, sum))
            self.eval(Example_list_dev, self.model, self.param)
            if not os.path.isdir(self.param.save_dir):
                os.makedirs(self.param.save_dir)
            save_prefix=os.path.join(self.param.save_dir,"snapshot")
            save_path='{}_steps{}.pt'.format(save_prefix, steps)
            torch.save(self.model,save_path)
            self.eval(Example_list_test, self.model,self.param)

    def eval(self,data_iter,model,param):
        """
        :param dataSet: 传入的数据集
        :param model:
        :return:
        """
        # model.dev_eval()
        total_num=len(data_iter)
        print("eval_num:",total_num)
        batch = self.param.batch
        if total_num % batch == 0:
            num_batch = total_num // batch
        else:
            num_batch = total_num // batch + 1

        #for x in range(1, 2):
        random.shuffle(data_iter)
        #print('这是第%d次迭代' % x)
        correct = 0
        sum = 0
        for i in range(num_batch):
            batch_list = []
            # print("Running create batch {}".format(i))
            for j in range(i * batch,
                           (i + 1) * batch if (i + 1) * batch < len(data_iter) else len(
                               data_iter)):
                batch_list.append(data_iter[j])
            random.shuffle(batch_list)  # 进行重新洗牌
            feature, target = self.toVariable(batch_list)

            if self.param.LSTM_model:
                if feature.size(0) == self.param.batch:
                    self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                               self.param.batch)
                else:
                    self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                               feature.size(0))
            elif self.param.GRU_model:
                if feature.size(0) == self.param.batch:
                    self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                               self.param.batch)
                else:
                    self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                               feature.size(0))
            elif self.param.BiLSTM_model:
                if feature.size(0) == self.param.batch:
                    self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                               self.param.batch)
                else:
                    self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                               feature.size(0))
            elif self.param.Attention_LSTM_model:
                if feature.size(0) == self.param.batch:
                    self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                               self.param.batch)
                else:
                    self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                               feature.size(0))
            elif self.param.Attention_BiLSTM_model:
                if feature.size(0) == self.param.batch:
                    self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                               self.param.batch)
                else:
                    self.model.hidden = self.model.init_hidden(self.param.num_layers,
                                                               feature.size(0))
            # optimizer.zero_grad()  #


            logit = model(feature)

            loss = F.cross_entropy(logit, target, size_average=False)  # 目标函数的求导
            # print('loss:',loss.  data[0])
            # loss.backward()
            # optimizer.step()
            for i in range(len(target)):
                if (target.data[i] == self.getMaxIndex(logit[i].view(1, self.param.label_size))):
                    correct += 1
                sum += 1

        print('eval acc:{} correct / sum {} / {}'.format(correct / sum, correct, sum))


    def toVariable(self,example_batch): #传入一个batch为单位的example
        batch =len(example_batch)
        maxLenth = 0
        for i in range(len(example_batch)): #得到每一个batch中 每一句话的最大长度
            if maxLenth < len(example_batch[i].word_indexes):
                maxLenth = len(example_batch[i].word_indexes)
        x = torch.autograd.Variable(torch.LongTensor(batch, maxLenth))#转化为batch*maxLenth的向量
        y = torch.autograd.Variable(torch.LongTensor(batch))    #转化为 1*batch
        for i in range(0,len(example_batch)):
            for j in range (len(example_batch[i].word_indexes)):
                x.data[i][j]= example_batch[i].word_indexes[j]
                for n in range(len(example_batch[i].word_indexes), maxLenth):     #这几句是处理没有以batch为单位的大小，剩余的example
                    x.data[i][n]=self.param.unknow
            y.data[i] = example_batch[i].label_index[0]
        return x,y

    def getMaxIndex(self, score):  # 获得最大的下标
        labelsize = score.size()[1]
        max = score.data[0][0]
        maxIndex = 0
        for idx in range(labelsize):
            tmp = score.data[0][idx]
            if max < tmp:
                max = tmp
                maxIndex = idx
        return maxIndex

classifier=Classifier()
classifier.train("./data/subj.all")
# classifier.train("./data/custrev.all")