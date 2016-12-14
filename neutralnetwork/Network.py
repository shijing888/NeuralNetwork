#coding=utf-8
'''
Created on Jul 20, 2016

@author: jshi.Sandy
'''
import numpy as np
import random
from sklearn import datasets
class Network(object):
    def __init__(self,sizes):
        '''
        parameters:
            sizes中保存了神经网络各层神经元个数
        functions:
                            对神经网络层与层之间的连接参数进行初始化
        '''
        #权重矩阵
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
        #偏置矩阵
        self.biases = [np.random.randn(x) for x in sizes[1:]]
        
    def init_parameters(self,parameters):
        '''
            functions:初始化模型参数
            parameters主要包括:
                epochs:迭代次数
                mini_batch_size:批处理大小
                eta:学习率
                nnLayers_size:神经网络层数
        '''
        self.epochs = parameters.get("epochs")
        self.mini_batch_size = parameters.get("mini_batch_size")
        self.eta = parameters.get("eta")
        self.nnLayers_size = parameters.get("nnLayers_size")
        
    def load_data(self):
        '''
            functions:加载数据集，这里使用的是sklearn自带的digit手写体数据集
        '''
        digits = datasets.load_digits()
        return digits.data, digits.target
    
    def feed_forword(self,data):
        '''
            parameters: 
                data:输入的图片表示数据，是一个一维向量
            functions:前向传导，将输入传递给输出，y = w*x + b
            return:传递到输出层的结果
        '''
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w,data) + b
            data = self.sigmoid(z)
        return data
    
    def sigmoid(self,z):
        '''
            functions:sigmoid函数
        '''
        return 1.0/(1.0+np.exp(-z))
    
    def crossEntrop(self,a, y):
        '''
            parameters:
                a:预测值
                y:真实值
            functions:交叉熵代价函数f=sigma(y*log(1/a))
        '''
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    def delta_crossEntrop(self,z,a,y):
        '''
            parameters:
                z:激活函数变量
                a:预测值
                y:真实值
        '''
        return self.sigmoid(z) - y
    
    def SGD(self,data):
        '''
            function:随即梯度下降算法来对参数进行更新
            parameters:
                data:数据集
        '''
        #数据集大小
        data_len = len(list(data))
        for _ in range(self.epochs):
            #将数据集按照指定大小划分成小的batch进行梯度训练，mini_batchs中的每个元素相当于一个小的样本集
            mini_batchs = [data[k:k+self.mini_batch_size]  for k in range(0,data_len,self.mini_batch_size)]
            
            for mini_batch in mini_batchs:
                #batch中的每个样本都会被用来更新参数
                self.update_parameter_by_mini_batch(mini_batch)

    def update_parameter_by_mini_batch(self,mini_batch):
        '''
            functions:按照梯度下降法批量对参数更新
        '''
        #首先初始化每个参数的偏导数
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        #将每个样本计算得到的参数偏导数进行累加
        for mini_x, mini_y in mini_batch:
            #每个样本通过后向传播得到两个导数张量，表示对w，b的导数
            delta_nabla_w,delta_nabla_b = self.derivative_by_backpropagate(mini_x, mini_y)
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
           
        self.weights = [w - self.eta * nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b - self.eta * nb for b,nb in zip(self.biases,nabla_b)]
        
    def derivative_by_backpropagate(self,x,y):
        '''
            functions:通过后向传播算法来计算每个参数的梯度值
        '''
        #首先初始化每个参数的偏导数
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        #激活值列表，元素为经过神经元后的激活输出，也即下一层的输入，此处记录下来用于计算梯度
        activations = [x]
        #线性组合值列表，元素为未经过神经元前的线性组合,z=w*x+b
        zs = []
        #初始输入
        activation = x
        #首先通过循环得到求导所需要的中间值
        for w, b in zip(self.weights,self.biases):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        #倒数第一层的导数计算，有交叉熵求导得来    
        delta = self.delta_crossEntrop(zs[-1],activations[-1], y) 
        nabla_w[-1] = np.dot(delta.reshape(len(delta),1), activations[-2].reshape(1,len(activations[-2])))
        nabla_b[-1] = delta
        #倒数第二层至正数第一层间的导数计算，有sigmoid函数求导得来
        for i in range(2,self.nnLayers_size):
            z = zs[-i]
            delta = np.dot(self.weights[-i+1].transpose(),delta.reshape(len(delta),1)) 
            delta_z = self.derivative_sigmoid(z)
            delta = np.multiply(delta, delta_z.reshape(len(delta_z),1))
            
            nabla_w[-i] = np.dot(np.transpose(delta),activations[-i].reshape(len(activations[-i]),1))
            delta = delta.reshape(len(delta))
            nabla_b[-i] = delta
            
        return (nabla_w,nabla_b)
     
    def derivative_sigmoid(self,z):
        '''
            functions:对sigmoid求导的结果
        '''
        return self.sigmoid(z) *(1-self.sigmoid(z)) 
    
    def evaluation(self,data):
        '''
            functions:性能评估函数
        '''
        result=[]
        right = 0
        for (x,y) in data:
            output = self.feed_forword(x)
            result.append((np.argmax(output),np.argmax(y)))
        
        for i,j in result:
            if(i == j):
                right += 1
        print("test data's size:",len(data))
        print("count of right prediction",right)
        print("the accuracy:",right/len(result))    
        return right/len(result)
    
    def suffle(self,data):
        '''
            parameters:
                data:元组数据
            functions:对数据进行打乱重组
        '''
        new_data = list(data)
        random.shuffle(new_data)
       
        return np.array(new_data)
    
    def transLabelToList(self,data_y):
        '''
            functions:将digit数据集中的标签转换成一个10维的列表，方便做交叉熵求导的计算
        '''
        data = []
        for y in data_y:
            item = [0,0,0,0,0,0,0,0,0,0]
            item[y] = 1
            data.append(item)
        return data
    
if __name__=="__main__":
    #神经网络的层数及各层神经元
    nnLayers = [64,15,10]
    nn=Network(nnLayers)
    parameters = {"epochs":100,"mini_batch_size":10,"eta":0.01,"nnLayers_size":len(nnLayers)}
    nn.init_parameters(parameters)
    #加载数据集
    data_x,data_y=nn.load_data()
    #将标签转换成一个10维列表表示，如1表示成[0,1,0,0,0,0,0,0,0,0]
    data_y = nn.transLabelToList(data_y)
    #将数据打包成元组形式
    data = zip(data_x,data_y)
    #将有序数据打乱
    data = nn.suffle(data)
    #将数据集划分为训练集和测试集
    train_data = data[:1500]
    test_data = data[1500:]
    nn.SGD(train_data)
    print(nn.evaluation(test_data))
