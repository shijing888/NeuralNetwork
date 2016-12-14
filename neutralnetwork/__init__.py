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
                    列表sizes保存了各层神经层的神经元个数
                    初始化时对权重和偏置初始化
        '''
        self.num_layers=len(sizes)
        self.sizes=sizes
        #有几个神经元就随机产生几个偏置
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        #产生相邻神经层的权重
        self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
#         print('权重：',self.weights[0][0][0])
    #加载数据集
    def loadData(self):
        digits=datasets.load_digits()
        return digits.data,digits.target
        
    #前向传导
    def feedforward(self,input_a):
        '''
                    神经层间的前向传导，返回输入经过神经层加工后的结果output_a=sigmoid(w*x+b)
        '''
        for w,b in zip(self.weights,self.biases):
#             print 'wx+b:',(np.dot(w,input_a)+b)[0]
            input_a=self.sigmoid(np.dot(w,input_a)+b)
#             print 'sig:',input_a[0]
        return input_a
    
    #sigmoid函数
    def sigmoid(self,z):
        '''
        sigmoid=1/(1+e^(-z))
        '''
#         print "len:",len(z),'z:',z,'sigmoid:',1.0/(1.0+np.exp(-z))
        return 1.0/(1.0+np.exp(-z))
    
    #mini-batch随机梯度下降算法
    def SGD(self,train_data,epochs,mini_batch_size,eta,test_data=None):
        '''
        train_data为训练数据集，每个元素为元组(x,y)；epochs为迭代次数；
        mini_batch_size为批处理的数据大小；eta为学习率
        '''
        if test_data:
            test_len=len(test_data)
        train_len=len(train_data)
        for i in range(8):
            #将训练集随机打乱
            random.shuffle(train_data)
            #将训练数据分批
            mini_batches=[
                          train_data[k:k+mini_batch_size]
                          for k in range(0,train_len,mini_batch_size)]
            #批量更新
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
#                 print'weight00:',self.weights[0][0][0]
                print ('Epoch{0}:{1}/{2}'.format(i,self.evaluate(test_data),test_len))
            else:
                print ('Epoch{0} is completed'.format(i))
    
    #批量更新参数函数
    def update_mini_batch(self,mini_batch,eta):
        '''
                    使用梯度下降法和后向传播算法对参数进行更新
        mini_batch是一个元组(x,y)列表,eta是一个学习率
        w=w-eta*(损失函数对w的偏导)
        '''
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        for x,y in mini_batch:
            delta_nabla_w,delta_nabla_b=self.backPropagate(x,y)
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
        self.weights=[w-(eta/len(mini_batch))*nw
                      for w,nw in zip(self.weights,nabla_w)]
#         print('w:', self.weights[0][0][0])
        self.biases=[b-(eta)/len(mini_batch)*nb
                     for b,nb in zip(self.biases,nabla_b)]
        
    #后向传播
    def backPropagate(self,x,y):
        '''
                     输入为一个样本元组(x,y)，输出为梯度元组(nable_w,nable_b)
        '''  
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        activation=x
        activations=[x]
        zs=[]
        for w,b in zip(self.weights,self.biases):
            z=np.dot(w,activation)+b
#             print('w:',w[0][0],'activation:',activation[0])
#             print ('z:',z[0])
            zs.append(z)
            activation=self.sigmoid(z)
            activations.append(activation)
        
        delta=self.cost_derivative(activations[-1],y)*self.sigmoid_prime(zs[-1])
        nabla_w[-1]=np.dot(delta,activations[-1].transpose())
        nabla_b[-1]=delta
#         print delta.shape
        for i in range(2,self.num_layers):
            z=zs[-i]
            sp=self.sigmoid_prime(z)
            delta=np.dot(self.weights[-i+1].transpose(),delta)*sp
#             print 'delta:',delta.shape,activations[-i].transpose().shape
            nabla_w[-i]=np.dot(delta,activations[-i].transpose())
            nabla_b[-i]=delta
        return (nabla_w,nabla_b)
    
    #sigmoid求导
    def sigmoid_prime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    #计算损失函数偏导
    def cost_derivative(self,output_activations,y):
        return output_activations-y
    
    def evaluate(self,test_data):
        test_results=[]
        for (x,y)in test_data:
#             print self.feedforward(x)
#             print (np.argmax(self.feedforward(x)), y)
            test_results.append((np.argmax(self.feedforward(x)), y))
#         test_results = [(np.argmax(self.feedforward(x)), y)
#                         for (x, y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
    
if __name__=="__main__":
    nn=Network([64,30,10])
    data,data_name=nn.loadData()
    train_data=list(zip(data[:1500],data_name[:1500]))
    test_data=list(zip(data[1500:],data_name[1500:]))
    nn.SGD(train_data, 50, 10, 3.0, test_data=test_data)
#     print nn.evaluate(test_data)
#     for i in range(8):
#         for j in range(8):
#             print str(1 if(int(data[i*8+j])>0) else 0)+" ",
#         print '\n'
        
#     nn.SGD(train_data, epochs, mini_batch_size, eta, test_data)
   