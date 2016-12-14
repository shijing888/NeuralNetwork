'''
Created on 2016年12月11日

@author: shijing
'''
import numpy as np
if __name__ == '__main__':
#     l1=np.array([1,2,3,4,5])
#     l2=np.array([2,3,4,5,6,7])
#     print(np.shape(l1))
#     print(np.shape(l2))
#     l3=np.dot(l1.reshape(len(l1),1),l2.reshape(1,len(l2)))
#     print(l3)
    
    l1 = np.array([1,2,3,4])
    l2 = np.array([2,2,3,3])
#     print("l1,shape",np.shape(l1))
#     print("l2,shape",np.shape(l2))
#     l3 = np.multiply(l1,l2)
#     l4 = np.dot(l1,l2)
#     l5 = np.matmul(l1,np.transpose(l2))
#     print("l3",l3)
#     print("l4",l4)
#     print("l5",l5)
#     
#     l0 = np.array([1,2,3,4])
#     print(np.reshape(l0, [len(l0),1]))
#     print(np.reshape(l0,[len(l0)]))
#     print(np.exp(-l1))
    data_y = [1,2,3,4,5]
    data=[]
    for y in data_y:
        item = [0,0,0,0,0,0,0,0,0,0]
        item[y] = 1
        data.append(item)
    print(data)