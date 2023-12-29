# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import collections
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab
import paddle.fluid as fluid
import time
import paddle
# from phe import paillier
import os
import paddle.static as static
# from Pyfhel import Pyfhel, PyPtxt, PyCtxt
import pickle
import tenseal as ts
import copy
import warnings
from paddle.vision.datasets import FashionMNIST
from paddle.vision.datasets import MNIST

import paddle.nn as nn
import paddle.optimizer as opt
from paddle.static import InputSpec

import paddle.nn.functional as F

import random

from fedlab.utils.dataset import MNISTPartitioner, CIFAR10Partitioner
from fedlab.utils.functional import partition_report
from matplotlib import pyplot as plt

import torchvision
import torchvision.transforms as transforms


#生成公私钥对
#16384
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree = 16384,
    coeff_mod_bit_sizes=[60, 40, 60]
    )
context.generate_galois_keys()
context.global_scale = 2**40

warnings.filterwarnings("ignore")

# 获取每个用户的模型
get_model = []

# 数据集1--minst
mnist_data_train = np.load('F:/研零/Zhao/train_mnist.npy')
mnist_data_test = np.load('F:/研零/Zhao/test_mnist.npy')
Fake_Data = np.load('F:/研零/Zhao/train_mnist.npy')
print("There are {} images for training".format(len(mnist_data_train)))
print("There are {} images for testing".format(len(mnist_data_test)))
# 数据和标签分离 便于后续处理
Label = [int(i[0]) for i in mnist_data_train]
Data = [i[1:] for i in mnist_data_train]
# 随机查看图像
img = np.reshape(Data[np.random.randint(0, 60000)], (28, 28))
plt.imshow(img)
pylab.show()
# 假数据处理
fake_data = [i[1:] for i in Fake_Data]
fake_data_label = [int(i[0]) for i in Fake_Data]

# # 数据集2——mnist_fasion
# mnist_fasion = FashionMNIST(mode = 'train')
# Fake_Data = FashionMNIST(mode = 'test')
# # 看数据集长度
# print("There are {} images for training".format(len(mnist_fasion)))
# # 标签与数据分开
# Label = [int(i[1]) for i in mnist_fasion]
# Data = [i[0] for i in mnist_fasion]
# # 拿一张图出来看看
# sample_i = mnist_fasion[1][0]
# sample_l = mnist_fasion[1][1]
# print(sample_l)
# plt.figure(figsize=(2,2))
# plt.imshow(sample_i)
# pylab.show()
# # 测试paddle导入的mnist——fasion有无问题
# # for i in range(len(mnist_fasion)):
# #     sample = mnist_fasion[i]
# #     print(sample[0].size, sample[1])
# # 假数据处理
# fake_data = [i[0] for i in Fake_Data]
# fake_data_label = [int(i[1]) for i in Fake_Data]

# for i in range(10):
#     img = np.reshape(fake_data[i], (28, 28))
#     label = fluid.dygraph.to_variable(label.reshape([label.shape[0], 1]))
#     label_new = np.array(fake_data_label[int(i)]).astype('int64')
#     print('该图片的label的类型是：')
#     print(type(label_new))
#
#     plt.imshow(img)
#     pylab.show()


# This is a sample Python script.
# 设置图像变换操作(考虑下面三种图像变换操作)
transform = transforms.Compose([])                                    # 不做图像变换，灰度范围为[0,255]
# transform = transforms.Compose([transforms.ToTensor()])               # 把灰度范围从[0,255]变换到[0,1]
# transform = transforms.Compose([transforms.ToTensor(),                  # 先把灰度范围从[0,255]变换到[0,1]
#                                 transforms.Normalize((0.5), (0.5))])    # 再把[0,1]变换到[-1,1]
# 获取训练集
train_dataset = torchvision.datasets.MNIST(root = './data/',			# 下载目录
                               			   train = True,                # 训练集
                               			   transform = transform,		# 对数据集进行的Transforms操作
                               			   download = True)				
# 获取测试集
test_dataset = torchvision.datasets.MNIST(root = './data/',
                               			  train = False,                # 测试集
                               			  transform = transform,
                               			  download = True)
                               			  
# train_dataset.data和train_dataset.targets都是Tensor类型                               			  
# 训练集大小										    # 	输出结果：
print(train_dataset.data.size())				    # 	torch.Size([60000, 28, 28])
print(train_dataset.targets.size())				    # 	torch.Size([60000])
# 测试集大小
print(test_dataset.data.size())					    # 	torch.Size([10000, 28, 28])
print(test_dataset.targets.size())				    # 	torch.Size([10000])

def mnist_non_iid(dataset, major_class, unbalance_factor, num_clients):
    """
    将样本进行non-iid划分
    :param dataset:     数据集
    :param major_class:     每个客户端样本数量较多的类有几个
    :param unbalance_factor: 控制客户端分布的因子
    :param num_clients: 客户端数量
    :return:
    """
    non_iid_mnist = MNISTPartitioner(
        dataset.targets,
        num_clients=num_clients,
        partition='noniid-labeldir',
        dir_alpha=0.5,
        major_classes_num=major_class,
        verbose=False)

    num_classes = non_iid_mnist.num_classes
    col_names = [f"class{i}" for i in range(num_classes)]
    plt.rcParams['figure.facecolor'] = 'white'
    csv_file = f'mnist_non_iid_client_{num_clients}_factor_{unbalance_factor}.csv'
    partition_report(dataset.targets, non_iid_mnist.client_dict,
                     class_num=num_classes,
                     verbose=False, file=csv_file)

    noniid_major_label_part_df = pd.read_csv(csv_file, header=1)
    noniid_major_label_part_df = noniid_major_label_part_df.set_index('client')
    for col in col_names:
        noniid_major_label_part_df[col] = (
                noniid_major_label_part_df[col] * noniid_major_label_part_df['Amount']).astype(int)

    # select first 10 clients for bar plot
    noniid_major_label_part_df[col_names].plot.barh(stacked=True)
    # plt.tight_layout()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('sample num')
    png_file = f'mnist_non_iid_client_{num_clients}_factor_{unbalance_factor}.png'
    plt.savefig(png_file, dpi=400, bbox_inches='tight')
    return non_iid_mnist.client_dict

def mnist_unbalance(dataset, unbalance_factor, num_clients):
    """
    将样本进行不平很的划分，每个客户端样本数量不同，但是分布相同
    :param dataset:     数据集
    :param unbalance_factor: 不平衡因子，越大越平衡，取值100以上相当于IID，一般0.1 就极度不平很，划分比例可以图示上看
    :param num_clients:    客户端数量
    :return: 各个客户端的划分字典
    """
    unbalance_mnist = MNISTPartitioner(
        dataset.targets,
        num_clients=num_clients,
        partition='unbalance',
        dir_alpha=unbalance_factor,
        verbose=False)

    num_classes = unbalance_mnist.num_classes
    col_names = [f"class{i}" for i in range(num_classes)]
    plt.rcParams['figure.facecolor'] = 'white'
    csv_file = f'mnist_unbalance_client_{num_clients}_factor_{unbalance_factor}.csv'
    partition_report(dataset.targets, unbalance_mnist.client_dict,
                     class_num=num_classes,
                     verbose=False, file=csv_file)

    noniid_major_label_part_df = pd.read_csv(csv_file, header=1)
    noniid_major_label_part_df = noniid_major_label_part_df.set_index('client')
    for col in col_names:
        noniid_major_label_part_df[col] = (
                noniid_major_label_part_df[col] * noniid_major_label_part_df['Amount']).astype(int)

    # select first 10 clients for bar plot
    noniid_major_label_part_df[col_names].plot.barh(stacked=True)
    # plt.tight_layout()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('sample num')
    png_file =  f'mnist_unbalance_client_{num_clients}_factor_{unbalance_factor}.png'
    plt.savefig(png_file,dpi=400, bbox_inches='tight')
    return unbalance_mnist.client_dict
"""
模型水印部分
"""
class watermark:
    def __init__(self, b, w) -> None:
        self.b = b
        self.b_w = []
        self.X = []
        self.y = []
        self.w = w
        self.len_w = len(w)
        self.len_b = len(b)
        self.loss = 0

    # def sigmoid(self, x):
    #     z = np.exp(-x)
    #     sig = 1 / (1 + z)

        # return sig
    
    # sigmoid函数
    def sigmoid(self, x):
        z = math.exp(-x)
        sig = 1 / (1 + z)

        return sig

    # 水印矩阵
    def matrix_X(self):
        self.X = np.random.randn(self.len_b, self.len_w)

    # 更新参数
    def refreash_w(self, w):
        self.w = w

    # 由参数计算水印y
    def watermark_dot_product(self):
        y = []

        # print('dot')
        # print(len(self.X))
        # print(len(self.X[0]))
        # print(len(self.X[1]))
        # print(len(self.X[2]))
        # print(len(self.w))

        y = np.dot(self.X, self.w)

        for i in range(len(y)):
            y[i] = self.sigmoid(y[i])
        
        self.y = y

    # 水印损失函数
    def watermark_loss(self):
        b = paddle.to_tensor(self.b)
        y = paddle.to_tensor(self.y)

        self.loss = 5 * paddle.sum(nn.functional.binary_cross_entropy(y, b))

        # dis = []
        # dis_new = []
        # loss = 0
        # for i in range(self.len_b):
        #     dis.append(0)
        #     dis_new.append(0)
        
        # for i in range(self.len_b):
        #     # dis[i] = ((self.y[i] - self.b[i]) * (self.y[i] - self.b[i]))\
            
        #     dis_new[i] = - (self.b[i] * math.log(self.y[i]) - (1 - self.b[i]) * math.log(1 - self.y[i] + 0.0000000000001))

        # for i in range(self.len_b):
        #     # loss = dis[i] + loss
        #     loss = dis_new[i] + loss
        # self.loss = loss

    # 水印嵌入
    def enbeding_watermark(self):
        
        return self.loss   
    
    # 水印提取b
    def extracting_watermark(self):
        b_w = []

        b_w = np.dot(self.X, self.w)

        for i in range(len(b_w)):
            if b_w[i] >= 0:
                b_w[i] = 1
            else:
                b_w[i] = 0

        print('提取的水印为：')
        print(b_w)
        # print(self.b)
        # print(self.X)
        # print(self.y)
        # print(self.w)
        # print(self.loss)

        self.b_w = b_w




"""
初始化部分
1.每个节点得到一个模型（在k循环那块把每个人的模型拿出来，未加密前的）
2.基于自己的数据集跑GAN网络生成假数据（可以在每个用户划分生成数据的时候就生成好对应的假数据）
3.把假数据给别人，给多少除以总数，得到我的共享系数(这个合并到第二点，第二点生成假数据直接用len（），除以该节点原有的len（）就可以了)
4.别人做预测，我自己也做预测
5.别人的预测结果和我的做比对，别人预测对的结果/我给出的假数据集总数，得到cij
6.cij与门限作比较，得到i对j的置信度评估
"""

# 初始化部分宏定义(移动到289行)
# the_prediction =[[], [], [], [], [], [], [], [], [], []]

# 第四点+第五点
def share_belief(the_pre):

    # 计算当前节点对其余节点的置信度
    belief = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    belief = []
    for i in range(len(the_pre)):
        belief.append(0)
    confidence = the_pre
    cth = 0.1  # 瞎设的
    for i in range(len(the_pre)):
        # 判断门限
        if confidence[i] > cth:
            belief[i] = 1
        else:
            belief[i] = 0

    return belief, confidence

"""
P2P部分
1.保存每个客户端的模型（存成一个长条形的矩阵）
2.对Cij排序（从初始化部分多加一个返回参数confidence，里面保存的是归一化的Cij）
3.计算每个参与方的积分，Pi = λ * 参数总个数 * （参与方个数 - 1）
4.根据Cij的排序向别人买参数
5.将结果还原Tensor类型
"""
# 全局参数
# client比例
C = 0.1
# clients数量
K = 100

watermark_w = []

clients_score = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 参与方的积分（钱）
clients_score = []
for i in range(int(C * K)):
    clients_score.append(0)

clients_lambda = []  # 参与方的共享系数λ
for i in range(int(C * K)):
    # clients_lambda.append(np.random.randint(50)/100)
    clients_lambda.append(0.5)
print('-------------------共享系数-------------------')
print(clients_lambda)
recording_result = []


def model_save_wm(tensor_para):

    global watermark_w
    watermark_w = []

    for key, value in tensor_para.items():
    
        if key == 'conv1.weight':
            value1 = value.numpy().tolist()
            for i in range(32):
                for j in range(5):
                    for k in range(5):
                        watermark_w.append(value1[i][0][j][k])

        # elif key == 'conv1.bias':
        #     value2 = value.numpy().tolist()
        #     for i in range(32):
        #         watermark_w.append(value2[i])

        # elif key == 'conv2.weight':
        #     value3 = value.numpy().tolist()
        #     for i in range(64):
        #         for j in range(32):
        #             for k in range(5):
        #                 for n in range(5):
        #                     watermark_w.append(value3[i][j][k][n])

        # elif key == 'conv2.bias':
        #     value4 = value.numpy().tolist()
        #     for i in range(64):
        #         watermark_w.append(value4[i])

        # elif key == 'fc1.weight':
        #     value5 = value.numpy().tolist()
        #     for i in range(1024):
        #         for j in range(512):
        #             watermark_w.append(value5[i][j])

        # elif key == 'fc1.bias':
        #     value6 = value.numpy().tolist()
        #     for i in range(512):
        #         watermark_w.append(value6[i])

        # elif key == 'fc2.weight':
        #     value7 = value.numpy().tolist()
        #     for i in range(512):
        #         for j in range(10):
        #             watermark_w.append(value7[i][j])

        # elif key == 'fc2.bias':
        #     value8 = value.numpy().tolist()
        #     for i in range(10):
        #         watermark_w.append(value8[i])



# 1.保存客户端的模型
# 形成的矩阵维度 1 * 582026
# conv1.weight = 0~799
# conv1.bias = 800~831
# conv2.weight = 832~52031
# conv2.bias = 52032~52095
# fc1.weight = 52096~576383
# fc1.bias = 576384~576895
# fc2.weight = 576896~582015
# fc2.bias = 582016~582025
def model_save(tensor_para, seq_numbers, parameters_of_clients):

    for key, value in tensor_para.items():

        if key == 'conv1.weight':
            value1 = value.numpy().tolist()
            for i in range(32):
                for j in range(5):
                    for k in range(5):
                        parameters_of_clients[seq_numbers].append(value1[i][0][j][k])

        elif key == 'conv1.bias':
            value2 = value.numpy().tolist()
            for i in range(32):
                parameters_of_clients[seq_numbers].append(value2[i])

        elif key == 'conv2.weight':
            value3 = value.numpy().tolist()
            for i in range(64):
                for j in range(32):
                    for k in range(5):
                        for n in range(5):
                            parameters_of_clients[seq_numbers].append(value3[i][j][k][n])

        elif key == 'conv2.bias':
            value4 = value.numpy().tolist()
            for i in range(64):
                parameters_of_clients[seq_numbers].append(value4[i])

        elif key == 'fc1.weight':
            value5 = value.numpy().tolist()
            for i in range(1024):
                for j in range(512):
                    parameters_of_clients[seq_numbers].append(value5[i][j])

        elif key == 'fc1.bias':
            value6 = value.numpy().tolist()
            for i in range(512):
                parameters_of_clients[seq_numbers].append(value6[i])

        elif key == 'fc2.weight':
            value7 = value.numpy().tolist()
            for i in range(512):
                for j in range(10):
                    parameters_of_clients[seq_numbers].append(value7[i][j])

        elif key == 'fc2.bias':
            value8 = value.numpy().tolist()
            for i in range(10):
                parameters_of_clients[seq_numbers].append(value8[i])

    return parameters_of_clients

# 2.对Cij进行排序
def sort_Cij(share_confidence):
    share_confidence_copy = share_confidence  # 为了保存Cij对应的index
    share_confidence_2d = share_confidence  # 将index和Cij绑定到一个二维列表里，第一个是index，第二个是Cij
    share_confidence.sort()

    for i in range(len(share_confidence)):
        for j in range(len(share_confidence_copy)):
            if share_confidence[i] == share_confidence_copy[j]:  # 说明排序后的第i个Cij对应的index为j
                share_confidence_2d[i] = [j, share_confidence]  # 将index和Cij绑定到一起
                break

    return share_confidence_2d

# 3.计算Pi
def score_of_clients(lambda_of_client, num_clients):

    temp_3 = lambda_of_client * 582026 * (num_clients - 1)  # 计算每个客户端的积分

    return temp_3

# 4.根据Cij向别人买参数
def buying(buyer, seller, score_for_all, cij_of_buyer, lambda_of_seller, remark, parameters_of_clients):

    dij = 0

    if score_for_all[buyer] > 0:
        buyer_desire = cij_of_buyer * 582026  # 买家愿意花多少积分
        seller_desire = lambda_of_seller * 582026  # 卖家愿意给多少积分
        if buyer_desire >= seller_desire:
            dij = buyer_desire  # 花掉的积分

            if dij > score_for_all[buyer]:  # 判断是不是‘过度花钱’
                dij = score_for_all[buyer]

            # 先做积分更新
            score_for_all[buyer] = score_for_all[buyer] - dij
            score_for_all[seller] = score_for_all[seller] + dij

            # 再做参数更新
            dij = math.floor(dij)
            i = 0
            for i in range(dij):
                index_of_renew = i + remark  # remark用于记录该节点买的参数买到哪个了
                parameters_of_clients[buyer][index_of_renew] = parameters_of_clients[seller][index_of_renew]
            remark = i + remark + 1  # i从0开始，所以要加多1

        else:
            dij = seller_desire  # 花掉的积分

            if dij > score_for_all[buyer]:  # 判断是不是‘过度花钱’
                dij = score_for_all[buyer]

            # 先做积分更新
            score_for_all[buyer] = score_for_all[buyer] - dij
            score_for_all[seller] = score_for_all[seller] + dij

            # 再做参数更新
            dij = math.floor(dij)
            i = 0
            for i in range(dij):
                index_of_renew = i + remark  # remark用于记录该节点买的参数买到哪个了
                parameters_of_clients[buyer][index_of_renew] = parameters_of_clients[seller][index_of_renew]
            remark = i + remark + 1  # i从0开始，所以要加多1

    return remark, parameters_of_clients

# 5.将结果还原成Tensor类型

# 形成的矩阵维度 1 * 582026
# conv1.weight = 0~799
# conv1.bias = 800~831
# conv2.weight = 832~52031
# conv2.bias = 52032~52095
# fc1.weight = 52096~576383
# fc1.bias = 576384~576895
# fc2.weight = 576896~582015
# fc2.bias = 582016~582025
def rebuild_tensor(list_data, global_weights):
    remark = 0
    for key, value in global_weights.items():
        if key == 'conv1.weight':
            value1 = value.numpy().tolist()
            i = 0
            f = 0
            for z in value1:
                for x in z:
                    g = 0
                    for c in x:
                        h = 0
                        for v in c:
                            value1[f][0][g][h] = list_data[remark]
                            remark = remark + 1
                            i = i + 1
                            h = h + 1
                        g = g + 1
                f = f + 1
            temp = paddle.to_tensor(value1, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)
            global_weights[key] = temp

        elif key == 'conv1.bias':
            value1 = value.numpy().tolist()
            i = 0
            for z in value1:
                value1[i] = list_data[remark]
                remark = remark + 1
                i = i + 1
            temp = paddle.to_tensor(value1, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)
            global_weights[key] = temp

        elif key == 'conv2.weight':
            value1 = value.numpy().tolist()
            i = 0
            j = 0
            f = 0
            for z in value1:
                g = 0
                for x in z:
                    h = 0
                    for c in x:
                        k = 0
                        for v in c:
                            value1[f][g][h][k] = list_data[remark]
                            remark = remark + 1
                            j = j + 1
                            i = j // 6400
                            k = k + 1
                        h = h + 1
                    g = g + 1
                f = f + 1
            temp = paddle.to_tensor(value1, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)
            global_weights[key] = temp

        elif key == 'conv2.bias':

            value1 = value.numpy().tolist()
            i = 0
            for z in value1:
                value1[i] = list_data[remark]
                remark = remark + 1
                i = i + 1
            temp = paddle.to_tensor(value1, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)
            global_weights[key] = temp

        elif key == 'fc1.weight':
            value1 = value.numpy().tolist()
            i = 0
            j = 0
            f = 0
            for z in value1:
                g = 0
                for x in z:
                    value1[f][g] = list_data[remark]
                    remark = remark + 1
                    j = j + 1
                    i = j // 8192
                    g = g + 1
                f = f + 1
            temp = paddle.to_tensor(value1, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)
            global_weights[key] = temp

        elif key == 'fc1.bias':

            value1 = value.numpy().tolist()
            i = 0
            for z in value1:
                value1[i] = list_data[remark]
                remark = remark + 1
                i = i + 1
            temp = paddle.to_tensor(value1, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)
            global_weights[key] = temp

        elif key == 'fc2.weight':
            value1 = value.numpy().tolist()
            i = 0
            f = 0
            for z in value1:
                g = 0
                for x in z:
                    value1[f][g] = list_data[remark]
                    remark = remark + 1
                    i = i + 1
                    g = g + 1
                f = f + 1
            temp = paddle.to_tensor(value1, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)
            global_weights[key] = temp

        elif key == 'fc2.bias':
            value1 = value.numpy().tolist()
            i = 0
            for z in value1:
                value1[i] = list_data[remark]
                remark = remark + 1
                i = i + 1
            temp = paddle.to_tensor(value1, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)
            global_weights[key] = temp

    if remark == 582026:
        return global_weights
    else:
        print('输出结果缺失！')
        return False

# 划分测试集
def divide_tests(num_clients):
    global fake_data
    global fake_data_label
    each_client = 10000 / num_clients
    each_client = int(each_client)
    fake_label_trans = fake_data_label
    fake_data_trans = fake_data

    # 初始化各个节点的测试集
    clients_label_tests = []
    clients_data_tests = []
    for i in range(num_clients):
        clients_label_tests.append([])
        clients_data_tests.append([])
        for j in range(each_client):
            clients_label_tests[i].append('0')
            clients_data_tests[i].append(0)

    # 将测试样例均分到各个节点
    remark = 0
    for i in range(num_clients):
        for j in range(each_client):
            clients_label_tests[i][j] = fake_label_trans[remark]
            clients_data_tests[i][j] = fake_data_trans[remark]
            remark = remark + 1

    return clients_label_tests, clients_data_tests


# 定义模型
class CNN(fluid.dygraph.Layer):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = fluid.dygraph.Conv2D(1,32,5,act='relu')
        self.pool1 = fluid.dygraph.Pool2D(pool_size=2,pool_stride=2)
        self.conv2 = fluid.dygraph.Conv2D(32,64,5,act='relu')
        self.pool2 = fluid.dygraph.Pool2D(pool_size=2,pool_stride=2)
        self.fc1 = fluid.dygraph.Linear(1024,512,act='relu') # 全连接层
        self.fc2 = fluid.dygraph.Linear(512,10,act='softmax')

    # @paddle.jit.to_static(input_spec=[InputSpec(shape=[10, 64, 4, 4], dtype='float32')])
    # @paddle.jit.to_static
    def forward(self, input):
        x = fluid.layers.reshape(self.pool2(self.conv2(self.pool1(self.conv1(input)))), [-1, 1024])
        return self.fc2(self.fc1(x))

# 构造IID数据
# 均匀采样，分配到各个client的数据集都是IID且数量相等
def IID(dataset, clients):
  num_items_per_client = int(len(dataset)/clients)
  client_dict = {}
  image_idxs = [i for i in range(len(dataset))]  # 为每幅图像标号

  for i in range(clients):
    client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False))  # 为每个client随机选取数据
    image_idxs = list(set(image_idxs) - client_dict[i])  # 将已经选取过的数据去除

  return client_dict

# 构造nonIID数据
# 对数据不均匀划分，同时各个client上的数据分布和数量都不同
def NonIID(dataset, clients, total_shards, shards_size, num_shards_per_client):
    shard_idxs = [i for i in range(total_shards)]
    client_dict = {i: np.array([], dtype='int64') for i in range(clients)}
    ## client_dict={}
    idxs = np.arange(len(dataset))
    data_labels = Label

    label_idxs = np.vstack((idxs, data_labels))  # 将标签和数据ID堆叠
    label_idxs = label_idxs[:, label_idxs[1, :].argsort()]
    idxs = label_idxs[0, :]

    for i in range(clients):
        rand_set = set(np.random.choice(shard_idxs, num_shards_per_client, replace=False))
        shard_idxs = list(set(shard_idxs) - rand_set)

        for rand in rand_set:
            client_dict[i] = np.concatenate((client_dict[i], idxs[rand * shards_size:(rand + 1) * shards_size]),axis=0)  # 拼接

    return client_dict

# 定义数据读取器 按batch_size读取数据
def feeder(data,labell,idx,batch_size):
    def reader():
        images, labels = [], []
        for i in idx:
            image = np.array(data[int(i)]).astype('float32')
            label = np.array(labell[int(i)]).astype('int64')
            images.append(image)
            labels.append(label)
            if len(labels) == batch_size:
                yield np.array(images), np.array(labels)
                images, labels = [], []
    return reader

# 假数据预测
def result_trans(soft_result):
    result = paddle.argmax(soft_result, axis=-1)
    
    return result
# 假数据预测
def gan_predict(model, Data, Label, iid_dict_test, batchsize):
    
    final_output = []
    test_loader = feeder(Data, Label, iid_dict_test, batchsize)

    # 预测
    for image, label in test_loader():
        image = np.reshape(image,[-1,1,28,28])
        image = fluid.dygraph.to_variable(image) 
        label = fluid.dygraph.to_variable(label.reshape([label.shape[0], 1]))
        output = model.forward(image)

        output = result_trans(output)
        final_output.append(output)

    return final_output

# 计算众数
def mode(data):
    basket = []
    for i in range(len(data)):
        basket.append(0)

    for i in range(len(data)):
        for j in range(len(basket)):
            if (data[i] == j):
                basket[j] += 1

    return basket.index(max(basket))


# 计算信誉度
def cal_confidence(user_pre_label):
    test_label = []
    accuracy = []
    acc_user = []
    final_conf = []

    for i in range(len(user_pre_label)):
        accuracy.append([])
        for j in range(len(user_pre_label[i])):
            accuracy[i].append(0)

    # 先按照少数服从多数的原理求出标签，user_pre_label[k][i][j]
    for i in range(len(user_pre_label[0])):
        test_label_i = []
        for j in range(len(user_pre_label[0][i][0])):
            List = []
            for k in range(len(user_pre_label)):
                List.append(user_pre_label[k][i][0][j])

            List = paddle.to_tensor(List, dtype=paddle.int64)
            List = paddle.reshape(List, [-1])
            test_label_i.append(mode(List))

        test_label.append(test_label_i)

    # 计算精度
    for i in range(len(user_pre_label)):
        for j in range(len(user_pre_label[i])):
            if i != j:
                acc = 0
                size = 0
                for true, pred in zip(test_label[j], user_pre_label[i][j][0]):

                    size = size + 1
                    if true == pred:
                        acc += 1
                accuracy[i][j] = acc / size

    print(accuracy)

    # 压缩成每个客户端的信誉值
    for i in range(len(accuracy)):
        avg_acc = sum(accuracy[i]) / (len(accuracy) - 1)
        acc_user.append(avg_acc)

    # 计算归一化信誉值
    conf_sum = sum(acc_user)
    for i in range(len(acc_user)):
        final_conf.append(acc_user[i] / conf_sum)

    return final_conf

# 本地训练 SGD 输出模型参数
class ClientUpdate(object):
    def __init__(self,data,labell,batchSize,learning_rate,epochs,idxs):
        self.train_loader = feeder(data, labell, idxs, batchSize)
        self.learning_rate = learning_rate
        self.epochs = epochs

    ## paddle动态图训练
    def train(self, model):
        with fluid.dygraph.guard():
            optimizer = fluid.optimizer.SGDOptimizer(learning_rate=self.learning_rate, parameter_list=model.parameters())
            # optimizer = fluid.optimizer.AdamOptimizer(learning_rate=self.learning_rate, parameter_list=model.parameters())
            e_loss = []
            model.train()

            for epoch in range(1, self.epochs+1):
                train_loss = []
                for image, label in self.train_loader():
                    image = np.reshape(image, [-1, 1, 28, 28])
                    image = fluid.dygraph.to_variable(image)  # 将numpy数据转为飞桨动态图variable形式
                    label = fluid.dygraph.to_variable(label.reshape([label.shape[0], 1]))
                    output = model(image)  # 前向计算
                    loss = fluid.layers.mean(fluid.layers.cross_entropy(output, label))  # 交叉熵：实际输出（概率分布）与期望输出（概率分布）的距离

                    loss.backward()  # 损失反馈
                    optimizer.minimize(loss)  # 最小化损失
                    optimizer.clear_gradients()  # 清除梯度
                    train_loss.append(loss.numpy())
                t_loss = sum(train_loss)/len(train_loss)  # 针对batch，所有batch的均值
                e_loss.append(t_loss)
            total_loss = sum(e_loss)/len(e_loss)  # 针对epoch，所有epoch的均值

        return model.state_dict(), total_loss

# 服务器更新 加权平均合并新模型 传递给clients
def training(model, rounds, batch_size, lr, ds, L, data_dict, C, K, E, plt_title, plt_color, watermark_b):
    
    """
    模型水印嵌入
    """
    global_weights = model.state_dict()
    model_save_wm(global_weights)

    # 创建水印对象
    global watermark_w
    watermark_embeding = watermark(watermark_b, watermark_w)

    # 生成水印矩阵
    watermark_embeding.matrix_X()
    
    # print(model.state_dict())
    train_loss = []
    loss_new = []
    start_time = time.time()
    # clients与server之间通信
    for curr_round in range(1, rounds + 1):
        w, local_loss = [], []
        m = max(int(C * K), 1)  # 随机选取参与更新的clients

        # S_t = np.random.choice(range(K), m, replace=False)
        # S_t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        S_t = []
        for i in range(1, int(C * K) + 1):
            S_t.append(i)

        # 用户参数存储轮数
        round_of_save = 0

        # 划分测试集
        num_clients = 100
        clients_test_label, clients_test_data = divide_tests(num_clients)

        # 生成结果矩阵
        global recording_result
        for i in range(num_clients):
            recording_result.append([])
            for j in range(int(10000 / num_clients)):
                recording_result[i].append('0')

        clients_acc = []
        for i in range(num_clients):
            clients_acc.append(0)

        # 用户数更新
        which_client = 1
        the_prediction = []
        for i in range(int(C * K)):
            the_prediction.append([])
            for j in range(int(C * K)):
                the_prediction[i].append('0')

        mnist_data_test = np.load('F:/研零/zhao/test_mnist.npy')
        # 测试集
        Label_test = [int(i[0]) for i in mnist_data_test]
        Data_test = [i[1:] for i in mnist_data_test]
        img = np.reshape(Data[np.random.randint(0,10000)],(28,28))
        user_pre_label = []
        for i in range(int(C * K)):
            user_pre_label.append([])

        parameters_of_clients = []
        for i in range(int(C * K)):
            parameters_of_clients.append([])
        print(parameters_of_clients)

        for k in S_t:# 对每一个用户的w进行操作

            local_update = ClientUpdate(ds, L, batchSize=batch_size, learning_rate=lr, epochs=E, idxs=data_dict[k])
            print('the paticipaints is ', str(k))
            weights, loss = local_update.train(model)

            # 保存客户端模型

            parameters_of_clients = model_save(weights, k-1, parameters_of_clients)

            # 精度计算begin
            label_np = []
            data_np = []
            data_final = []
            label_final = []
            for i in range(num_clients):
                label_np.append([])
                data_np.append([])
                data_final.append([])
                label_final.append([])
                for j in range(int(10000 / num_clients)):
                    label_np[i].append(0)
                    data_np[i].append(0)
                    data_final[i].append(0)
                    label_final[i].append(0)

            # 进行预测

            iid_dict_test = IID(mnist_data_test, K)
            for i in range(int(C * K)):
                user_pre_label[k - 1].append(gan_predict(model, Data_test, Label_test, iid_dict_test[i], K)) 
        
            for j in range(len(clients_test_label[which_client])):
                # 将标签和图像转成np格式
                label_np[which_client][j] = np.array(clients_test_label[int(which_client)][int(j)]).astype('int64')
                data_np[which_client][j] = np.array(clients_test_data[int(which_client)][int(j)]).astype('float32')

                # 调整图片大小
                data_np[which_client][j] = np.reshape(data_np[which_client][j], [-1, 1, 28, 28])
                # print(data_np[which_client][j])

                # 将图片转成paddle合适的variable格式
                data_np[which_client][j] = fluid.dygraph.to_variable(data_np[which_client][j])

                # 预测
                a = model.forward(data_np[which_client][j])
                # print('测试集预测矩阵')
                # print(a)

                # 预测结果转换
                a = a.numpy()
                the_end = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                # 每张图的标签结果输出
                for m in range(10):  # 0~10的结果
                    if a[0][m] > 0.7:
                        # 输出预测结果
                        data_final[which_client][j] = the_end[m]  # 将第i个图的标签结果给到这个客户端的第i个图的位置

                # 与label比对
                if int(data_final[which_client][j]) == int(label_np[which_client][j]):
                    recording_result[which_client][j] = 1
                else:
                    recording_result[which_client][j] = 0
            # print('当前节点的预测结果为')
            # print(recording_result[which_client])

            # 计算精度
            cunt_acc = 0
            for j in range(len(recording_result[which_client])):
                if recording_result[which_client][j] == 1:
                    cunt_acc = cunt_acc + 1
            clients_acc[which_client] = cunt_acc / len(recording_result[which_client])
            # print('精度计算结果为')
            # print(clients_acc)
            which_client = which_client + 1

            # 计算贡献度里的精度部分
            clients_devote_acc = []
            for i in range(num_clients):
                clients_devote_acc.append(0)
            add_devote_acc = 0
            for i in range(num_clients):
                add_devote_acc = add_devote_acc + clients_acc[i]
            if add_devote_acc == 0:
                add_devote_acc = 1
            for i in range(num_clients):
                clients_devote_acc[i] = clients_acc[i] / add_devote_acc
            # 精度计算end

            # # 初始化begin
            # # paddle.save(global_weights, 'after_training')
            # # 每个客户端跑10张假图(由于这10张是固定的，可以假设是i客户端给剩余客户端的假数据，可以用来仿真i客户端的初始化)
            # for i in range(10):
            #     # 将图像转换成np格式且符合大小要求
            #     a = np.array(fake_data[int(i)]).astype('float32')
            #     fake_img = np.reshape(a, [-1, 1, 28, 28])

            #     # 转换数据格式
            #     fake_img = fluid.dygraph.to_variable(fake_img)  # 将numpy数据转为飞桨动态图variable形式
            #     f_p = model.forward(fake_img)

            #     # 预测结果转换
            #     the_end = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            #     f_p_np = f_p.numpy()
            #     # print('预测结果原矩阵')
            #     # print(f_p)

            #     # 每张图的标签结果输出
            #     for j in range(10):  # 0~10的结果
            #         if f_p_np[0][j] > 0.6:
            #             # 输出预测结果
            #             the_prediction[which_client][i] = the_end[j]  # 将第i个图的标签结果给到这个客户端的第i个图的位置
            #             # print('预测结果是:')
            #             # print(the_prediction[which_client][i])
            #             # break
            # which_client = which_client + 1
            # # print('10个节点完成假数据预测')
            # # print(the_prediction)
            # # print('num_clients')
            # # print(which_client)
            # print(the_prediction)

        # user_pre_label[i][j][0][k]才到每一个数字
        con = cal_confidence(user_pre_label)
        
        the_prediction = con
        # 计算置信度
        share_confidence = []
        share_threshold, share_confidence = share_belief(the_prediction)
        print('归一化置信度')
        print(share_confidence)
        
        print('置信度是否满足门限')
        print(share_threshold)
        # 初始化end

        # 对Cij排序
        sort_after_cij = sort_Cij(share_confidence)

        # 产生贡献度的λ部分
        global clients_lambda
        clients_devote_lambda = []

        for i in range(int(C * K)):
            clients_devote_lambda.append(0)

        add_devote_lambda = 0
        for i in range(int(C * K)):
            add_devote_lambda = clients_lambda[i] + add_devote_lambda

        for i in range(int(C * K)):
            clients_devote_lambda[i] = clients_lambda[i] / add_devote_lambda

        # 计算积分
        global clients_score
        for i in range(int(C * K)):
            clients_score[i] = score_of_clients(clients_lambda[0], i)
            # clients_score.append(score_of_clients(clients_lambda, i))

        # 向别人买参数并更新
        remark = 0
        for i in range(1, int(C * K)):
            remark, parameters_of_clients = buying(0, i, clients_score, sort_after_cij[i][1], clients_lambda, remark, parameters_of_clients)

        # print('保存的参数')
        # print(len(parameters_of_clients[0]))
            
        # weights = rebuild_tensor(parameters_of_clients[0], weights)  # 将i节点（对应parameters_of_clients[0]）的参数转回原样子
        
        # print('P2P得到的参数结果')
        # print(weights)
        local_loss.append(loss)
        # # 绘图
        # clients_devote = []
        # for i in range(10):
        #     clients_devote.append(0)
        # for i in range(10):
        #     clients_devote[i] = clients_devote_lambda[i] + clients_devote_acc[i]

        # fig, ax = plt.subplots()
        # x_axis = np.array(clients_devote)
        # y_axis = np.array(clients_acc)
        # ax.plot(x_axis, y_axis, 'tab:' + "green")
        # t = 'devoted_and_acc(random_and_iid)'
        # kkk = str(t) + 'sss.csv'
        # kkk3 = 'new.csv'
        # s12 = pd.DataFrame(data=clients_acc)
        # s12.to_csv(kkk, encoding='utf-8')
        # s13 = pd.DataFrame(data=clients_devote)
        # s13.to_csv(kkk3, encoding='utf-8')
        # ax.set(xlabel='Contribution', ylabel='accuracy', title='relationship between contribution and accuracy')
        # ax.grid()
        # fig.savefig('贡献度与精度关系图' + '.jpg', format='jpg')


        # 加密过程的中间参数
        # add_list = []
        # remark_confidence = 0
        # for k in S_t:  # 对每一个用户的w进行操作
        #     encrypted_list = []
        #     start = time.time()
        #     for key, value in weights.items():
        #         if key == 'conv1.weight' :
        #             encrypted_temp_list = []
        #             value1 = value.numpy().tolist()
        #             ssss = ts.ckks_tensor(context,value1)
        #             for i in range(32):
        #                 for j in range(5):
        #                     for k in range(5):
        #                         # value1[i][0][j][k] = public_key.encrypt(value1[i][0][j][k])
        #                         encrypted_temp_list.append(value1[i][0][j][k])

        #                         # arr_gen1[i][0][j][k] = HE.encryptFrac(value123[i][0][j][k])
        #                         # print(i,j,k)
        #             # print(encrypted_temp_list)
        #             a = ts.ckks_vector(context,encrypted_temp_list)
        #             encrypted_list.append(a)
        #             # print(a.decrypt())
        #         elif key == 'conv1.bias':
        #             # value123 = value.numpy()
        #             # arr_gen1 = np.empty(value123.shape, dtype=PyCtxt)
        #             value2 = value.numpy().tolist()
        #             encrypted_temp_list = []
        #             for i in range(32):
        #                 encrypted_temp_list.append(value2[i])
        #                 # value2[i] = public_key.encrypt(value2[i])
        #                 # print(i)
        #             # print(encrypted_temp_list)
        #             a = ts.ckks_vector(context,encrypted_temp_list)
        #             encrypted_list.append(a)
        #             # print(a.decrypt())
        #             # encrypted_list.append(value2)
        #         elif key == 'conv2.weight':
        #             # value123 = value.numpy()
        #             # arr_gen1 = np.empty(value123.shape, dtype=PyCtxt)
        #             value3 = value.numpy().tolist()
        #             encrypted_temp_list = []
        #             for i in range(64):
        #                 for j in range(32):
        #                     for k in range(5):
        #                         for l in range(5):
        #                             # arr_gen1[i][j][k][l] = HE.encryptFrac(value123[i][j][k][l])
        #                             # value3[i][j][k][l] = public_key.encrypt(value3[i][j][k][l])
        #                             encrypted_temp_list.append(value3[i][j][k][l])
        #                             # print(i, j, k, l)
        #                 if (i % 8) == 7:
        #                     # print(encrypted_temp_list)
        #                     a = ts.ckks_vector(context, encrypted_temp_list)
        #                     encrypted_list.append(a)
        #                     # print(a.decrypt())
        #                     encrypted_temp_list = []
        #             # encrypted_list.append(value3)
        #         elif key == 'conv2.bias':
        #             encrypted_temp_list = []
        #             # value123 = value.numpy()
        #             # arr_gen1 = np.empty(value123.shape, dtype=PyCtxt)
        #             value4 = value.numpy().tolist()
        #             for i in range(64):
        #                 # arr_gen1[i] = HE.encryptFrac(value123[i])
        #                 # value4[i] = public_key.encrypt(value4[i])
        #                 encrypted_temp_list.append(value4[i])
        #                 # print(i)
        #             # encrypted_list.append(value4)
        #             # print(encrypted_temp_list)
        #             a = ts.ckks_vector(context, encrypted_temp_list)
        #             encrypted_list.append(a)
        #             # print(a.decrypt())
        #         elif key == 'fc1.weight':
        #             # value123 = value.numpy()
        #             # arr_gen1 = np.empty(value123.shape, dtype=PyCtxt)
        #             value5 = value.numpy().tolist()
        #             encrypted_temp_list = []
        #             for i in range(1024):
        #                 for j in range(512):
        #                     # arr_gen1[i][j] = HE.encryptFrac(value123[i][j])
        #                     # value5[i][j] = public_key.encrypt(value5[i][j])
        #                     encrypted_temp_list.append(value5[i][j])
        #                     # print(i, j)
        #                 if (i % 16) == 15:
        #                     # print(encrypted_temp_list)
        #                     a = ts.ckks_vector(context, encrypted_temp_list)
        #                     encrypted_list.append(a)
        #                     # print(a.decrypt())
        #                     encrypted_temp_list = []
        #             # encrypted_list.append(value5)
        #         elif key == 'fc1.bias':
        #             # value123 = value.numpy()
        #             # arr_gen1 = np.empty(value123.shape, dtype=PyCtxt)
        #             value6 = value.numpy().tolist()
        #             encrypted_temp_list = []
        #             for i in range(512):
        #                 # arr_gen1[i] = HE.encryptFrac(value123[i])
        #                 # value6[i] = public_key.encrypt(value6[i])
        #                 encrypted_temp_list.append(value6[i])
        #                 # print(i)
        #             # print(encrypted_temp_list)
        #             a = ts.ckks_vector(context, encrypted_temp_list)
        #             encrypted_list.append(a)
        #             # print(a.decrypt())
        #             # encrypted_list.append(value6)
        #         elif key == 'fc2.weight':
        #             # value123 = value.numpy()
        #             # arr_gen1 = np.empty(value123.shape, dtype=PyCtxt)
        #             value7 = value.numpy().tolist()
        #             encrypted_temp_list = []
        #             for i in range(512):
        #                 for j in range(10):
        #                     # arr_gen1[i][j] = HE.encryptFrac(value123[i][j])
        #                     # value7[i][j] = public_key.encrypt(value7[i][j])
        #                     encrypted_temp_list.append(value7[i][j])
        #                     # print(i, j)
        #             # print(encrypted_temp_list)
        #             a = ts.ckks_vector(context, encrypted_temp_list)
        #             encrypted_list.append(a)
        #             # print(a.decrypt())
        #             # encrypted_list.append(value7)
        #         elif key == 'fc2.bias':
        #             # value123 = value.numpy()
        #             # arr_gen1 = np.empty(value123.shape, dtype=PyCtxt)
        #             value8 = value.numpy().tolist()
        #             encrypted_temp_list = []
        #             for i in range(10):
        #                 # arr_gen1[i] = HE.encryptFrac(value123[i])
        #                 # value8[i] = public_key.encrypt(value8[i])
        #                 encrypted_temp_list.append(value8[i])
        #                 # print(i)
        #             # print(encrypted_temp_list)
        #             a = ts.ckks_vector(context, encrypted_temp_list)
        #             encrypted_list.append(a)
        #             # print(a.decrypt())
        #             # encrypted_list.append(value8)
        #         # print(key, value)
        #         # value1 = value.numpy()
        #         # out = value1.shape
        #         # value2 = value1.tolist()
        #         # sdad = len(out)
        #         # print(sdad)
        #         # value2[0][0][0][0] =  public_key.encrypt(value2[0][0][0][0])
        #         # print(private_key.decrypt(value2[0][0][0][0]))
        #         # value2[0][0][0][0] = private_key.decrypt(value2[0][0][0][0])
        #         # temp = paddle.to_tensor(value2, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)

        #     end = time.time()
        #     print(f"encrypted time: {end - start}")
        #     if len(add_list):
        #         start = time.time()
                
        #         # 置信度引入
        #         for res in encrypted_list:
        #             res = res.mul(share_confidence[remark_confidence])

        #         for i in range(78):
        #             add_list[i] = add_list[i] + encrypted_list[i]
        #         end = time.time()
        #         print(f"once add time: {end - start}")
        #     else:
        #         # add_list = copy.deepcopy(encrypted_list)
        #         add_list = encrypted_list
        #     #计算完共78个密文，下标为0-77：
        #     # 'conv1.weight' -> 0 -> 800 * 1
        #     # 'conv1.bias' -> 1 -> 32 * 1
        #     # 'conv2.weight' -> 2-9 -> 6400 * 8
        #     # 'conv2.bias' -> 10 ->64 * 1
        #     # 'fc1.weight' -> 11-74  -> 8192 * 64
        #     # 'fc1.bias' -> 75  -> 512 * 1
        #     # 'fc2.weight' -> 76 -> 5120 * 1
        #     # 'fc2.bias' -> 77 -> 10 * 1
        #     local_loss.append(loss)
        #     remark_confidence = remark_confidence + 1
        #     # num.append(len(data_dict[k]))

        # reslut = []
        # start = time.time()
        # for res in add_list:
        #     res = res.mul(0.1)
        #     temp = res.decrypt()
        #     reslut.append(temp)  # 解密后的参数（还是一个二维矩阵，第一维有78）
        # end = time.time()
        # print(f"decrypt time: {end - start}")
        # i = 0
        # j = 0
        # for first in reslut:
        #     j = 0
        #     for second in first:
        #         # second = second / 10
        #         reslut[i][j] = format(second,'.8f')
        #         j = j + 1
        #     i = i + 1

        # for key, value in global_weights.items():
        #     start = time.time()
        #     if key == 'conv1.weight':
        #         value1 = value.numpy().tolist()
        #         i = 0
        #         f = 0
        #         for z in value1:
        #             for x in z:
        #                 g = 0
        #                 for c in x:
        #                     h = 0
        #                     for v in c:
        #                         value1[f][0][g][h] = reslut[0][i]
        #                         i = i+1
        #                         h = h+1
        #                     g = g + 1
        #             f = f + 1
        #         temp = paddle.to_tensor(value1, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)
        #         global_weights[key] = temp
        #     elif key == 'conv1.bias':
        #         value1 = value.numpy().tolist()
        #         i = 0
        #         for z in value1:
        #             value1[i] = reslut[1][i]
        #             i = i + 1
        #         temp = paddle.to_tensor(value1, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)
        #         global_weights[key] = temp

        #     elif key == 'conv2.weight':
        #         value1 = value.numpy().tolist()
        #         i = 0
        #         j = 0
        #         f = 0
        #         for z in value1:
        #             g = 0
        #             for x in z:
        #                 h = 0
        #                 for c in x:
        #                     k = 0
        #                     for v in c:
        #                         value1[f][g][h][k] = reslut[2+i][j % 6400]
        #                         j = j + 1
        #                         i = j // 6400
        #                         k = k + 1
        #                     h = h+1
        #                 g = g+1
        #             f = f+1

        #         temp = paddle.to_tensor(value1, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)
        #         global_weights[key] = temp
        #     elif key == 'conv2.bias':

        #         value1 = value.numpy().tolist()
        #         i = 0
        #         for z in value1:
        #             value1[i] = reslut[10][i]
        #             i = i + 1
        #         temp = paddle.to_tensor(value1, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)
        #         global_weights[key] = temp

        #     elif key == 'fc1.weight':
        #         value1 = value.numpy().tolist()
        #         i = 0
        #         j = 0
        #         f = 0
        #         for z in value1:
        #             g = 0
        #             for x in z:
        #                 value1[f][g] = reslut[11+i][j % 8192]
        #                 j = j + 1
        #                 i = j // 8192
        #                 g = g+1
        #             f = f+1
        #         temp = paddle.to_tensor(value1, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)
        #         global_weights[key] = temp

        #     elif key == 'fc1.bias':

        #         value1 = value.numpy().tolist()
        #         i = 0
        #         for z in value1:
        #             value1[i]= reslut[75][i]
        #             i = i + 1
        #         temp = paddle.to_tensor(value1, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)
        #         global_weights[key] = temp

        #     elif key == 'fc2.weight':
        #         value1 = value.numpy().tolist()
        #         i = 0
        #         f = 0
        #         for z in value1:
        #             g = 0
        #             for x in z:
        #                 value1[f][g] = reslut[76][i]
        #                 i = i + 1
        #                 g = g + 1
        #             f = f + 1
        #         temp = paddle.to_tensor(value1, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)
        #         global_weights[key] = temp

        #     elif key == 'fc2.bias':
        #         value1 = value.numpy().tolist()
        #         i = 0
        #         for z in value1:
        #             value1[i] = reslut[77][i]
        #             i = i + 1
        #         temp = paddle.to_tensor(value1, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)
        #         global_weights[key] = temp
        #         # value1 = value.numpy()
        #         # out = value1.shape
        #         # value2 = value1.tolist()
        #         # sdad = len(out)
        #         # print(sdad)
        #         # value2[0][0][0][0] =  public_key.encrypt(value2[0][0][0][0])
        #         # print(private_key.decrypt(value2[0][0][0][0]))
        #         # value2[0][0][0][0] = private_key.decrypt(value2[0][0][0][0])

        # # 更新global weights
        # weights_avg = w[0]
        # for k in weights_avg.keys():
        #     for i in range(1, len(w)):
        #         # weights_avg[k] += (num[i]/sum(num))*w[i][k]
        #         weights_avg[k] = weights_avg[k] + w[i][k]
        #     weights_avg[k] = weights_avg[k] / len(w)
        # global_weights = weights_avg

        # 模型加载最新的参数
        global_weights = weights
        model.load_dict(global_weights)

        params = []
        params = model_save_wm(global_weights)
        params = watermark_w

        csv_weights = 'F:/研零/Zhao/2.17_test/weights.csv'
        s14 = pd.DataFrame(data = params)
        s14.to_csv(csv_weights, encoding = 'utf-8')
        # print(params)

        # # 水印重新加载w
        # global_weights = model.state_dict()
        # model_save_wm(global_weights)
        # watermark_embeding.refreash_w(watermark_w)

        loss_avg = sum(local_loss) / len(local_loss)
        #if curr_round % 10 == 0:
        print('Round: {}... \tAverage Loss: {}'.format(curr_round, np.round(loss_avg, 3)))
        train_loss.append(loss_avg)

        # 保存每轮的loss
        # loss_np = loss_avg.numpy().tolist()
        loss_new.append(loss_avg)
        # print(loss_new)

        # 进行预测
        final_acc = []
        Label_test = [int(i[0]) for i in mnist_data_test]
        Data_test = [i[1:] for i in mnist_data_test]
        test_data_dict = mnist_unbalance(test_dataset, 20, 100)
        max_acc = 0
        max_acc_i = 0
        output_standard = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        final_test_loss = 0
        final_test_acc = 0
        # if rounds == 10:
        for i in range(10000):
            image = np.array(Data_test[int(i)]).astype('float32')
            image = np.reshape(image, [-1, 1, 28, 28])
            image = fluid.dygraph.to_variable(image)  # 将numpy数据转为飞桨动态图variable形式
            output = model.forward(image)  # 前向计算
            max_acc = output[0][0]
            max_acc = np.array(max_acc).astype('int64')

            for j in range(10):
                if output[0][j] > max_acc:
                    max_acc = output[0][j]
                    max_acc_i = output_standard[j]

            image_output = np.array(max_acc_i).astype('int64')
            label = np.array(Label_test[int(i)]).astype('int64')

            if label == image_output:
                final_test_acc = final_test_acc + 1
            else:
                final_test_loss = final_test_loss + 1
                    # print('输出')
                    # print(image_output)
                    # print(label)
            
        # 精度、错误率计算
        final_test_loss = final_test_loss / 10000
        final_test_acc = final_test_acc / 10000
        final_acc.append(final_test_acc)
                
        print('测试精度为')
        print(final_test_acc)

    end_time = time.time()

    # # 水印提取
    # watermark_embeding.extracting_watermark()
    
    csv_final_test_acc = 'F:/研零/Zhao/2.17_test/final_test_acc.csv'
    csv_final_loss = 'F:/研零/Zhao/2.17_test/final_loss.csv'

    s14 = pd.DataFrame(data = final_acc)
    s14.to_csv(csv_final_test_acc, encoding = 'utf-8')
    s15 = pd.DataFrame(data = loss_new)
    s15.to_csv(csv_final_loss, encoding = 'utf-8')

    fig, ax = plt.subplots()
    x_axis = np.arange(1, rounds + 1)
    y_axis = np.array(train_loss)
    ax.plot(x_axis, y_axis, 'tab:' + plt_color)
    t = time.time()
    kkk = str(t) + 'sss.csv'
    s12 = pd.DataFrame(data=train_loss)
    s12.to_csv(kkk, encoding='utf-8')
    ax.set(xlabel='Number of Rounds', ylabel='Train Loss', title=plt_title)
    ax.grid()
    fig.savefig(plt_title + '.jpg', format='jpg')
    print("Training Done!")
    print("Total time taken to Train: {}".format(end_time - start_time))
    
    

    return model.state_dict()

# # 开始训练

# IID数据训练
# 通信轮数
rounds = 10
# client比例
C = 0.1
# clients数量
K = 100
# 每次通信在本地训练的epoch
E = 20
# batch size
batch_size = 10
# 学习率
lr=0.001
# 数据切分
iid_dict = IID(mnist_data_train, 100)
# iid_dict = IID(mnist_fasion, 100)
#导入模型
mnist_cnn = CNN()
# 生成水印
watermark_b = np.ones(800, dtype="double")
# print('The embedding watermark is:', str(watermark_b))

mnist_cnn_iid_trained = training(mnist_cnn, rounds, batch_size, lr, Data, Label, iid_dict, C, K, E, "MNIST CNN on IID Dataset", "orange", watermark_b)


# data_dict = NonIID(mnist_data_train, 100, 200, 300, 2)
# mnist_cnn_non_iid_trained = training(mnist_cnn, rounds, batch_size, lr, Data,Label, data_dict, C, K, E, "MNIST CNN on Non-IID Dataset 3", "green")
# exec_strategy = fluid.ExecutionStrategy()
# exec_strategy.num_threads = 8
#
# # 配置构图策略，对于CPU训练而言，应该使用Reduce模式进行训练
# build_strategy = fluid.BuildStrategy()
# if int(os.getenv("CPU_NUM")) > 1:
#     build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
# 
# pe = fluid.ParallelExecutor(
#     use_cuda=False,
#     main_program=training(mnist_cnn, rounds, batch_size, lr, Data, Label, iid_dict, C, K, E, "MNIST CNN on IID Dataset", "orange"),
#     build_strategy=build_strategy,
#     exec_strategy=exec_strategy)



# # non-IID数据训练
# # 通信轮数
# rounds = 10
# # client比例
# C = 0.3
# # clients数量
# K = 100
# # 每次通信在本地训练的epoch
# E = 5
# # batch size
# batch_size = 10
# # 学习率
# lr=0.001
# # 数据切分
# # data_dict = NonIID(mnist_data_train, 100, 200, 300, 2)
# # data_dict = mnist_non_iid(train_dataset, 2, 20, 100) # 数据数量和分布都不同
# data_dict = mnist_unbalance(train_dataset, 20, 100) # 数据数量不同，但是分布相同
# # data_dict = NonIID(mnist_fasion, 100, 200, 300, 2)
# # 导入模型
# mnist_cnn = CNN()

# # 生成水印
# watermark_b = [1, 0, 1]

# mnist_cnn_non_iid_trained = training(mnist_cnn, rounds, batch_size, lr, Data, Label, data_dict, C, K, E, "MNIST CNN on Non-IID Dataset 3", "green", watermark_b)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/



