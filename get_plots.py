from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

SEED = 1234
import numpy as np
import math
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

import time
np.random.seed(SEED)
import keras
from keras import backend as K
import tensorflow as tf
import os, shutil, scipy.io
from model import *

# 基本设置
tf.set_random_seed(SEED)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
K.set_learning_phase(0)

caseNo = 118 # 总线数
weight_4_mag = 100
weight_4_ang = 1# 2*math.pi/360

# 数据导入
psse_data = scipy.io.loadmat('dist2_118FASE_data.mat')
matlab_predicts = scipy.io.loadmat('118_bus_GStest_err.mat')
gs_predicts = matlab_predicts['GS_voltage']
gs_predicts = np.transpose(gs_predicts)
print(gs_predicts.shape)
data_x = psse_data['inputs']
data_y = psse_data['labels']

# scale the mags,
data_y[0:caseNo,:] = weight_4_mag*data_y[0:caseNo,:]
data_y[caseNo:,:] = weight_4_ang*data_y[caseNo:,:]
# 数据分成 training 80%, test 20%
split_train = int(0.8*psse_data['inputs'].shape[1])
split_val = psse_data['inputs'].shape[1] - split_train # int(0.25*psse_data['inputs'].shape[1])
train_x = np.transpose(data_x[:, :split_train])
train_y = np.transpose(data_y[:, :split_train])
val_x   = np.transpose(data_x[:, split_train:split_train+split_val])
val_y   = np.transpose(data_y[:, split_train:split_train+split_val])
test_x  = np.transpose(data_x[:, split_train+split_val:])
test_y  = np.transpose(data_y[:, split_train+split_val:])

print(train_x.shape)
print(val_x.shape)

#训练模型
input_shape = (train_x.shape[1],)
train_xconv = np.expand_dims(train_x, axis=-1)
val_xconv = np.expand_dims(val_x, axis=-1)
input_shapeconv = (train_xconv.shape[1],train_xconv.shape[2])

start_time = time.time()
#分别加载 lav 6FNN 8FNN的模型参数
lav_weights = 'model_logs/118_lav_PSSE_epoch_200.h5'
nn1_6H_weights = 'model_logs/118_nn1_6H_PSSE_epoch_200.h5'
nn1_8H_weights = 'model_logs/118_nn1_8H_PSSE_epoch_200.h5'
nn1_conv1_weights = 'model_logs/118_nn1_conv_epoch_100.h5'

#分别设置lav 6FNN 8FNN的网络
lav_model =  lav_psse(input_shape, train_y.shape[1], weights=lav_weights)
nn1_6H_model =  nn1_psse(input_shape, train_y.shape[1], weights=nn1_6H_weights)
nn1_8H_model =  nn1_8H_psse(input_shape, train_y.shape[1], weights=nn1_8H_weights)
conv_model = nn1_conv(input_shapeconv, train_y.shape[1], weights=nn1_conv1_weights)

#train_lav_predicts = lav_model.predict(train_x)
#开始预测4种模型的效果
lav_predicts = lav_model.predict(val_x)
NN6H_predicts = nn1_6H_model.predict(val_x)
NN8H_predicts = nn1_8H_model.predict(val_x)
conv_predicts = conv_model.predict(val_xconv)

print("--- %s seconds ---" % (time.time() - start_time))




val_predic = lav_predicts
test_no = 3706 # 测试例子数
# 作者定义的距离度量的方法 也就是评价指标之类的
def rmse(val_predic, val_y, voltage_distance = np.zeros((test_no,caseNo)), voltage_norm = np.zeros((test_no,1))):
    for i in range(test_no):
        for j in range(caseNo):
            predic_r, predic_i = (1/weight_4_mag)* val_predic[i, j]*math.cos(val_predic[i, j+caseNo]*2*math.pi/360), (1/weight_4_mag)*val_predic[i,j]*math.sin(val_predic[i, j+caseNo]*2*math.pi/360)
            val_r, val_i = (1/weight_4_mag)*val_y[i,j]*math.cos(val_y[i,j+caseNo]*2*math.pi/360), (1/weight_4_mag)*val_y[i][j]*math.sin(val_y[i][j+caseNo]*2*math.pi/360)
            voltage_distance[i,j] = (predic_r-val_r)**2 + (predic_i-val_i)**2
            #print(i, j, val_predic[i, j], val_predic[i, j+caseNo], val_y[i,j], val_y[i,j+caseNo])
        voltage_norm[i,] = (1/caseNo)*np.sqrt(np.sum(voltage_distance[i,:]))
    return np.mean(voltage_norm) *100

#分别计算lav 6FNN 8FNN的评价指标 有他的距离度量函数 计算预测结果和标签之间的距离
print("\n distance from the true states in terms of \|\|_2: %.4f%%" % rmse(lav_predicts, val_y))
print("\n distance from the true states in terms of \|\|_2: %.4f%%" % rmse(NN6H_predicts, val_y))
print("\n distance from the true states in terms of \|\|_2: %.4f%%" % rmse(NN8H_predicts, val_y))
print("\n distance from the true states in terms of \|\|_2: %.4f%%" % rmse(conv_predicts, val_y))


#把结果画在图上 坐标定义
fig_num = 0
plt.figure(fig_num)
fig_num += 1

busNo = caseNo
start_bus, end_bus = 0, 50
timeslot = 999
plt.subplot(2,1,1)
plt.plot(range(start_bus+1, end_bus+1), (1/weight_4_mag)* val_y[timeslot][start_bus: end_bus], color='k', marker='D') # here we have start_bus+1 is due to that in our paper, index starts 1 while python starts 0
plt.hold
plt.plot(range(start_bus+1, end_bus+1), (1/weight_4_mag)* lav_predicts[timeslot][start_bus: end_bus], linestyle='--',  color='r', marker='*')
plt.hold
plt.plot(range(start_bus+1, end_bus+1), (1/weight_4_mag)* NN6H_predicts[timeslot][start_bus: end_bus],linestyle='-.', color='c')
plt.hold
plt.plot(range(start_bus+1, end_bus+1), (1/weight_4_mag)* NN8H_predicts[timeslot][start_bus: end_bus] , linestyle='-.', color='g')
plt.hold
plt.plot(range(start_bus+1, end_bus+1), (1/weight_4_mag)* conv_predicts[timeslot][start_bus: end_bus],linestyle=':', color='m')

plt.ylabel('电压幅值 （伏）')
plt.title(str(timeslot+1) + '测试例子的电压')
axes = plt.gca()
axes.set_xticks(list(range(start_bus+1, end_bus+1, 10)))

plt.subplot(2,1,2)
plt.plot(range(start_bus+1, end_bus+1), val_y[timeslot][start_bus+ busNo: end_bus+ busNo], color='k',  marker='D')
plt.hold
plt.plot(range(start_bus+1, end_bus+1), lav_predicts[timeslot][start_bus+ busNo: end_bus+ busNo], linestyle= '--', color='r',  marker='*')
plt.hold
plt.plot(range(start_bus+1, end_bus+1), NN6H_predicts[timeslot][start_bus+ busNo: end_bus+ busNo], linestyle= '-.', color='c')
plt.hold
plt.plot(range(start_bus+1, end_bus+1), NN8H_predicts[timeslot][start_bus+ busNo: end_bus+ busNo], color='g')
plt.hold
plt.plot(range(start_bus+1, end_bus+1), conv_predicts[timeslot][start_bus+ busNo: end_bus+ busNo],linestyle=':', color='m')
plt.ylabel('电压幅值 （伏）')
plt.xlabel('总线序号')
plt.legend( ['真实值', '近似线性网络', '6层 FNN', '8层 FNN','LSTM'], loc=3, prop={'size': 8})
# plt.show()

axes = plt.gca()
axes.set_xticks(list(range(start_bus+1, end_bus+1, 10)))

#定义的plt bus 方法
def plt_bus(busShow = 100, x_step = 5, start = 999, end = 1050, fig_num = 0):
    plt.figure(fig_num)
# this is for a certain bus across different time slots
    ax1 = plt.subplot(2,1,1)
    plt.plot(range(start+1, end+1), (1/weight_4_mag)* val_y[start: end, busShow], color='k',  marker='oD')
    plt.hold
    plt.plot(range(start+1, end+1), (1/weight_4_mag)* lav_predicts[start: end, busShow], linestyle= '--',  color='r',  marker='*')
    plt.hold
    plt.plot(range(start+1, end+1), (1/weight_4_mag)* NN6H_predicts[start: end, busShow], linestyle= '-.', color='c')
    plt.hold
    plt.plot(range(start + 1, end + 1), (1 / weight_4_mag) * NN8H_predicts[start: end, busShow], color='g')
    plt.hold
    plt.plot(range(start + 1, end + 1), (1 / weight_4_mag) * conv_predicts[start: end, busShow], linestyle=':',color='m')
    #plt.hold
    #plt.plot(range(start+1, end+1), gs_predicts[start: end, busShow], color='c',linestyle= ':')

    ax1.set_xticks(list(range(start+1, end+1, x_step)))
    plt.ylabel('电压幅值 （伏）')
    plt.title(str(busShow) + '总线的电压')

    plt.subplot(2,1,2)
    plt.plot(range(start+1, end+1), val_y[start: end, busShow + busNo], color='k',  marker='D')
    plt.hold
    plt.plot(range(start+1, end+1), lav_predicts[start: end, busShow + busNo], linestyle= '--', color='r',  marker='*')
    plt.hold
    plt.plot(range(start+1, end+1), NN6H_predicts[start: end, busShow + busNo],  linestyle= '-.', color='y')
    plt.hold
    plt.plot(range(start + 1, end + 1), NN8H_predicts[start: end, busShow + busNo], color='b')
    plt.hold
    plt.plot(range(start + 1, end + 1), conv_predicts[start: end, busShow + busNo], linestyle=':', color='g')
    plt.legend( ['真实值', '近似线性网络', '6层 FNN', '8层 FNN','LSTM'], loc=3, prop={'size': 8})

    #plt.hold
    #plt.plot(range(start+1, end+1),  gs_predicts[start: end, busShow+ busNo], color='c',linestyle= ':')

    plt.ylabel('电压相角 (度)')
    plt.xlabel('测试例子序列')
    axes = plt.gca()
    axes.set_xticks(list(range(start+1, end+1, x_step)))

bus_list = [50, 80, 100, 105]

for i, busShow in enumerate(bus_list):
    plt_bus(busShow = busShow, fig_num = i + 1)
plt.show()


