# Umap的纯连续型数据降维与可视化
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes 
from sklearn.svm import SVR
import umap
import pdb
from data_utils.Parameter_dataset import CNNParameters_Mnist
import matplotlib.pyplot as plt


root1 = './mnist_trained'  #path to the trained models
root2 = './mnist_random' #path to the random models
set = CNNParameters_Mnist(root1) #set.data.shape: torch[1100,5066]
random_set = CNNParameters_Mnist(root2) # [1000, 5066]

trained = set.data.numpy()
random = random_set.data.numpy()
for i in range(10):
    img = trained[i][:4900]
    img = (img - img.min())/(img.max() - img.min())
    img = img.reshape(70,70)
    plt.imshow(img)
    plt.savefig(f'./img/trained{i}.png')
for i in range(10):
    img = random[i][:4900]
    img = (img - img.min())/(img.max() - img.min())
    img = img.reshape(70,70)
    plt.imshow(img)
    plt.savefig(f'./img/random{i}.png')
 
# # 将数组还原成图片 Image.fromarray方法 传入数组 和 通道
#     img = Image.fromarray(x_train[0])
# # img.save('1.jpg')

    plt.imshow(img)
    # plt.clf()
    plt.savefig(f'./img/trained{i}.png')
