"""
Author: Tomasz Hachaj, 2021
Department of Signal Processing and Pattern Recognition
Institute of Computer Science in Pedagogical University of Krakow, Poland
https://sppr.up.krakow.pl/hachaj/
Data source:
https://credo.nkg-mn.com/hits.html
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
# %matplotlib notebook

point_size = 2
skip_artefacts = False
df=pd.read_csv('d:\\Projects\\Python\\PycharmProjects\\DLIB_Pytorch_Kropki\\my_vgg\\v2\\results\\VGG16_pretrained\\res_VGG16_pretrained_0.txt', header=None)
X = df.iloc[:,5:9].to_numpy()
df_ref = df.iloc[:,1:5].to_numpy()

df=pd.read_csv('d:\\Projects\\Python\\PycharmProjects\\DLIB_Pytorch_Kropki\\my_vgg\\v2\\results\\all_class\\1.zip.txt')
#xxx = df.iloc[:,2:5]
X = df.iloc[0:10000,2:6].to_numpy()
df_ref = df.iloc[0:10000,2:6].to_numpy()
#y = data.target
#scaler = StandardScaler(with_mean=False, with_std=False)
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

pca = PCA(n_components=4)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

ex_variance=np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio)
print(ex_variance)


Xax = X_pca[:,0]
Yax = X_pca[:,1]
Zax = X_pca[:,2]

cdict = {0:'red',1:'green'}
labl = {0:'Malignant',1:'Benign'}
marker = {0:'*',1:'o'}
alpha = {0:.3, 1:.5}

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

fig.patch.set_facecolor('white')

"""
sample_size = df.shape[0]
my_seed = 0
import random
random.seed(my_seed)
my_random_sample = random.sample(range(df.shape[0]), sample_size)
mask = np.zeros(df.shape[0], dtype=bool)
mask[my_random_sample] = True

df_train = df[mask]
"""

def return_color(Kropki,Kreski,Robaki,Artefakty):
    values = [Kropki,Kreski,Robaki,Artefakty]
    rv = values.index(max(values))
    color = 'red'
    if rv == 0:
        color = 'red'
    if rv == 1:
        color = 'green'
    if rv == 2:
        color = 'blue'
    if rv == 3:
        color = 'black'
    return color

for a in range(df_ref.shape[0]):
    my_color = return_color(df_ref[a,0],
                            df_ref[a,1],
                            df_ref[a,2],
                            df_ref[a,3])

    if skip_artefacts == False:
        ax.scatter(Xax[a], Yax[a], Zax[a], s=point_size, c=my_color)
    else:
        if my_color != 'black':
            ax.scatter(Xax[a], Yax[a], Zax[a], s=point_size, c=my_color)


ax.set_xlabel("1st PC Component", fontsize=10)
ax.set_ylabel("2nd Principal Component", fontsize=10)
ax.set_zlabel("3th Principal Component", fontsize=10)

#ax.view_init(elev=50., azim=50)
#ax.view_init(elev=80., azim=80)
ax.view_init(elev=22., azim=69)
ax.legend()
plt.show()