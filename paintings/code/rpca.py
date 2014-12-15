"""Visualize the data in 2d rPCA"""
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from sklearn.decomposition import RandomizedPCA

all_data = np.loadtxt('../data/Paintings/two_class/big/Paintings_train.csv',delimiter=',')
data = all_data[:,:-1]
y = all_data[:,-1]
pca = RandomizedPCA(n_components=2)
X = pca.fit_transform(data)
df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1],\
                   "label":np.where(y==1, "realism", "abstract")})
colors = ["red", "yellow"]
for label, color in zip(df['label'].unique(), colors):
    mask = df['label']==label
    pl.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
pl.legend()
pl.title('Randomized PCA')
pl.show()

