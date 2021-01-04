import torch
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='notebook', rc={'figure.figsize':(10,8)})
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def p_plot(z, y, exp_name):

    principalComponents = print_pca(z, y)
    x_max, y_max = np.max(principalComponents[:,0]),np.max(principalComponents[:,1])
    x_min, y_min = np.min(principalComponents[:,0]),np.min(principalComponents[:,1])

    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], s= 50, c=y, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(3)-0.5).set_ticks(np.arange(2))
    plt.title('Visualizing for experiment: {}'.format(exp_name), fontsize=24)
    plt.xlabel('Embedding Component 1')
    plt.ylabel('Embedding Component 2')
    plt.xlim(x_min - 0.1, x_max + 0.1)
    plt.ylim(y_min - 0.1, y_max + 0.1)

    plt.show()

def print_pca(x, y):

    scaler = StandardScaler()
    
    pca = PCA(n_components=50)

    tsne = TSNE(n_components=2, 
        verbose = 0, 
        n_iter = 1000,
        learning_rate= 100,
        perplexity=30)

    scaler.fit(x)
    x = scaler.transform(x)
    x = pca.fit_transform(x)
    x = tsne.fit_transform(x)

    return x


def vis(path, file_name):

    en_us_z_f = np.load(path+file_name+'feat_z.npy')
    en_us_z_y = np.load(path+file_name+'labels_z.npy')

    en_us_z_f_t = np.load(path+file_name+'feat_z_.npy')
    en_us_z_y_t = np.load(path+file_name+'labels_z_.npy')

    z = en_us_z_f
    y = en_us_z_y

    z_t = en_us_z_f_t
    y_t = en_us_z_y_t

    p_plot(z, y, file_name+" z")

    p_plot(z_t,y_t, file_name+"z tilde")


if __name__ = '__main__':
    path = 'path to files'
    files = 'files'

    vis(path,files)

