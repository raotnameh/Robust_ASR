#%%
#from pca import *
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

def p_plot(z, y, exp_name):
    principalComponents = print_pca(z, y)

    #x_max = np.max(principalComponents)#[:,0]),np.max(principalComponents[:,1])
    #x_min = np.min(principalComponents)#[:,0]),np.min(principalComponents[:,1])


    x_max, y_max = np.max(principalComponents[:,0]),np.max(principalComponents[:,1])
    x_min, y_min = np.min(principalComponents[:,0]),np.min(principalComponents[:,1])

    #principal_df = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], s= 50, c=y, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(3)-0.5).set_ticks(np.arange(2))
    plt.title('Visualizing for experiment: {}'.format(exp_name), fontsize=24)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.xlim(x_min - 0.1, x_max + 0.1)
    plt.ylim(y_min - 0.1, y_max + 0.1)

    plt.show()

def print_pca(x, y):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from metric_learn import NCA
    #import umap
    #sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, LatentDirichletAllocation
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    scaler = StandardScaler()
    #clf =  LDA(n_components=1)
    nca = NCA(max_iter = 25, n_components = 2, verbose = True)
    #KPCA = KernelPCA(n_components=2, kernel='rbf')
    #up = umap.UMAP()
    pca = PCA(n_components=50)
    #pca = TruncatedSVD(n_components=100)
    tsne = TSNE(n_components=2, 
        verbose = 0, 
        n_iter = 1000,
        learning_rate= 100,
        perplexity=30)
    scaler.fit(x)
    x = scaler.transform(x)
    #x = pca.fit_transform(x)
    x = tsne.fit_transform(x)
    #clf.fit(x, y)

    #x_min = np.amin(x, axis = 0)

    #print(x_min)

    #x = x - x_min[:,np.newaxis].T

    #print(np.amin(x, axis = 0))

    #clf.fit(x)

    #x = nca.fit_transform(x, y)

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


path = '/media/data_dump/asr/atul/visual/dev_sorted_EN_US'

files = 'exp_0.1_0.001'
''','exp_0.2_0.1', 'exp_0.5_0.001', 'exp_0.5_0.4',
'exp_0.01_0.2', 'exp_0.01_0.4', 'exp_0.0001_0.001', 
'exp_1e-05_0.2', 'exp_0.2_0.2', 'exp_0.2_0.001', 'exp_0.2_0.01', 'exp_1e-05_0.01', 
'exp_0.5_0.1', 'exp_0.01_0.01', 'exp_asr_0.001', 
'exp_0.5_0.2', 'exp_0.01_0.1', 'exp_0.5_0.01', 'exp_asr_0.5', 'exp_1e-05_0.001'
, 'exp_0.001_0.001', 'exp_0.1_0.001']'''

vis(path,files)
'''
for f in files:
    try:
        vis(path, f)
    except:
        print("I'm continuing")
        continue
'''

# %%
