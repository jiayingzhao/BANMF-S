import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import networkx as nx


def minST(data, dist_type, k = 10, ndim = 50, reduction = 'std', filt = None):

    # data: expression matrix, size is #of cells by #of genes
    # dist_type: 'seuclidan','minkowski','mahalanobis' or any distance matrix
    # data = pd.read_csv('cellbygene_lognorm.csv', index_col=0)
    np.random.seed(123)

    if isinstance(data, np.ndarray):
        print("the input data is a numpy array")
    else:
        data = np.array(data)


    ncell, ngene = data.shape
    # ndim = round(prop*ngene)

    if ndim > min(ncell,ngene):
        ndim = round(0.5*min(ncell,ngene))

    if reduction=='pca':
        pcainstance=PCA(ndim)
        data_dim = pcainstance.fit_transform(data)
        
    else:
        stddata = np.std(data,axis=0)    
        sind = np.argsort(-stddata)
        data_dim = data[:, sind[0:ndim]]

    print(dist_type)


    if dist_type == 'seuclidean':
        data_dist = pdist(data_dim,'seuclidean')
    elif dist_type == 'minkowski':
        data_dist = pdist(data_dim,'minkowski')
    elif dist_type == 'mahalanobis':
        data_dist = pdist(data_dim,'mahalanobis')
    elif dist_type == 'cosine':
        data_dist = pdist(data_dim,'cosine') 
    elif dist_type == 'cityblock':
        data_dist = pdist(data_dim,'cityblock') 
    elif dist_type == 'correlation':
        data_dist = pdist(data_dim,'correlation') 
    elif dist_type == 'hamming':
        data_dist = pdist(data_dim,'hamming') 
    elif dist_type == 'jaccard':
        data_dist = pdist(data_dim,'jaccard') 
    elif dist_type == 'spearman':
        data_dist = pdist(data_dim,'spearman')  
    else:
        data_dist = pdist(data_dim,metric=dist_type)
    
    data_dist = squareform(data_dist)

    G = nx.from_numpy_array(data_dist)
    data_mst = nx.minimum_spanning_tree(G = G)
    A_mst = nx.adjacency_matrix(data_mst)

    G = nx.from_numpy_array(A_mst)

    ncell = A_mst.shape[0]
    csimmat = np.zeros((ncell, ncell))
    
    for i in range(ncell-1):
        for j in range(i+1,ncell):
            pList = nx.shortest_path(G, source=i, target=j)
            pDist = [A_mst[pList[k],pList[k+1]] for k in range(len(pList)-1)]
            csimmat[i,j] = 1/sum(pDist)

    csimmat = csimmat + csimmat.T
    
    if filt != None:
        csimmat[csimmat<filt] = 0

    return csimmat




