import numpy as np
import pandas as pd
from scipy import sparse
from scipy import io as sio
from sklearn.cluster import KMeans
from scipy.optimize import nnls


def BuildBinsFromMat(mat, B, size_blk):
    '''
    Build bins from matrix
    '''
    len_r, len_c = mat.shape
    print('The shape of the matrix is ...')
    print (len_r, len_c)
    len_blk_r, len_blk_c = size_blk
    bins = [[[] for b2 in range(B)] for b1 in range(B)]
    for br in range(B):
        for bc in range(B):
            r1 = int(len_blk_r * br)
            r2 = int(min(len_blk_r * (br+1), len_r))
            c1 = int(len_blk_c * bc)
            c2 = int(min(len_blk_c * (bc+1), len_c))
            bins[br][bc] = mat[r1:r2, c1:c2]
    return bins



def Adjmat2Bins(filename, B, sym_flag, shuf_flag, diag_flag, seed):
    '''
        Read adjacency matrix from file in the form of adj matrix, organize it into bins.
        Output: array list
    '''
    if seed == None:
        np.random.seed(12345)
    else:
        np.random.seed(seed)

    A = pd.read_csv(filename,index_col=0)
    A = np.array(A)
    print('The shape of the input matrix is...')
    print(A.shape)

    size_blk = np.ceil(np.array(A.shape) / float(B))

    if (sym_flag == 1) & (A.shape[0] == A.shape[1]):
        A = A + A.T
    if (diag_flag == 1) & (A.shape[0] == A.shape[1]):
        A = A + np.eye(A.shape[0])

    if shuf_flag == 1:
        shuf = np.random.permutation(A.shape[0])
    else:
        shuf = np.array(range(A.shape[0]))

    print ("Number of edges: ", np.sum(A>0))
    A = BuildBinsFromMat(A, B, size_blk)     

    return A, shuf


def X2Bins(filename, B, shuf_flag, shufa, shufl):
    '''
    Read attribute matrix from file, organize it into bins.
    Output: A list of array
    '''
    X = pd.read_csv(filename,index_col=0)
    X = np.array(X)

    size_X = np.array(X.shape)
    print('The shape of the input matrix is...')
    print(X.shape)

    # Shuffle if required
    if shuf_flag == 1:
        X = X[shufa, shufl]

    # Compute bin sizes
    size_blk = np.ceil(size_X/float(B))

    # compute masking operator
    M = X.copy()
    M [M > 0] = 1

    # core function
    X = BuildBinsFromMat(X, B, size_blk)
    M = BuildBinsFromMat(M, B, size_blk)

    return X, M

def WHinit1(filename, B, shuf_flag, shufa, shufl, R, seed = None):
    '''
    Initialize W and H using SVD.
    Output: list of arrays W and H
    '''
    if seed == None:
        np.random.seed(12345)
    else:
        np.random.seed(seed)

    X = pd.read_csv(filename,index_col=0)
    X = np.array(X)
    print('The shape of the expression matrix is...')
    print(X.shape)

    # Shuffle if required
    if shuf_flag == 1:
        X = X[shufa, shufl]

    U, sigma, VT = np.linalg.svd(X)
    Sigma = np.diag(sigma[0:R])
    H_sigma = np.sqrt(Sigma * X.shape[1] / X.shape[0])
    H = H_sigma .dot(VT[0:R, :])
    H1_rand = np.multiply(H[0, :], 2 * np.random.rand(X.shape[1])-1)
    H[0, :] = H1_rand
  
    W = np.dot(X, np.linalg.pinv(H))

    len_cH = H.shape[1]
    len_rW = W.shape[0]
    len_blk_W = np.ceil(len_rW/float(B))
    len_blk_H = np.ceil(len_cH/float(B))

    W_bins = [[] for b2 in range(B)]
    H_bins = [[] for b2 in range(B)]
    for b2 in range(B):
        r1 = int(len_blk_W * b2)
        r2 = int(min(len_blk_W * (b2 + 1), len_rW))
        c1 = int(len_blk_H * b2)
        c2 = int(min(len_blk_H * (b2 + 1), len_cH))
        W_bins[b2]=W[r1:r2, :]
        H_bins[b2]=H[: ,c1:c2]

    for b in range(B):
        
        W_bins[b] = W_bins[b].clip(min=0)
        H_bins[b] = H_bins[b].clip(min=0)

    return W_bins, H_bins


def WHinit2(filename, B, shuf_flag, shufa, shufl, R, seed = None):
    '''
    Initialize W and H using SVD.
    Output: list of arrays W and H
    '''

    if seed == None:
        np.random.seed(12345)
    else:
        np.random.seed(seed)

    X = pd.read_csv(filename,index_col=0)
    X = np.array(X)
    print('The shape of the expression matrix is...')
    print(X.shape) 

    # Shuffle if required
    if shuf_flag == 1:
        X = X[shufa, shufl]

    U, sigma, VT = np.linalg.svd(X)
    V = np.transpose(VT)
    Sigma = np.diag(sigma[0:R])
    W = np.zeros((X.shape[0], R))
    H = np.zeros((R, X.shape[1]))

    U[:,1] = np.multiply(U[:, 1], 2 * np.random.rand(X.shape[0]) - 1)
    V[:,1] = np.multiply(V[:, 1], 2 * np.random.rand(X.shape[1]) - 1)


    for b in range(0, R):
        uu = U[:, b]
        vv = V[:, b]
        uup = uu.clip(min=0)
        uun = uu.clip(max=0)
        uun = -uun
        vvp = vv.clip(min=0)
        vvn = vv.clip(max=0)
        vvn = - vvn
        n_uup = np.linalg.norm(uup)
        n_vvp = np.linalg.norm(vvp)
        n_uun = np.linalg.norm(uun)
        n_vvn = np.linalg.norm(vvn)
        termp = n_uup * n_vvp
        termn = n_uun * n_vvn

        if termp >= termn :
            W[:, b]=np.sqrt(Sigma[b, b] * termp  * X.shape[0] / X.shape[1]) * uup / n_uup
            H[b, :]= np.sqrt(Sigma[b, b] * termp * X.shape[1] / X.shape[0]) * np.transpose(vvp) / n_vvp
        else :
            W[:, b] = np.sqrt(Sigma[b, b] * termn * X.shape[0] / X.shape[1]) * uun / n_uun
            H[b, :] = np.sqrt(Sigma[b, b] * termn * X.shape[1] / X.shape[0]) * np.transpose(vvn) / n_vvn

   

    len_cH = H.shape[1]
    len_rW = W.shape[0]
    len_blk_W = np.ceil(len_rW/float(B))
    len_blk_H = np.ceil(len_cH/float(B))

    W_bins = [[] for b2 in range(B)]
    H_bins = [[] for b2 in range(B)]
    for b2 in range(B):
        r1 = int(len_blk_W * b2)
        r2 = int(min(len_blk_W * (b2 + 1), len_rW))
        c1 = int(len_blk_H * b2)
        c2 = int(min(len_blk_H * (b2 + 1), len_cH))
        W_bins[b2]=W[r1:r2, :]
        H_bins[b2]=H[: ,c1:c2]

    return W_bins, H_bins


def WHinit3(filename, B, k, shuf_flag, shufa, shufl, R, seed = None):
    '''
    Initialize W and H using kmeans.
    Output: list of arrays W and H
    '''

    if seed == None:
        np.random.seed(12345)
    else:
        np.random.seed(seed)

    X = pd.read_csv(filename,index_col=0)
    X = np.array(X)
    print('The shape of the expression matrix is...')
    print(X.shape) 

    # Shuffle if required
    if shuf_flag == 1:
        X = X[shufa, shufl]

    kmeans = KMeans(n_clusters  = k).fit(X)
    print('the kmeans result is of shape: '+str(kmeans.labels_.shape))
    tmp = pd.DataFrame(X)
    tmp['group'] = kmeans.labels_

    H = tmp.groupby('group').mean()
    print('the kmeans init H is of shape: '+str(H.shape))

    H = np.array(H)
    
    W = np.zeros((X.shape[0],k))

    for i in range(X.shape[0]):
        W[i,:], _ = nnls(H.T,X.T[:,i])

    len_cH = H.shape[1]
    len_rW = W.shape[0]
    len_blk_W = np.ceil(len_rW/float(B))
    len_blk_H = np.ceil(len_cH/float(B))

    W_bins = [[] for b2 in range(B)]
    H_bins = [[] for b2 in range(B)]
    for b2 in range(B):
        r1 = int(len_blk_W * b2)
        r2 = int(min(len_blk_W * (b2 + 1), len_rW))
        c1 = int(len_blk_H * b2)
        c2 = int(min(len_blk_H * (b2 + 1), len_cH))
        W_bins[b2]=W[r1:r2, :]
        H_bins[b2]=H[: ,c1:c2]

    return W_bins, H_bins
