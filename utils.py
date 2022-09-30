import numpy as np
import random
import os
import scipy.io
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
from numpy import linalg as LA
from scipy import signal
from scipy.linalg import toeplitz, eig, eigh
from scipy.stats import zscore, pearsonr


def eig_sorted(X, option='descending'):
    '''
    Eigenvalue decomposition, with ranked eigenvalues
    X = V @ np.diag(lam) @ LA.inv(V)
    '''
    lam, V = LA.eig(X)
    # lam = np.real(lam)
    # V = np.real(V)
    if option == 'descending':
        idx = np.argsort(-lam)
    elif option =='ascending':
        idx = np.argsort(lam)
    else:
        idx = range(len(lam))
        print('Warning: Not sorted')
    lam = lam[idx] # rank eigenvalues
    V = V[:, idx] # rearrange eigenvectors accordingly
    return lam, V


def PCAreg_inv(X, rank):
    '''
    PCA Regularized inverse of a symmetric square matrix X
    rank could be a smaller number than rank(X)
    '''
    lam, V = eig_sorted(X)
    lam = lam[:rank]
    V = V[:, :rank]
    inv = V @ np.diag(1/lam) @ np.transpose(V)
    return inv


def convolution_mtx(L_timefilter, x):
    first_col = np.zeros(L_timefilter)
    first_col[0] = x[0]
    conv_mtx = np.transpose(toeplitz(first_col, x))
    return conv_mtx


def split(EEG, Sti, fold=10, fold_idx=1):
    
    T, _ = EEG.shape
    len_test = T // fold
    EEG_test = EEG[len_test*(fold_idx-1):len_test*fold_idx,:]
    EEG_train = np.delete(EEG, range(len_test*(fold_idx-1), len_test*fold_idx), axis=0)
    Sti_test = Sti[len_test*(fold_idx-1):len_test*fold_idx]
    Sti_train = np.delete(Sti, range(len_test*(fold_idx-1), len_test*fold_idx), axis=0)
    return EEG_train, EEG_test, Sti_train, Sti_test


def corr_component(X, n_components):
    '''
    Input:
    X: EEG data with shape (T, D, N) [T: # sample, D: # channel, N: # subjects]
    '''
    _, D, N = X.shape
    Rw = np.zeros([D,D])
    for n in range(N):
        Rw += np.cov(np.transpose(X[:,:,n])) # Inside np.cov: observations in the columns
    Rt = N**2*np.cov(np.transpose(np.average(X, axis=2)))
    Rb = (Rt - Rw)/(N-1)
    rank = LA.matrix_rank(Rw)
    if rank < D:
        invRw = PCAreg_inv(Rw, rank)
    else:
        invRw = LA.inv(Rw)
    ISC, W = eig_sorted(invRw@Rb)
    # ISC also equals to np.diag((np.transpose(W)@Rb@W)/(np.transpose(W)@Rw@W))
    return ISC[:n_components], W[:,:n_components]
    

def cano_corr(X, Y, K_regu=7):
    '''
    Input:
    X: EEG data T(#sample)xD(#channel)
    Y: Stimulus T(#sample)xL(#tap)
    '''
    _, D = X.shape
    _, L = Y.shape
    # compute covariance matrices
    covXY = np.cov(X, Y, rowvar=False)
    Rx = covXY[:D,:D]
    Ry = covXY[D:D+L,D:D+L]
    Rxy = covXY[:D,D:D+L]
    Ryx = covXY[D:D+L,:D]
    # PCA regularization is recommended (set K_regu<rank(Rx))
    # such that the small eigenvalues dominated by noise are discarded
    invRx = PCAreg_inv(Rx, K_regu)
    invRy = PCAreg_inv(Ry, K_regu)
    A = invRx@Rxy@invRy@Ryx
    B = invRy@Ryx@invRx@Rxy
    # lam of A and lam of B should be the same
    # can be used as a preliminary check for correctness
    # the correlation coefficients are already available by taking sqrt of the eigenvalues: corr_coe = np.sqrt(lam[:K_regu])
    # or we do the following to obtain transformed X and Y and calculate corr_coe from there
    _, V_A = eig_sorted(A)
    _, V_B = eig_sorted(B)
    V_A = np.real(V_A[:,:K_regu])
    V_B = np.real(V_B[:,:K_regu])
    X_trans = X@V_A
    Y_trans = Y@V_B
    corr_pvalue = [pearsonr(X_trans[:,k], Y_trans[:,k]) for k in range(K_regu)]
    corr_coe = np.array([corr_pvalue[k][0] for k in range(K_regu)])
    # P-value-null hypothesis: the distributions underlying the samples are uncorrelated and normally distributed.
    p_value = np.array([corr_pvalue[k][1] for k in range(K_regu)])
    # to match filters v_a and v_b s.t. corr_coe is always positive
    V_A[:,corr_coe<0] = -1*V_A[:,corr_coe<0]
    corr_coe[corr_coe<0] = -1*corr_coe[corr_coe<0]
    return corr_coe, p_value, V_A, V_B


def GCCA(X, n_components, regularization='lwcov'):
    _, D, N = X.shape
    # From [X1; X2; ... XN] to [X1 X2 ... XN]
    # each column represents a variable, while the rows contain observations
    X_list = [X[:,:,n] for n in range(N)]
    X = np.concatenate(tuple(X_list), axis=1)
    if regularization == 'lwcov':
        Rxx = LedoitWolf().fit(X).covariance_
    else:
        Rxx = np.cov(X, rowvar=False)
    Dxx = np.zeros_like(Rxx)
    for n in range(N):
        Dxx[n*D:(n+1)*D,n*D:(n+1)*D] = Rxx[n*D:(n+1)*D,n*D:(n+1)*D]
    # Dxx and Rxx are symmetric matrices, so here we can use eigh
    # Otherwise we should use eig, which is much slower
    # Generalized eigenvalue decomposition
    # Dxx @ W = Rxx @ W @ np.diag(lam)
    # Dxx @ W[:,i] = lam[i] * Rxx @ W[:,i]
    lam, W = eigh(Dxx, Rxx, subset_by_index=[0,n_components-1]) # automatically ascend
    # lam also equals to np.diag(np.transpose(W)@Dxx@W)
    W = np.reshape(W, (N,D,-1))
    W = np.transpose(W, [1,0,2]) # W: D*N*n_components
    return lam, W


def avg_corr_coe(X, W, N, n_components=5):
    avg_corr = np.zeros(n_components)
    for component in range(n_components):
        avg_corr[component] = 0
        count = 0
        if np.ndim(W) == 3:
            GCCA = True
        else:
            GCCA = False
        for k in range(N):
            if GCCA:
                w1 = W[:,k,component]
            else:
                w1 = W[:,component]
            y1 = X[:,:,k]@w1
            for l in range(N):
                if l > k:
                    count = count + 1
                    if GCCA:
                        w2 = W[:,l,component]
                    else:
                        w2 = w1
                    y2 = X[:,:,l]@w2
                    corr_pvalue = pearsonr(y1, y2)
                    avg_corr[component] = avg_corr[component] + corr_pvalue[0]
        avg_corr[component] = avg_corr[component]/count
    return avg_corr


def shuffle_block(X, t, fs):
    block_len = t*fs
    T, D, N = X.shape
    append_arr = np.zeros((block_len-T%block_len, D, N))
    X = np.concatenate((X, append_arr), axis=0)
    T_appended = X.shape[0]
    X_shuffled = np.zeros_like(X)
    for n in range(N):
        blocks = [X[i:i+block_len, :, n] for i in range(0, T_appended, block_len)]
        random.shuffle(blocks)
        X_shuffled[:,:,n] = np.concatenate(tuple(blocks), axis=0)
    return X_shuffled


def permutation_test(X, Y, num_test, t, fs, topK, V_A=None, V_B=None):
    corr_coe_topK = np.empty((0, topK))
    X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=(1,2))
    for i in tqdm(range(num_test)):
        X_shuffled = np.squeeze(shuffle_block(X, t, fs), axis=2)
        Y_shuffled = np.squeeze(shuffle_block(Y, t, fs), axis=(1,2))
        L_timefilter = fs
        conv_mtx_shuffled = convolution_mtx(L_timefilter, Y_shuffled)
        if V_A is not None:
            X_shuffled_trans = X_shuffled@V_A
            Y_shuffled_trans = conv_mtx_shuffled@V_B
            K_regu = min(V_A.shape[1], V_B.shape[1])
            corr_pvalue = [pearsonr(X_shuffled_trans[:,k], Y_shuffled_trans[:,k]) for k in range(K_regu)]
            corr_coe = np.array([corr_pvalue[k][0] for k in range(K_regu)])
        else:
            corr_coe, _, _, _ = cano_corr(X_shuffled, conv_mtx_shuffled)
        corr_coe_topK = np.concatenate((corr_coe_topK, np.expand_dims(corr_coe[:topK], axis=0)), axis=0)
    return corr_coe_topK


def leave_one_fold_out(EEG, Sti, L_timefilter, K_regu=7, fold=10, fold_idx=1):
    EEG_train, EEG_test, Sti_train, Sti_test = split(EEG, Sti, fold=fold, fold_idx=fold_idx)
    conv_mtx_train = convolution_mtx(L_timefilter, Sti_train)
    corr_coe_train, p_value_train, V_A_train, V_B_train = cano_corr(EEG_train, conv_mtx_train)
    conv_mtx_test = convolution_mtx(L_timefilter, Sti_test)
    EEG_test_trans = EEG_test@V_A_train
    conv_mtx_test_trans = conv_mtx_test@V_B_train
    corr_pvalue = [pearsonr(EEG_test_trans[:,k], conv_mtx_test_trans[:,k]) for k in range(K_regu)]
    return corr_coe_train, corr_pvalue


def data_superbowl(head, datatype='preprocessed', year='2012', view='Y1'):
    path = head+'/'+datatype+'/'+year+'/'
    datafiles = os.listdir(path)
    X = []
    for datafile in datafiles:
        EEGdata = scipy.io.loadmat(path+datafile)
        fs = int(EEGdata['fsref'])
        data_per_subject = np.concatenate(tuple([EEGdata['Y1'][i][0] for i in range(len(EEGdata[view]))]),axis=1)
        data_per_subject = np.nan_to_num(data_per_subject, copy=False)
        X.append(np.transpose(data_per_subject))
    X = np.stack(tuple(X), axis=2)
    return X, fs