import numpy as np
import random
import os
import mne
import scipy.io
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
from numpy import linalg as LA
from scipy import signal
from scipy.linalg import toeplitz, eig, eigh, sqrtm
from scipy.sparse.linalg import eigs
from scipy.stats import zscore, pearsonr
from numba import jit


def eig_sorted(X, option='descending'):
    '''
    Eigenvalue decomposition, with ranked eigenvalues
    X = V @ np.diag(lam) @ LA.inv(V)
    Could be replaced by eig in scipy.linalg
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


def convolution_mtx(L_timefilter, x, causal=True):
    '''
    Calculate the convolution matrix
    Convolution: y(t)=x(t)*h(t)
    In matrix form: y=Xh E.g. time lag = 3
    If causal,
    h = h(0); h(1); h(2)
    X = 
    x(0)   x(-1)  x(-2)
    x(1)   x(0)   x(-1)
            ...
    x(T-1) x(T-2) x(T-3)
    If non-causal,
    h = h(-1); h(0); h(1)
    X = 
    x(1)   x(0)   x(-1)
    x(2)   x(1)   x(0)
            ...
    x(T)   x(T-1) x(T-2)
    Unknown values are set as 0
    '''
    first_col = np.zeros(L_timefilter)
    first_col[0] = x[0]
    if causal:
        conv_mtx = np.transpose(toeplitz(first_col, x))
    else:
        assert(L_timefilter % 2 == 1)
        L = int((L_timefilter-1)/2)
        x = np.append(x, [np.zeros((1,L))])
        conv_mtx = np.transpose(toeplitz(first_col, x))
        conv_mtx = conv_mtx[L:,:]
    return conv_mtx


def block_Hankel(X, L, causal=False):
    '''
    For spatial-temporal filter, calculate the block Hankel matrix
    Inputs:
    X: T(#sample)xD(#channel)
    L: number of time lags; from -(L-1) to 0 (causal) or -(L-1)/2 to (L-1)/2 (non-causal)
    causal: default false
    '''
    if L == 1:
        blockHankel = X
    else:
        Hankel_list = [convolution_mtx(L, X[:,i], causal=causal) for i in range(X.shape[1])]
        blockHankel = np.concatenate(tuple(Hankel_list), axis=1)
    return blockHankel


def split(EEG, Sti, fold=10, fold_idx=1):
    '''
    Split datasets as one fold specified by fold_idx (test set), and the rest folds (training set). 
    '''
    T = EEG.shape[0]
    len_test = T // fold
    if np.ndim(EEG)==2:
        EEG_test = EEG[len_test*(fold_idx-1):len_test*fold_idx,:]
        EEG_train = np.delete(EEG, range(len_test*(fold_idx-1), len_test*fold_idx), axis=0)
    elif np.ndim(EEG)==3:
        EEG_test = EEG[len_test*(fold_idx-1):len_test*fold_idx,:,:]
        EEG_train = np.delete(EEG, range(len_test*(fold_idx-1), len_test*fold_idx), axis=0)
    else:
        print('Warning: Check the dimension of EEG data')
    if np.ndim(Sti)==1:
        Sti = np.expand_dims(Sti, axis=1)
    Sti_test = Sti[len_test*(fold_idx-1):len_test*fold_idx,:]
    Sti_train = np.delete(Sti, range(len_test*(fold_idx-1), len_test*fold_idx), axis=0)
    return EEG_train, EEG_test, Sti_train, Sti_test


def split_balance(EEG_list, Sti_list, fold=10, fold_idx=1):
    '''
    For multiple videos in Sti_list and the corresponding EEG responses in EEG_list, 
    split them as one fold specified by fold_idx (test set), and the rest folds (training set).
    Merge the EEG responses and stimuli from different videos into one training set and one test set.
    '''
    split_list = [split(EEG, Sti, fold, fold_idx) for EEG, Sti in zip(EEG_list, Sti_list)]
    EEG_train = np.concatenate(tuple([split_list[i][0] for i in range(len(split_list))]), axis=0)
    EEG_test = np.concatenate(tuple([split_list[i][1] for i in range(len(split_list))]), axis=0)
    Sti_train = np.concatenate(tuple([split_list[i][2] for i in range(len(split_list))]), axis=0)
    Sti_test = np.concatenate(tuple([split_list[i][3] for i in range(len(split_list))]), axis=0)
    if np.ndim(Sti_train)==1:
        Sti_train = np.expand_dims(Sti_train, axis=1)
        Sti_test = np.expand_dims(Sti_test, axis=1)
    return EEG_train, EEG_test, Sti_train, Sti_test


def split_multi_mod(datalist, fold=10, fold_idx=1):
    '''
    Split datasets as one fold specified by fold_idx (test set), and the rest folds (training set). 
    Datasets are organized in datalist.
    '''
    train_list = []
    test_list = []
    for data in datalist:
        T = data.shape[0]
        len_test = T // fold
        if np.ndim(data)==1:
            data_test = np.expand_dims(data[len_test*(fold_idx-1):len_test*fold_idx], axis=1)
            data_train = np.expand_dims(np.delete(data, range(len_test*(fold_idx-1), len_test*fold_idx)), axis=1)
        elif np.ndim(data)==2:
            data_test = data[len_test*(fold_idx-1):len_test*fold_idx,:]
            data_train = np.delete(data, range(len_test*(fold_idx-1), len_test*fold_idx), axis=0)
        elif np.ndim(data)==3:
            data_test = data[len_test*(fold_idx-1):len_test*fold_idx,:,:]
            data_train = np.delete(data, range(len_test*(fold_idx-1), len_test*fold_idx), axis=0)
        else:
            print('Warning: Check the dimension of data')
        train_list.append(data_train)
        test_list.append(data_test)
    return train_list, test_list


def split_mm_balance(nested_datalist, fold=10, fold_idx=1):
    '''
    Datasets are organized in nested datalist: [[EEG_1, EEG_2, ... ], [Vis_1, Vis_2, ... ], [Sd_1, Sd_2, ... ]]
    Split using split_multi_mod for [EEG_i, Vis_i, Sd_i] for i=1,2,..., and merge the results into
    - A traininng list: [EEG_train, Vis_train, Sd_train]
    - A test list: [EEG_test, Vis_test, Sd_test]
    '''
    nb_clips = len(nested_datalist[0])
    nb_mod = len(nested_datalist)
    re_arrange = []
    for i in range(nb_clips):
       re_arrange.append([nested_datalist[j][i] for j in range(nb_mod)]) 
    split_list = [split_multi_mod(data, fold, fold_idx) for data in re_arrange]
    train_list = []
    test_list = []
    for i in range(nb_mod):
        train_list.append(np.concatenate(tuple([split_list[j][0][i] for j in range(nb_clips)]), axis=0))
        test_list.append(np.concatenate(tuple([split_list[j][1][i] for j in range(nb_clips)]), axis=0))
    return train_list, test_list


def corr_component(X, n_components, W_train=None):
    '''
    Inputs:
    X: EEG data with shape (T, D, N) [T: # sample, D: # channel, N: # subjects]
    n_components: number of components
    W_train: If not None, then goes to test mode.
    Outputs:
    ISC: inter-subject correlation
    W: weights
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
    if W_train is not None: # Test mode
        W = W_train
        ISC = np.diag((np.transpose(W)@Rb@W)/(np.transpose(W)@Rw@W))
    else: # Train mode
        ISC, W = eig_sorted(invRw@Rb)
    # TODO: ISC here is an approximation of real average pairwise correlation
    return ISC[:n_components], W[:,:n_components]
    

def cano_corr(X, Y, Lx=1, Ly=1, causalx=False, causaly=True, n_components=5, regularization='lwcov', K_regu=None, V_A=None, V_B=None, Lam=None):
    '''
    Input:
    X: EEG data T(#sample)xDx(#channel)
    Y: Stimulus T(#sample)xDy(#feature dim)
    Lx/Ly: If use (spatial-) temporal filter, the number of taps
    causalx/causaly: If use (spatial-) temporal filter, causal or not
    n_components: Number of components to be returned
    regularization: Regularization of the estimated covariance matrix
    K_regu: Number of eigenvalues to be kept. Others will be set to zero. Keep all if K_regu=None
    V_A/V_B: Filters of X and Y. Use only in test mode.
    '''
    _, Dx = X.shape
    _, Dy = Y.shape
    mtx_X = block_Hankel(X, Lx, causal=causalx)
    mtx_Y = block_Hankel(Y, Ly, causal=causaly)
    if V_A is not None: # Test Mode
        flag_test = True
    else: # Train mode
        flag_test = False
        # compute covariance matrices
        covXY = np.cov(mtx_X, mtx_Y, rowvar=False)
        if regularization=='lwcov':
            Rx = LedoitWolf().fit(mtx_X).covariance_
            Ry = LedoitWolf().fit(mtx_Y).covariance_
        else:
            Rx = covXY[:Dx*Lx,:Dx*Lx]
            Ry = covXY[Dx*Lx:Dx*Lx+Dy*Ly,Dx*Lx:Dx*Lx+Dy*Ly]
        Rxy = covXY[:Dx*Lx,Dx*Lx:Dx*Lx+Dy*Ly]
        Ryx = covXY[Dx*Lx:Dx*Lx+Dy*Ly,:Dx*Lx]
        # PCA regularization is recommended (set K_regu<rank(Rx))
        # such that the small eigenvalues dominated by noise are discarded
        if K_regu is None:
            invRx = PCAreg_inv(Rx, LA.matrix_rank(Rx))
            invRy = PCAreg_inv(Ry, LA.matrix_rank(Ry))
        else:
            K_regu = min(LA.matrix_rank(Rx), LA.matrix_rank(Ry), K_regu)
            invRx = PCAreg_inv(Rx, K_regu)
            invRy = PCAreg_inv(Ry, K_regu)
        A = invRx@Rxy@invRy@Ryx
        B = invRy@Ryx@invRx@Rxy
        # lam of A and lam of B should be the same
        # can be used as a preliminary check for correctness
        # the correlation coefficients are already available by taking sqrt of the eigenvalues: corr_coe = np.sqrt(lam[:K_regu])
        # or we do the following to obtain transformed X and Y and calculate corr_coe from there
        Lam, V_A = eig_sorted(A)
        _, V_B = eig_sorted(B)
        Lam = np.real(Lam[:n_components])
        V_A = np.real(V_A[:,:n_components])
        V_B = np.real(V_B[:,:n_components])
    X_trans = mtx_X@V_A
    Y_trans = mtx_Y@V_B
    corr_pvalue = [pearsonr(X_trans[:,k], Y_trans[:,k]) for k in range(n_components)]
    corr_coe = np.array([corr_pvalue[k][0] for k in range(n_components)])
    # P-value-null hypothesis: the distributions underlying the samples are uncorrelated and normally distributed.
    p_value = np.array([corr_pvalue[k][1] for k in range(n_components)])
    if not flag_test:
        # to match filters v_a and v_b s.t. corr_coe is always positive
        V_A[:,corr_coe<0] = -1*V_A[:,corr_coe<0]
        corr_coe[corr_coe<0] = -1*corr_coe[corr_coe<0]
    return corr_coe, p_value, V_A, V_B, Lam


def GCCA(X_stack, L, causal, n_components, regularization='lwcov'):
    '''
    LEGACY. USE GCCA_multi_modal INSTEAD.
    Inputs:
    X_stack: stacked (along axis 2) data of different subjects
    n_components: number of components
    regularization: regularization method when estimating covariance matrices (Default: LedoitWolf)
    W_train: If not None, then goes to test mode.
    Outputs:
    lam: eigenvalues, related to mean squared error (not used in analysis)
    W_stack: (rescaled) weights with shape (D*N*n_components)
    avg_corr: average pairwise correlation
    '''
    T, D, N = X_stack.shape
    # From [X1; X2; ... XN] to [X1 X2 ... XN]
    # each column represents a variable, while the rows contain observations
    X_list = [block_Hankel(X_stack[:,:,n], L, causal) for n in range(N)]
    X = np.concatenate(tuple(X_list), axis=1)
    Rxx = np.cov(X, rowvar=False)
    Dxx = np.zeros_like(Rxx)
    for n in range(N):
        if regularization == 'lwcov':
            Rxx[n*D*L:(n+1)*D*L, n*D*L:(n+1)*D*L] = LedoitWolf().fit(X[:, n*D*L:(n+1)*D*L]).covariance_
        Dxx[n*D*L:(n+1)*D*L, n*D*L:(n+1)*D*L] = Rxx[n*D*L:(n+1)*D*L, n*D*L:(n+1)*D*L]
    lam, W = eigh(Dxx, Rxx, subset_by_index=[0,n_components-1]) # automatically ascend
    Lam = np.diag(lam)
    # Right scaling
    W = W @ sqrtm(LA.inv(Lam.T @ W.T @ Rxx @ W @ Lam))
    # Forward models
    F_redun = T * Dxx @ W
    # Reshape W as (DL*n_components*N)
    W_stack = np.reshape(W, (N,D*L,-1))
    W_stack = np.transpose(W_stack, [1,0,2])
    F_redun_stack = np.reshape(F_redun, (N,D*L,-1))
    F_redun_stack = np.transpose(F_redun_stack, [1,0,2])
    F_stack = F_organize(F_redun_stack, L, causal, avg=True)
    return W_stack, F_stack, lam


def SI_GCCA(datalist, Llist, causal_list, n_components, rho, regularization='lwcov'):
    EEG, Stim = datalist
    T, D_eeg, N = EEG.shape
    _, D_stim = Stim.shape
    L_EEG, L_Stim = Llist
    dim_list = [D_eeg*L_EEG]*N + [D_stim*L_Stim]
    causal_EEG, causal_Stim = causal_list
    EEG_list = [block_Hankel(EEG[:,:,n], L_EEG, causal_EEG) for n in range(N)]
    EEG_Hankel = np.concatenate(tuple(EEG_list), axis=1)
    Stim_Hankel = block_Hankel(Stim, L_Stim, causal_Stim)
    X = np.concatenate((EEG_Hankel, Stim_Hankel), axis=1)
    Rxx = np.cov(X, rowvar=False)
    Dxx = np.zeros_like(Rxx)
    dim_accumu = 0
    for dim in dim_list:
        if regularization == 'lwcov':
            Rxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim] = LedoitWolf().fit(X[:,dim_accumu:dim_accumu+dim]).covariance_
        Dxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim] = Rxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim]
        dim_accumu = dim_accumu + dim
    Rxx[:,-D_stim*L_Stim:] = Rxx[:,-D_stim*L_Stim:]*rho
    lam, W = eig(Dxx, Rxx)
    idx = np.argsort(lam)
    lam = np.real(lam[idx]) # rank eigenvalues
    W = np.real(W[:, idx]) # rearrange eigenvectors accordingly
    Lam = np.diag(lam)[:n_components,:n_components]
    Rxx[-D_stim*L_Stim:, :] = Rxx[-D_stim*L_Stim:, :]*rho
    W = W[:,:n_components]
    # Right scaling
    W = W @ sqrtm(LA.inv(Lam.T @ W.T @ Rxx @ W @ Lam))
    # Forward models
    F = T * Dxx @ W
    # Organize weights of different modalities
    Wlist = W_organize(W, datalist, Llist)
    Flist = W_organize(F, datalist, Llist)
    Fstack = F_organize(Flist[0], L_EEG, causal_EEG, avg=True)
    return Wlist, Fstack, lam[:n_components]


def GCCA_multi_modal(datalist, Llist, causal_list, n_components, rhos, regularization='lwcov'):
    '''
    Inputs:
    datalist: data of different modalities (a list) E.g, [EEG_stack, Stim]
    Llist: number of taps of different modalities.
    causal_list: causal filter or not
    n_components: number of components
    rhos: controls the weights of different modalities; should have the same length as the datalist
    regularization: regularization method when estimating covariance matrices (Default: LedoitWolf)
    Outputs:
    Wlist: weights of different modalities (a list)
    '''
    dim_list = []
    flatten_list = []
    rho_list = []
    for i in range(len(datalist)):
        rawdata = datalist[i]
        L = Llist[i]
        causal = causal_list[i]
        rho = rhos[i]
        if np.ndim(rawdata) == 3:
            T, D, N = rawdata.shape
            X_list = [block_Hankel(rawdata[:,:,n], L, causal) for n in range(N)]
            X = np.concatenate(tuple(X_list), axis=1)
            flatten_list.append(X)
            dim_list = dim_list + [D*L]*N
            rho_list = rho_list + [rho]*(D*L*N)
        elif np.ndim(rawdata) == 2:
            T, D = rawdata.shape
            flatten_list.append(block_Hankel(rawdata, L, causal))
            dim_list.append(D*L)
            rho_list = rho_list + [rho]*(D*L)
        else:
            print('Warning: Check dim of data')
    X_mm = np.concatenate(tuple(flatten_list), axis=1)
    Rxx = np.cov(X_mm, rowvar=False)
    Dxx = np.zeros_like(Rxx)
    dim_accumu = 0
    for dim in dim_list:
        if regularization == 'lwcov':
            Rxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim] = LedoitWolf().fit(X_mm[:,dim_accumu:dim_accumu+dim]).covariance_
        Dxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim] = Rxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim]
        dim_accumu = dim_accumu + dim
    Rxx = Rxx*np.expand_dims(np.array(rho_list), axis=0)
    # Dxx and Rxx are symmetric matrices, so here we can use eigh
    # Otherwise we should use eig, which is much slower
    # Generalized eigenvalue decomposition
    # Dxx @ W = Rxx @ W @ np.diag(lam)
    # Dxx @ W[:,i] = lam[i] * Rxx @ W[:,i]
    # lam, W = eigh(Dxx, Rxx, subset_by_index=[0,n_components-1]) # automatically ascend
    # lam, W = eigs(Dxx, n_components, Rxx, which='SR')
    # W = np.real(W)
    lam, W = eig(Dxx, Rxx)
    idx = np.argsort(lam)
    lam = np.real(lam[idx]) # rank eigenvalues
    W = np.real(W[:, idx]) # rearrange eigenvectors accordingly
    Lam = np.diag(lam)[:n_components,:n_components]
    Rxx = Rxx * np.expand_dims(np.array(rho_list), axis=1)
    W = W[:,:n_components]
    # Right scaling
    W = W @ sqrtm(LA.inv(Lam.T @ W.T @ Rxx @ W @ Lam))
    # Forward models
    F = T * Dxx @ W
    # Organize weights of different modalities
    Wlist = W_organize(W, datalist, Llist)
    Flist = W_organize(F, datalist, Llist)
    return Wlist, Flist, lam[:n_components]


def W_organize(W, datalist, Llist):
    '''
    Input: 
    W generated by GCCA_multi_modal
    Output:
    Organized W list containing W of each modality 
    '''
    W_list = []
    dim_start = 0
    for i in range(len(datalist)):
        rawdata = datalist[i]
        L = Llist[i]
        if np.ndim(rawdata) == 3:
            _, D, N = rawdata.shape
            dim_end = dim_start + D*L*N
            W_temp = W[dim_start:dim_end,:]
            W_stack = np.reshape(W_temp, (N,D*L,-1))
            W_list.append(np.transpose(W_stack, [1,0,2]))
        elif np.ndim(rawdata) == 2:
            _, D = rawdata.shape
            dim_end = dim_start + D*L
            W_list.append(W[dim_start:dim_end,:])
        else:
            print('Warning: Check the dim of data')
        dim_start = dim_end
    return W_list


def F_organize(F_redun, L, causal, avg=True):
    '''
    Input: 
    F_redun (not list) generated by GCCA_multi_modal
    Output:
    Forward model DxNxK
    '''
    if np.ndim(F_redun) == 3:
        DL, _, _ = F_redun.shape
    else:
        DL, _ = F_redun.shape
    D = int(DL/L)
    if causal:
        offset = 0
    else:
        offset = L // 2
    indices = [i*L+offset for i in range(D)]
    if np.ndim(F_redun) == 3:
        F = F_redun[indices,:,:]
        if avg:
            F = np.average(F, axis=1)
    else:
        F = F_redun[indices,:]
    return F


def forward_model(X, W_Hankel, L=1, causal=False):
    '''
    Reference: On the interpretation of weight vectors of linear models in multivariate neuroimaging https://www.sciencedirect.com/science/article/pii/S1053811913010914
    Backward models: Extract latent factors as functions of the observed data s(t) = W^T x(t)
    Forward models: Reconstruct observations from latent factors x(t) = As(t) + n(t)
    x(t): D-dimensional observations
    s(t): K-dimensional latent factors
    W: backward model
    A: forward model

    In our use case the backward model can be found using (G)CCA. Latent factors are the representations generated by different components.
    X_Hankel W_Hankel = S     X:TxDL W:DLxK S:TxK
    S F.T = X                 F: DxK
    F = X.T X_Hankel W_Hankel inv(W_Hankel.T X_Hankel.T X_Hankel W_Hankel)

    Inputs:
    X: observations (one subject) TxD
    W_Hankel: filters/backward models DLxK
    L: time lag (if temporal-spatial)
    causal: causal or non-causal filter

    Output:
    F: forward model
    '''
    if L == 1:
        Rxx = np.cov(X, rowvar=False)
        F = Rxx@W_Hankel@LA.inv(W_Hankel.T@Rxx@W_Hankel)
    else:
        X_block_Hankel = block_Hankel(X, L, causal)
        F = X.T@X_block_Hankel@W_Hankel@LA.inv(W_Hankel.T@X_block_Hankel.T@X_block_Hankel@W_Hankel)
    return F


def rescale(W, Dxx):
    '''
    LEGACY
    To make w_n^H R_{xn xn} w_n = 1 for all n. Then the denominators of correlation coefficients between every pairs are the same.
    '''
    _, N, n_componets = W.shape
    for i in range(n_componets):
        W_split = np.split(W[:,:,i], N, axis=1)
        W_blkdiag = scipy.sparse.block_diag(W_split)
        scales = np.diag(np.transpose(W_blkdiag)@Dxx@W_blkdiag)
        W[:,:,i] = W[:,:,i]/np.sqrt(scales)
    return W


def avg_corr_coe(X_stack, W, L, causal, n_components=5):
    '''
    Calculate the pairwise average correlation.
    Inputs:
    X_stack: stacked (along axis 2) data of different subjects
    W: weights 1) dim(W)=2: results of correlated component analysis 2) dim(W)=3: results of GCCA
    L: number of taps (if we use spatial-temporal filter)
    causal: whether the filter is a causal filter
    n_components: number of components
    Output:
    avg_corr: average pairwise correlation
    '''
    _, _, N = X_stack.shape
    Hankellist = [np.expand_dims(block_Hankel(X_stack[:,:,n], L, causal), axis=2) for n in range(N)]
    X_stack = np.concatenate(tuple(Hankellist), axis=2)
    avg_corr = np.zeros(n_components)
    if np.ndim (W) == 2:
        W = np.expand_dims(W, axis=1)
        W = np.repeat(W, N, axis=1)
    for component in range(n_components):
        w = W[:,:,component]
        w = np.expand_dims(w, axis=1)
        X_trans = np.einsum('tdn,dln->tln', X_stack, w)
        X_trans = np.squeeze(X_trans, axis=1)
        corr_mtx = np.corrcoef(X_trans, rowvar=False)
        avg_corr[component] = np.sum(corr_mtx-np.eye(N))/N/(N-1)
    return avg_corr


def avg_corr_coe_multi_modal(datalist, Wlist, Llist, causal_list, n_components=5, regularization=None, ISC=True):
    '''
    Calculate the pairwise average correlation.
    Inputs:
    datalist: data of different modalities (a list) E.g, [EEG_stack, Stim]
    Wlist: weights of different modalities (a list)
    Llist: number of taps of different modalities.
    causal_list: causal filter or not
    n_components: number of components
    Output:
    avg_corr: average pairwise correlation
    '''
    if ISC and (np.ndim(datalist[0]) != 2): # calculate avg correlation across only EEG views, unless there is only one EEG subject (CCA)
        avg_corr = avg_corr_coe(datalist[0], Wlist[0], Llist[0], causal_list[0], n_components=n_components)
    else:
        avg_corr = np.zeros(n_components)
        n_mod = len(datalist)
        for component in range(n_components):
            X_trans_list = []
            for i in range(n_mod):
                W = Wlist[i]
                rawdata = datalist[i]
                L = Llist[i]
                causal = causal_list[i]
                if np.ndim(W) == 3:
                    w = W[:,:,component]
                    w = np.expand_dims(w, axis=1)
                    data_trans = [np.expand_dims(block_Hankel(rawdata[:,:,n],L,causal),axis=2) for n in range(rawdata.shape[2])]
                    data = np.concatenate(tuple(data_trans), axis=2)
                    X_trans = np.einsum('tdn,dln->tln', data, w)
                    X_trans = np.squeeze(X_trans, axis=1)
                if np.ndim(W) == 2:
                    w = W[:,component]
                    data = block_Hankel(rawdata,L,causal)
                    X_trans = data@w
                    X_trans = np.expand_dims(X_trans, axis=1)
                X_trans_list.append(X_trans)
            X_trans_all = np.concatenate(tuple(X_trans_list), axis=1)
            if regularization=='lwcov':
                cov_mtx = LedoitWolf().fit(X_trans_all).covariance_
                cov_diag = np.expand_dims(np.diag(cov_mtx).shape, axis = 1)
                corr_mtx = cov_mtx/np.sqrt(cov_diag)/np.sqrt(cov_diag.T)
            else:
                corr_mtx = np.corrcoef(X_trans_all, rowvar=False)
            N = X_trans_all.shape[1]
            avg_corr[component] = np.sum(corr_mtx-np.eye(N))/N/(N-1)
    return avg_corr


def cross_val_CCA(EEG_list, feature_list, fs, L_EEG=1, L_feat=1, causalx=False, causaly=True, fold=10, n_components=5, regularization='lwcov', K_regu=None, message=True, signifi_level=True, pool=True):
    corr_train = np.zeros((fold, n_components))
    corr_test = np.zeros((fold, n_components))
    for idx in range(fold):
        # EEG_train, EEG_test, Sti_train, Sti_test = split(EEG_list, feature_list, fold=fold, fold_idx=idx+1)
        EEG_train, EEG_test, Sti_train, Sti_test = split_balance(EEG_list, feature_list, fold=fold, fold_idx=idx+1)
        corr_train[idx,:], _, V_A_train, V_B_train, Lam = cano_corr(EEG_train, Sti_train, Lx=L_EEG, Ly=L_feat, causalx=causalx, causaly=causaly, n_components=n_components, regularization=regularization, K_regu=K_regu)
        corr_test[idx,:], _, _, _, _ = cano_corr(EEG_test, Sti_test, Lx=L_EEG, Ly=L_feat, causalx=causalx, causaly=causaly, n_components=n_components, regularization=regularization, K_regu=K_regu, V_A=V_A_train, V_B=V_B_train, Lam=Lam)
    if signifi_level:
        if pool:
            corr_trials = permutation_test(EEG_test, Sti_test, Lx=L_EEG, Ly=L_feat, causalx=causalx, causaly=causaly, num_test=1000, block_len=1, n_components=n_components, regularization=regularization, K_regu=K_regu, V_A=V_A_train, V_B=V_B_train, Lam=Lam)
            corr_trials = np.sort(abs(corr_trials), axis=None)
            print('Significance level: {}'.format(corr_trials[-50*n_components])) # top 5%
        else:
            corr_trials = permutation_test(EEG_test, Sti_test, Lx=L_EEG, Ly=L_feat, causalx=causalx, causaly=causaly, num_test=1000, block_len=20*fs, n_components=n_components, regularization=regularization, K_regu=K_regu, V_A=V_A_train, V_B=V_B_train, Lam=Lam)
            corr_trials = np.sort(abs(corr_trials), axis=0)
            print('Significance level of each component: {}'.format(corr_trials[-50,:])) # top 5%
    if message:
        print('Average correlation coefficients of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train, axis=0)))
        print('Average correlation coefficients of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test, axis=0)))
    return corr_train, corr_test, V_A_train, V_B_train


def cross_val_GCCA(nested_data, L, causal, fs, fold=10, n_components=5, regularization='lwcov', message=True, signifi_level=True, ISC=True, pool=True):
    corr_train = np.zeros((fold, n_components))
    corr_test = np.zeros((fold, n_components))
    for idx in range(fold):
        train_list, test_list = split_mm_balance([nested_data], fold=fold, fold_idx=idx+1)
        W_train, F_train, _ = GCCA(train_list[0], L, causal, n_components=n_components, regularization=regularization)
        corr_train[idx,:] = avg_corr_coe(train_list[0], W_train, L, causal, n_components=n_components)
        corr_test[idx,:] = avg_corr_coe(test_list[0], W_train, L, causal, n_components=n_components)
    if signifi_level:
        if pool:
            corr_trials = permutation_test_GCCA(test_list, [L], [causal], num_test=1000, block_len=1, n_components=n_components, Wlist=[W_train], ISC=ISC)
            corr_trials = np.sort(abs(corr_trials), axis=None)
            print('Significance level: {}'.format(corr_trials[-50*n_components]))
        else:
            corr_trials = permutation_test_GCCA(test_list, [L], [causal], num_test=1000, block_len=20*fs, n_components=n_components, Wlist=[W_train], ISC=ISC)
            corr_trials = np.sort(abs(corr_trials), axis=0)
            print('Significance level of each component: {}'.format(corr_trials[-50,:])) # top 5%
    if message:
        print('Average correlation coefficients of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train, axis=0)))
        print('Average correlation coefficients of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test, axis=0)))
    return corr_train, corr_test, W_train, F_train


def cross_val_SI_GCCA(nested_datalist, Llist, causal_list, rho, fs, fold=10, n_components=5, regularization='lwcov', message=True, signifi_level=True, ISC=True, pool=True):
    corr_train = np.zeros((fold, n_components))
    corr_test = np.zeros((fold, n_components))
    for idx in range(fold):
        train_list, test_list = split_mm_balance(nested_datalist, fold=fold, fold_idx=idx+1)
        Wlist_train, F_train, _ = SI_GCCA(train_list, Llist, causal_list, n_components=n_components, rho=rho, regularization=regularization)
        corr_train[idx,:] = avg_corr_coe_multi_modal(train_list, Wlist_train, Llist, causal_list, n_components=n_components, ISC=ISC)
        corr_test[idx,:] = avg_corr_coe_multi_modal(test_list, Wlist_train, Llist, causal_list, n_components=n_components, ISC=ISC)
    if signifi_level:
        if pool:
            corr_trials = permutation_test_GCCA(test_list, Llist, causal_list, num_test=1000, block_len=1, n_components=n_components, Wlist=Wlist_train, ISC=ISC)
            corr_trials = np.sort(abs(corr_trials), axis=None)
            print('Significance level: {}'.format(corr_trials[-50*n_components]))
        else:
            corr_trials = permutation_test_GCCA(test_list, Llist, causal_list, num_test=1000, block_len=20*fs, n_components=n_components, Wlist=Wlist_train, ISC=ISC)
            corr_trials = np.sort(abs(corr_trials), axis=0)
            print('Significance level of each component: {}'.format(corr_trials[-50,:])) # top 5%
    if message:
        print('Average correlation coefficients of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train, axis=0)))
        print('Average correlation coefficients of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test, axis=0)))
    return corr_train, corr_test, Wlist_train, F_train


def cross_val_GCCA_multi_mod(nested_datalist, Llist, causal_list, rhos, fs, fold=10, n_components=5, regularization='lwcov', message=True, signifi_level=True, ISC=True, pool=True):
    corr_train = np.zeros((fold, n_components))
    corr_test = np.zeros((fold, n_components))
    for idx in range(fold):
        # train_list, test_list = split_multi_mod(datalist, fold=fold, fold_idx=idx+1)
        train_list, test_list = split_mm_balance(nested_datalist, fold=fold, fold_idx=idx+1)
        Wlist_train, Flist_train, _ = GCCA_multi_modal(train_list, Llist, causal_list, n_components=n_components, rhos=rhos, regularization=regularization)
        corr_train[idx,:] = avg_corr_coe_multi_modal(train_list, Wlist_train, Llist, causal_list, n_components=n_components, ISC=ISC)
        corr_test[idx,:] = avg_corr_coe_multi_modal(test_list, Wlist_train, Llist, causal_list, n_components=n_components, ISC=ISC)
    if signifi_level:
        if pool:
            corr_trials = permutation_test_GCCA(test_list, Llist, causal_list, num_test=1000, block_len=1, n_components=n_components, Wlist=Wlist_train, ISC=ISC)
            corr_trials = np.sort(abs(corr_trials), axis=None)
            print('Significance level: {}'.format(corr_trials[-50*n_components]))
        else:
            corr_trials = permutation_test_GCCA(test_list, Llist, causal_list, num_test=1000, block_len=20*fs, n_components=n_components, Wlist=Wlist_train, ISC=ISC)
            corr_trials = np.sort(abs(corr_trials), axis=0)
            print('Significance level of each component: {}'.format(corr_trials[-50,:])) # top 5%
    if message:
        print('Average correlation coefficients of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train, axis=0)))
        print('Average correlation coefficients of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test, axis=0)))
    return corr_train, corr_test, Wlist_train, Flist_train


def shuffle_block(X, block_len):
    '''
    Shuffle the blocks of X along the time axis for each subject.
    '''
    T, D, N = X.shape
    if T%block_len != 0:
        append_arr = np.zeros((block_len-T%block_len, D, N))
        X = np.concatenate((X, append_arr), axis=0)
    T_appended = X.shape[0]
    X_shuffled = np.zeros_like(X)
    for n in range(N):
        blocks = [X[i:i+block_len, :, n] for i in range(0, T_appended, block_len)]
        random.shuffle(blocks)
        X_shuffled[:,:,n] = np.concatenate(tuple(blocks), axis=0)
    return X_shuffled


def shuffle_2D(X, block_len):
    T, D = X.shape
    if T%block_len != 0:
        append_arr = np.zeros((block_len-T%block_len, D))
        X = np.concatenate((X, append_arr), axis=0)
        T, _ = X.shape
    X_block = X.reshape((T//block_len, block_len, D))
    X_shuffle_block = np.random.permutation(X_block)
    X_shuffle = X_shuffle_block.reshape((T, D))
    return X_shuffle


def shuffle_3D(X, block_len):
    '''
    Same as shuffle_block(X, block_len)
    '''
    T, D, N = X.shape
    if T%block_len != 0:
        append_arr = np.zeros((block_len-T%block_len, D, N))
        X = np.concatenate((X, append_arr), axis=0)
    X_shuffled = np.zeros_like(X)
    for n in range(N):
        X_shuffled[:,:,n] = shuffle_2D(X[:,:,n], block_len)
    return X_shuffled


def permutation_test(X, Y, Lx, Ly, causalx, causaly, num_test, block_len, n_components, regularization, K_regu, V_A, V_B, Lam):
    corr_coe_topK = np.zeros((num_test, n_components))
    for i in tqdm(range(num_test)):
        X_shuffled = shuffle_2D(X, block_len)
        Y_shuffled = shuffle_2D(Y, block_len)
        corr_coe_topK[i,:], _, _, _, _ = cano_corr(X_shuffled, Y_shuffled, Lx=Lx, Ly=Ly, causalx=causalx, causaly=causaly, n_components=n_components, regularization=regularization, K_regu=K_regu, V_A=V_A, V_B=V_B, Lam=Lam)
    return corr_coe_topK


def permutation_test_GCCA(datalist, Llist, causal_list, num_test, block_len, n_components, Wlist, ISC):
    corr_coe_topK = np.empty((0, n_components))
    for i in tqdm(range(num_test)):
        datalist_shuffled = []
        for data in datalist:
            if np.ndim(data) == 2:
                datalist_shuffled.append(shuffle_2D(data, block_len))
            elif np.ndim(data) == 3:
                datalist_shuffled.append(shuffle_3D(data, block_len))
        corr_coe = avg_corr_coe_multi_modal(datalist_shuffled, Wlist, Llist, causal_list, n_components=n_components, ISC=ISC)
        corr_coe_topK = np.concatenate((corr_coe_topK, np.expand_dims(corr_coe[:n_components], axis=0)), axis=0)
    return corr_coe_topK


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


def rho_sweep(nested_datalist, sweep_list, Llist, causal_list, fs, fold=10, n_components=5, message=False, ISC=True, iflist=False):
    corr_best = -np.Inf
    for i in sweep_list:
        if iflist:
            rho = [1, 10**i]
            corr_train, corr_test, _, _ = cross_val_GCCA_multi_mod(nested_datalist, Llist, causal_list, rho, fs, fold, n_components, regularization='lwcov', message=False, signifi_level=False, ISC=ISC)
        else:
            rho = 10**i
            corr_train, corr_test, _, _ = cross_val_SI_GCCA(nested_datalist, Llist, causal_list, rho, fs, fold, n_components, regularization='lwcov', message=False, signifi_level=False, ISC=ISC)
        avg_corr_train = np.average(corr_train, axis=0)
        avg_corr_test = np.average(corr_test, axis=0)
        if message:
            print('Average ISC across different training sets when rho=10**{}: {}'.format(i, avg_corr_train))
            print('Average ISC across different test sets when rho=10**{}: {}'.format(i, avg_corr_test))
        if max(avg_corr_test) > corr_best:
            rho_best = rho
            corr_best = max(avg_corr_test)
    return rho_best


def EEG_normalization(data, len_seg):
    '''
    Normalize the EEG data.
    Subtract data of each channel by the mean of it
    Divide data into several segments, and for each segment, divide the data matrix by its Frobenius norm.
    Inputs:
    data: EEG data D x T
    len_seg: length of the segments
    Output:
    normalized_data
    '''
    _, T = data.shape
    n_blocks = T // len_seg + 1
    data_blocks = np.array_split(data, n_blocks, axis=1)
    data_zeromean = [db - np.mean(db, axis=1, keepdims=True) for db in data_blocks]
    normalized_blocks = [db/LA.norm(db) for db in data_zeromean]
    normalized_data = np.concatenate(tuple(normalized_blocks), axis=1)
    return normalized_data


def extract_highfreq(EEG, resamp_freqs, band=[15,20], noramalize=True):
    '''
    EEG signals -> band-pass filter -> high-frequency signals -> Hilbert transform -> signal envelope -> low-pass filter -> down-sampled envelope -> noramalized envelope
    Inputs:
    EEG: EEG signals with original sampling rate
    resamp_freqs: resampling frequency
    band: the frequency band to be kept
    Outputs:
    envelope: the envelope of high-frequency signals
    '''
    EEG_band = EEG.filter(l_freq=band[0], h_freq=band[1], picks=['eeg'])
    envelope = EEG_band.apply_hilbert(picks=['eeg'], envelope=True)
    envelope = envelope.resample(sfreq=resamp_freqs)
    if noramalize:
        eeg_channel_indices = mne.pick_types(envelope.info, eeg=True)
        eegdata, _ = envelope[eeg_channel_indices]
        envelope._data[eeg_channel_indices, :] = EEG_normalization(eegdata, resamp_freqs*60)
    return envelope


def preprocessing(file_path, HP_cutoff = 0.5, AC_freqs=50, band=None, resamp_freqs=None, bads=[], eog=True, regression=True, normalize=True):
    '''
    Preprocessing of the raw signal
    Re-reference -> Highpass filter (-> downsample)
    No artifact removal technique has been applied yet
    Inputs:
    file_path: location of the eeg dataset
    HP_cutoff: cut off frequency of the high pass filter (for removing DC components and slow drifts)
    AC_freqs: AC power line frequency
    resamp_freqs: resampling frequency (if None then resampling is not needed)
    bads: list of bad channels
    eog: if contains 4 eog channels
    regression: whether regresses eog out
    Output:
    preprocessed: preprocessed eeg
    fs: the sample frequency of the EEG signal (original or down sampled)
    '''
    raw_lab = mne.io.read_raw_eeglab(file_path, preload=True)
    raw_lab.info['bads'] = bads
    fsEEG = raw_lab.info['sfreq']
    # Rename channels and set montages
    biosemi_layout = mne.channels.read_layout('biosemi')
    ch_names_map = dict(zip(raw_lab.info['ch_names'], biosemi_layout.names))
    raw_lab.rename_channels(ch_names_map)
    montage = mne.channels.make_standard_montage('biosemi64')
    raw_lab.set_montage(montage)
    if len(bads)>0:
        # Interpolate bad channels
        raw_lab.interpolate_bads()
    # Re-reference
    # raw_lab.set_eeg_reference(ref_channels=['Cz']) # Select the reference channel to be Cz
    raw_lab.set_eeg_reference(ref_channels='average')
    # If there are EOG channels, first treat them as EEG channels and do re-referencing, filtering and resampling.
    if eog:
        misc_names = [raw_lab.info.ch_names[i] for i in mne.pick_types(raw_lab.info, misc=True)]
        eog_data, _ = raw_lab[misc_names]
        eog_channel_indices = mne.pick_channels(raw_lab.info['ch_names'], include=misc_names)
        type_eeg = ['eeg']*len(misc_names)
        change_type_dict = dict(zip(misc_names, type_eeg))
        raw_lab.set_channel_types(change_type_dict)
        # Take the average of four EOG channels as the reference
        raw_lab._data[eog_channel_indices, :] = eog_data - np.average(eog_data, axis=0)
    # Highpass filter - remove DC components and slow drifts
    raw_highpass = raw_lab.copy().filter(l_freq=HP_cutoff, h_freq=None)
    # raw_highpass.compute_psd().plot(average=True)
    # Remove power line noise
    raw_notch = raw_highpass.copy().notch_filter(freqs=AC_freqs)
    # raw_notch.compute_psd().plot(average=True)
    # Then set EOG channels to their true type
    if eog:
        type_true = ['eog']*len(misc_names)
        change_type_dict = dict(zip(misc_names, type_true))
        raw_notch.set_channel_types(change_type_dict)
    if regression:
        EOGweights = mne.preprocessing.EOGRegression(picks='eeg', proj=False).fit(raw_notch)
        raw_notch = EOGweights.apply(raw_notch, copy=False)
    # Resampling:
    # Anti-aliasing has been implemented in mne.io.Raw.resample before decimation
    # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.resample
    if resamp_freqs is not None:
        if band is not None:
            highfreq = extract_highfreq(raw_notch.copy(), resamp_freqs, band, normalize)
        else:
            highfreq = None
        raw_downsampled = raw_notch.copy().resample(sfreq=resamp_freqs)
        # raw_downsampled.compute_psd().plot(average=True)
        preprocessed = raw_downsampled
        fs = resamp_freqs
    else:
        highfreq = None
        preprocessed = raw_notch
        fs = fsEEG
    if normalize:
        eeg_channel_indices = mne.pick_types(preprocessed.info, eeg=True)
        eegdata, _ = preprocessed[eeg_channel_indices]
        preprocessed._data[eeg_channel_indices, :] = EEG_normalization(eegdata, fs*60)
    return preprocessed, fs, highfreq


def name_paths(eeg_path_head, feature_path_head):
    '''
    Find the name of the videos and the paths of the corresponding eeg signals and features
    Inputs:
    eeg_path_head: relative path of the eeg folder
    feature_path_head: relative path of the feature folder
    Output:
    videonames: names of the video
    eeg_sets_paths: relative paths of the eeg sets
    feature_sets_paths: relative paths of the feature sets
    '''
    eeg_list = os.listdir(eeg_path_head)
    eeg_sets = [i for i in eeg_list if i.endswith('.set')]
    eeg_sets_paths = [eeg_path_head+i for i in eeg_sets]
    feature_list = os.listdir(feature_path_head)
    feature_sets = [i for i in feature_list if i.endswith('.mat')]
    feature_sets_paths = [feature_path_head+i for i in feature_sets]
    videonames = [i[:-4] for i in eeg_sets]
    return videonames, eeg_sets_paths, feature_sets_paths


def load_eeg_feature(idx, videonames, eeg_sets_paths, feature_sets_paths, feature_type='muFlow', bads=[], eog=False, regression=False):
    '''
    Load the features and eeg signals of a specific dataset
    Inputs:
    idx: the index of the wanted dataset
    videonames: names of the video
    eeg_sets_paths: relative paths of the eeg sets
    feature_sets_paths: relative paths of the feature sets
    Outputs:
    eeg_downsampled: down sampled (and preprocessed) eeg signals
    normalized_features: normalized features 
    times: time axis 
    fsStim: sample rate of both stimulus and eeg signals
    '''
    # Load features and EEG signals
    videoname = videonames[idx]
    matching = [s for s in feature_sets_paths if videoname in s]
    assert len(matching) == 1
    features_data = scipy.io.loadmat(matching[0])
    fsStim = int(features_data['fsVideo']) # fs of the video 
    features = np.nan_to_num(features_data[feature_type]) # feature: optical flow
    eeg_prepro, _ = preprocessing(eeg_sets_paths[idx], HP_cutoff = 0.5, AC_freqs=50, resamp_freqs=fsStim, bads=bads, eog=eog, regression=regression)
    # Clip data
    eeg_channel_indices = mne.pick_types(eeg_prepro.info, eeg=True)
    eeg_downsampled, times = eeg_prepro[eeg_channel_indices]
    if len(features) > len(times):
        features = features[:len(times)]
    else:
        times = times[:len(features)]
        eeg_downsampled = eeg_downsampled[:,:len(features)]
    eeg_downsampled = eeg_downsampled.T
    # normalized_features = features/LA.norm(features)  # normalize features
    fs = fsStim
    # export_path = eeg_sets_paths[idx][:-4] + '.mat'
    # scipy.io.savemat(export_path, {'eeg'+videoname: eeg_downsampled.T, 'fs': fs})
    return eeg_downsampled, features, times, fs


def concatenate_eeg_feature(videonames, eeg_sets_paths, feature_sets_paths, feature_type='muFlow', bads=[], eog=False, regression=False):
    eeg_downsampled_list = []
    features_list = []
    for idx in range(len(videonames)):
        eeg_downsampled, features, _, fs = load_eeg_feature(idx, videonames, eeg_sets_paths, feature_sets_paths, feature_type, bads, eog, regression)
        eeg_downsampled_list.append(eeg_downsampled)
        features_list.append(features)
    eeg_concat = np.concatenate(eeg_downsampled_list, axis=0)
    features_concat = np.concatenate(features_list)
    times = np.array(range(len(features_concat)))/fs
    return eeg_concat, features_concat, times, fs


def load_eeg_env(idx, audionames, eeg_sets_paths, env_sets_paths, resamp_freq=20, band=[2, 9]):
    # Load features and EEG signals
    audioname = audionames[idx]
    matching = [s for s in env_sets_paths if audioname in s]
    assert len(matching) == 1
    envelope = np.squeeze(scipy.io.loadmat(matching[0])['envelope'])
    eeg_prepro, fsEEG = preprocessing(eeg_sets_paths[idx], HP_cutoff = 0.5, AC_freqs=50)
    # Clip data
    eeg_channel_indices = mne.pick_types(eeg_prepro.info, eeg=True)
    eeg, times = eeg_prepro[eeg_channel_indices]
    if len(envelope) > len(times):
        envelope = envelope[:len(times)]
    else:
        eeg = eeg[:,:len(envelope)]
    # Band-pass and down sample
    sos_bp = signal.butter(4, band, 'bandpass', output='sos', fs=fsEEG)
    eeg_filtered = signal.sosfilt(sos_bp, eeg)
    env_filtered = signal.sosfilt(sos_bp, envelope)
    eeg_downsampled = signal.resample_poly(eeg_filtered, resamp_freq, fsEEG, axis=1)
    env_downsampled = signal.resample_poly(env_filtered, resamp_freq, fsEEG)
    eeg_downsampled = eeg_downsampled.T
    times = np.array(range(len(eeg_downsampled)))/resamp_freq
    return eeg_downsampled, env_downsampled, times


def concatenate_eeg_env(audionames, eeg_sets_paths, env_sets_paths, resamp_freq=20, band=[2, 9]):
    eeg_downsampled_list = []
    env_downsampled_list = []
    for idx in range(len(audionames)):
        eeg_downsampled, env_downsampled, _ = load_eeg_env(idx, audionames, eeg_sets_paths, env_sets_paths, resamp_freq, band)
        eeg_downsampled_list.append(eeg_downsampled)
        env_downsampled_list.append(env_downsampled)
    # TODO: Do we need to normalize eeg signals when concatenating them?
    eeg_concat = np.concatenate(eeg_downsampled_list, axis=0)
    env_concat = np.concatenate(env_downsampled_list)
    times = np.array(range(len(env_concat)))/resamp_freq
    return eeg_concat, env_concat, times


def multisub_data_org(subjects, video, folder='EOG', feature_type=['muFlow'], bads=[], eog=False, regression=False, normalize=True):
    feature_path = '../../Experiments/Videos/stimuli/' + video + '_features.mat'
    features_data = scipy.io.loadmat(feature_path)
    fsStim = int(features_data['fsVideo']) # fs of the video 
    features_list = [np.abs(np.nan_to_num(features_data[type])) for type in feature_type]
    features = np.concatenate(tuple(features_list), axis=1)
    # features = features/LA.norm(features) # Normalize here or normalize the concatenated features?
    T = features.shape[0]
    eeg_list = []
    for sub in subjects:
        eeg_path = '../../Experiments/data/'+ sub +'/' + folder + '/' + video + '.set'
        eeg_prepro, fs, _ = preprocessing(eeg_path, HP_cutoff = 0.5, AC_freqs=50, band=None, resamp_freqs=fsStim, bads=bads, eog=eog, regression=regression, normalize=normalize)
        eeg_channel_indices = mne.pick_types(eeg_prepro.info, eeg=True)
        eeg_downsampled, _ = eeg_prepro[eeg_channel_indices]
        eeg_downsampled = eeg_downsampled.T
        eeg_list.append(eeg_downsampled)
        if eeg_downsampled.shape[0] < T:
            T = eeg_downsampled.shape[0]
    # Clip data
    features = features[:T, :]
    eeg_list = [np.expand_dims(eeg[:T,:], axis=2) for eeg in eeg_list]
    eeg_multisub = np.concatenate(tuple(eeg_list), axis=2)
    times = np.array(range(T))/fs
    return features, eeg_multisub, fs, times