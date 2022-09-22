import numpy as np
import random
import os
import scipy.io
from tqdm import tqdm
from numpy import linalg as LA
from scipy import signal
from scipy.linalg import toeplitz
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


def cano_corr(X, Y, K_regu=7):
    '''
    Input:
    X: EEG data T(#sample)xD(#channel)
    Y: Stimulus T(#sample)xL(#tap)
    '''
    _, D = X.shape
    _, L = Y.shape
    covXY = np.cov(X, Y, rowvar=False)
    Rx = covXY[:D,:D]
    Ry = covXY[D:D+L,D:D+L]
    Rxy = covXY[:D,D:D+L]
    Ryx = covXY[D:D+L,:D]
    # if LA.matrix_rank(Rx) < D:
    #     invRx = PCAreg_inv(Rx, LA.matrix_rank(Rx))
    # else:
    #     invRx = LA.inv(Rx)
    # if LA.matrix_rank(Ry) < L:
    #     invRy = PCAreg_inv(Ry, LA.matrix_rank(Ry))
    # else:
    #     invRy = LA.inv(Ry)
    invRx = PCAreg_inv(Rx, K_regu)
    invRy = PCAreg_inv(Ry, K_regu)
    A = invRx@Rxy@invRy@Ryx
    B = invRy@Ryx@invRx@Rxy
    # lam of A and lam of B should be the same
    # can be used as a preliminary check for correctness
    # can't tell neg or pos, therefore not used
    _, V_A = eig_sorted(A)
    _, V_B = eig_sorted(B)
    # corr_coe = np.sqrt(lam[:K_regu]) # wrong 
    # _, V_A = LA.eig(A)
    # _, V_B = LA.eig(B)
    X_trans = X@np.real(V_A[:,:K_regu])
    Y_trans = Y@np.real(V_B[:,:K_regu])
    corr_pvalue = [pearsonr(X_trans[:,k], Y_trans[:,k]) for k in range(K_regu)]
    corr_coe = np.array([corr_pvalue[k][0] for k in range(K_regu)])
    # idx = np.argsort(-np.abs(corr_coe))
    # corr_coe = corr_coe[idx]
    # null hypothesis that the distributions underlying the samples are uncorrelated and normally distributed.
    p_value = np.array([corr_pvalue[k][1] for k in range(K_regu)]) 
    # p_value = p_value[idx]
    return corr_coe, p_value, V_A[:,:K_regu], V_B[:,:K_regu]


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


def permutation_test(X, Y, num_test, t, fs, topK):
    corr_coe_topK = np.empty((0, topK))
    X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=(1,2))
    for i in tqdm(range(num_test)):
        X_shuffled = np.squeeze(shuffle_block(X, t, fs), axis=2)
        Y_shuffled = np.squeeze(shuffle_block(Y, t, fs), axis=(1,2))
        L_timefilter = fs
        conv_mtx_shuffled = convolution_mtx(L_timefilter, Y_shuffled)
        corr_coe, _, _, _ = cano_corr(X_shuffled, conv_mtx_shuffled)
        corr_coe_topK = np.concatenate((corr_coe_topK, np.expand_dims(corr_coe[:topK], axis=0)), axis=0)
    return corr_coe_topK


def leave_one_fold_out(EEG, Sti, L_timefilter, K_regu=7, fold=10, fold_idx=1):
    EEG_train, EEG_test, Sti_train, Sti_test = split(EEG, Sti, fold=fold, fold_idx=fold_idx)
    conv_mtx_train = convolution_mtx(L_timefilter, Sti_train)
    corr_coe_train, _, V_A_train, V_B_train = cano_corr(EEG_train, conv_mtx_train)
    conv_mtx_test = convolution_mtx(L_timefilter, Sti_test)
    EEG_test_trans = EEG_test@np.real(V_A_train)
    conv_mtx_test_trans = conv_mtx_test@np.real(V_B_train)
    corr_pvalue = [pearsonr(EEG_test_trans[:,k], conv_mtx_test_trans[:,k]) for k in range(K_regu)]
    return corr_coe_train, corr_pvalue