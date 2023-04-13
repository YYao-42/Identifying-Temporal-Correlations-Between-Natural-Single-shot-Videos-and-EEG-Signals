import numpy as np
import copy
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
from numpy import linalg as LA
from scipy.linalg import eig, eigh, sqrtm, lstsq
from scipy.stats import pearsonr
import utils


class BackwardLeastSquares:
    def __init__(self, EEG_list, Stim_list, fs, L_EEG, offset_EEG, fold=10, message=True):
        self.EEG_list = EEG_list
        self.Stim_list = Stim_list
        self.fs = fs
        self.L_EEG = L_EEG
        self.offset_EEG = offset_EEG
        self.fold = fold
        self.message = message

    def decoder(self, EEG, stim, W=None):
        '''
        Inputs:
        EEG: T(#sample)xD_eeg(#channel) array
        stim: T(#sample)xD_stim(#feature dim) array
        W: only used in test mode
        Output:
        W: (D_eeg*L_eeg)xD_stim(#feature dim)x array
        '''
        EEG_Hankel = utils.block_Hankel(EEG, self.L_EEG, self.offset_EEG)
        if W is not None: # Test Mode
            pass
        else:
            W = lstsq(EEG_Hankel, stim)[0]
        filtered_EEG = EEG_Hankel@W
        mse = np.mean((filtered_EEG-stim)**2)
        return W, mse

    def cross_val(self):
        fold = self.fold
        mse_train = np.zeros((fold, 1))
        mse_test = np.zeros((fold, 1))
        for idx in range(fold):
            EEG_train, EEG_test, Sti_train, Sti_test = utils.split_balance(self.EEG_list, self.Stim_list, fold=fold, fold_idx=idx+1)
            W_train, mse_train[idx] = self.decoder(EEG_train, Sti_train)
            _, mse_test[idx] = self.decoder(EEG_test, Sti_test, W=W_train)
        if self.message:
            print('Average mse across {} training folds: {}'.format(self.fold, np.average(mse_train)))
            print('Average mse across {} test folds: {}'.format(self.fold, np.average(mse_test)))
        return mse_train, mse_test, W_train


class CanonicalCorrelationAnalysis:
    def __init__(self, EEG_list, Stim_list, fs, L_EEG, L_Stim, offset_EEG=0, offset_Stim=0, fold=10, n_components=5, regularization='lwcov', K_regu=None, message=True, signifi_level=True, pool=True, n_permu=1000, p_value=0.05):
        '''
        EEG_list: list of EEG data, each element is a T(#sample)xDx(#channel) array
        Stim_list: list of stimulus, each element is a T(#sample)xDy(#feature dim) array
        fs: Sampling rate
        L_EEG/L_Stim: If use (spatial-) temporal filter, the number of taps
        offset_EEG/offset_Stim: If use (spatial-) temporal filter, the offset of time lags
        n_components: Number of components to be returned
        regularization: Regularization of the estimated covariance matrix
        K_regu: Number of eigenvalues to be kept. Others will be set to zero. Keep all if K_regu=None
        V_A/V_B: Filters of X and Y. Use only in test mode.
        Lam: Eigenvalues. Use only in test mode.
        '''
        self.EEG_list = EEG_list
        self.Stim_list = Stim_list
        self.fs = fs
        self.L_EEG = L_EEG
        self.L_Stim = L_Stim
        self.offset_EEG = offset_EEG
        self.offset_Stim = offset_Stim
        self.fold = fold
        self.n_components = n_components
        self.regularization = regularization
        self.K_regu = K_regu
        self.message = message
        self.signifi_level = signifi_level
        self.pool = pool
        self.n_permu = n_permu
        self.p_value = p_value

    def cano_corr(self, X, Y, V_A=None, V_B=None, Lam=None):
        if np.ndim(Y) == 1:
            Y = np.expand_dims(Y, axis=1)
        _, Dx = X.shape
        _, Dy = Y.shape
        Lx = self.L_EEG
        Ly = self.L_Stim
        n_components = self.n_components
        mtx_X = utils.block_Hankel(X, Lx, self.offset_EEG)
        mtx_Y = utils.block_Hankel(Y, Ly, self.offset_Stim)
        if V_A is not None: # Test Mode
            flag_test = True
        else: # Train mode
            flag_test = False
            # compute covariance matrices
            covXY = np.cov(mtx_X, mtx_Y, rowvar=False)
            if self.regularization=='lwcov':
                Rx = LedoitWolf().fit(mtx_X).covariance_
                Ry = LedoitWolf().fit(mtx_Y).covariance_
            else:
                Rx = covXY[:Dx*Lx,:Dx*Lx]
                Ry = covXY[Dx*Lx:Dx*Lx+Dy*Ly,Dx*Lx:Dx*Lx+Dy*Ly]
            Rxy = covXY[:Dx*Lx,Dx*Lx:Dx*Lx+Dy*Ly]
            Ryx = covXY[Dx*Lx:Dx*Lx+Dy*Ly,:Dx*Lx]
            # PCA regularization is recommended (set K_regu<rank(Rx))
            # such that the small eigenvalues dominated by noise are discarded
            if self.K_regu is None:
                invRx = utils.PCAreg_inv(Rx, LA.matrix_rank(Rx))
                invRy = utils.PCAreg_inv(Ry, LA.matrix_rank(Ry))
            else:
                K_regu = min(LA.matrix_rank(Rx), LA.matrix_rank(Ry), self.K_regu)
                invRx = utils.PCAreg_inv(Rx, K_regu)
                invRy = utils.PCAreg_inv(Ry, K_regu)
            A = invRx@Rxy@invRy@Ryx
            B = invRy@Ryx@invRx@Rxy
            # lam of A and lam of B should be the same
            # can be used as a preliminary check for correctness
            # the correlation coefficients are already available by taking sqrt of the eigenvalues: corr_coe = np.sqrt(lam[:K_regu])
            # or we do the following to obtain transformed X and Y and calculate corr_coe from there
            Lam, V_A = utils.eig_sorted(A)
            _, V_B = utils.eig_sorted(B)
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
        TSC = np.sum(np.square(corr_coe[:2]))
        ChDist = np.sqrt(2-TSC)
        return corr_coe, ChDist, p_value, V_A, V_B, Lam

    def permutation_test(self, X, Y, V_A, V_B, Lam, block_len):
        corr_coe_topK = np.zeros((self.n_permu, self.n_components))
        for i in tqdm(range(self.n_permu)):
            X_shuffled = utils.shuffle_2D(X, block_len)
            Y_shuffled = utils.shuffle_2D(Y, block_len)
            corr_coe_topK[i,:], _, _, _, _, _ = self.cano_corr(X_shuffled, Y_shuffled, V_A=V_A, V_B=V_B, Lam=Lam)
        return corr_coe_topK

    def cross_val(self):
        fold = self.fold
        n_components = self.n_components
        corr_train = np.zeros((fold, n_components))
        corr_test = np.zeros((fold, n_components))
        dist_train = np.zeros((fold, 1))
        dist_test = np.zeros((fold, 1))
        for idx in range(fold):
            # EEG_train, EEG_test, Sti_train, Sti_test = split(EEG_list, feature_list, fold=fold, fold_idx=idx+1)
            EEG_train, EEG_test, Sti_train, Sti_test = utils.split_balance(self.EEG_list, self.Stim_list, fold=fold, fold_idx=idx+1)
            corr_train[idx,:], dist_train[idx], _, V_A_train, V_B_train, Lam = self.cano_corr(EEG_train, Sti_train)
            corr_test[idx,:], dist_test[idx], _, _, _, _ = self.cano_corr(EEG_test, Sti_test, V_A=V_A_train, V_B=V_B_train, Lam=Lam)
        if self.signifi_level:
            if self.pool:
                corr_trials = self.permutation_test(EEG_test, Sti_test, V_A=V_A_train, V_B=V_B_train, Lam=Lam, block_len=1)
                corr_trials = np.sort(abs(corr_trials), axis=None)
                sig_idx = -int(self.n_permu*self.p_value*n_components)
                print('Significance level: {}'.format(corr_trials[sig_idx]))
            else:
                corr_trials = self.permutation_test(EEG_test, Sti_test, V_A=V_A_train, V_B=V_B_train, Lam=Lam, block_len=20*self.fs)
                corr_trials = np.sort(abs(corr_trials), axis=0)
                sig_idx = -int(self.n_permu*self.p_value)
                print('Significance level of each component: {}'.format(corr_trials[sig_idx,:])) 
        if self.message:
            print('Average correlation coefficients of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train, axis=0)))
            print('Average correlation coefficients of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test, axis=0)))
        return corr_train, corr_test, dist_train, dist_test, V_A_train, V_B_train


class GeneralizedCCA:
    def __init__(self, EEG_list, fs, L, offset, fold=10, n_components=5, regularization='lwcov', message=True, signifi_level=True, pool=True, n_permu=1000, p_value=0.05):
        '''
        EEG_list: list of EEG data, each element is a T(#sample)xDx(#channel) array
        fs: Sampling rate
        L: If use (spatial-) temporal filter, the number of taps
        offset: If use (spatial-) temporal filter, the offset of the time lags
        fold: Number of folds for cross-validation
        n_components: Number of components to be returned
        regularization: Regularization of the estimated covariance matrix
        message: If print message
        signifi_level: If calculate significance level
        pool: If pool the significance level of all components
        n_permu: Number of permutations for significance level calculation
        p_value: P-value for significance level calculation
        '''
        self.EEG_list = EEG_list
        self.fs = fs
        self.L = L
        self.offset = offset
        self.fold = fold
        self.n_components = n_components
        self.regularization = regularization
        self.message = message
        self.signifi_level = signifi_level
        self.pool = pool
        self.n_permu = n_permu
        self.p_value = p_value

    def GCCA(self, X_stack):
        '''
        Inputs:
        X_stack: stacked (along axis 2) data of different subjects
        Outputs:
        lam: eigenvalues, related to mean squared error (not used in analysis)
        W_stack: (rescaled) weights with shape (D*N*n_components)
        avg_corr: average pairwise correlation
        '''
        T, D, N = X_stack.shape
        L = self.L
        # From [X1; X2; ... XN] to [X1 X2 ... XN]
        # each column represents a variable, while the rows contain observations
        X_list = [utils.block_Hankel(X_stack[:,:,n], L, self.offset) for n in range(N)]
        X = np.concatenate(tuple(X_list), axis=1)
        Rxx = np.cov(X, rowvar=False)
        Dxx = np.zeros_like(Rxx)
        for n in range(N):
            if self.regularization == 'lwcov':
                Rxx[n*D*L:(n+1)*D*L, n*D*L:(n+1)*D*L] = LedoitWolf().fit(X[:, n*D*L:(n+1)*D*L]).covariance_
            Dxx[n*D*L:(n+1)*D*L, n*D*L:(n+1)*D*L] = Rxx[n*D*L:(n+1)*D*L, n*D*L:(n+1)*D*L]
        lam, W = eigh(Dxx, Rxx, subset_by_index=[0,self.n_components-1]) # automatically ascend
        Lam = np.diag(lam)
        # Right scaling
        W = W @ sqrtm(LA.inv(Lam.T @ W.T @ Rxx * T @ W @ Lam))
        # Forward models
        F_redun = T * Dxx @ W
        # Reshape W as (DL*n_components*N)
        W_stack = np.reshape(W, (N,D*L,-1))
        W_stack = np.transpose(W_stack, [1,0,2])
        F_redun_stack = np.reshape(F_redun, (N,D*L,-1))
        F_redun_stack = np.transpose(F_redun_stack, [1,0,2])
        F_stack = utils.F_organize(F_redun_stack, L, self.offset, avg=True)
        return W_stack, F_stack, lam
    
    def avg_corr_coe(self, X_stack, W_stack):
        '''
        Calculate the pairwise average correlation.
        Inputs:
        X_stack: stacked (along axis 2) data of different subjects
        W_stack: weights 1) dim(W)=2: results of correlated component analysis 2) dim(W)=3: results of GCCA
        Output:
        avg_corr: pairwise average correlation
        avg_ChDist: pairwise average Chordal distance
        avg_TSC: pairwise average total squared correlation
        '''
        _, _, N = X_stack.shape
        n_components = self.n_components
        Hankellist = [np.expand_dims(utils.block_Hankel(X_stack[:,:,n], self.L, self.offset), axis=2) for n in range(N)]
        X_stack = np.concatenate(tuple(Hankellist), axis=2)
        corr_mtx_stack = np.zeros((N,N,n_components))
        avg_corr = np.zeros(n_components)
        if np.ndim (W_stack) == 2: # for correlated component analysis
            W_stack = np.expand_dims(W_stack, axis=1)
            W_stack = np.repeat(W_stack, N, axis=1)
        for component in range(n_components):
            w = W_stack[:,:,component]
            w = np.expand_dims(w, axis=1)
            X_trans = np.einsum('tdn,dln->tln', X_stack, w)
            X_trans = np.squeeze(X_trans, axis=1)
            corr_mtx_stack[:,:,component] = np.corrcoef(X_trans, rowvar=False)
            avg_corr[component] = np.sum(corr_mtx_stack[:,:,component]-np.eye(N))/N/(N-1)
        Squared_corr = np.sum(np.square(corr_mtx_stack[:,:,:3]), axis=2)
        avg_TSC = np.sum(Squared_corr-3*np.eye(N))/N/(N-1)
        Chordal_dist = np.sqrt(3-Squared_corr)
        avg_ChDist = np.sum(Chordal_dist)/N/(N-1)
        return avg_corr, avg_ChDist, avg_TSC

    def permutation_test(self, X_stack, W_stack, block_len):
        corr_coe_topK = np.empty((0, self.n_components))
        for i in tqdm(range(self.n_permu)):
            X_shuffled = utils.shuffle_3D(X_stack, block_len)
            corr_coe, _, _ = self.avg_corr_coe(X_shuffled, W_stack)
            corr_coe_topK = np.concatenate((corr_coe_topK, np.expand_dims(corr_coe[:self.n_components], axis=0)), axis=0)
        return corr_coe_topK
    
    def cross_val(self):
        fold = self.fold
        n_components = self.n_components
        corr_train = np.zeros((fold, n_components))
        corr_test = np.zeros((fold, n_components))
        tsc_train = np.zeros((fold, 1))
        tsc_test = np.zeros((fold, 1))
        dist_train = np.zeros((fold, 1))
        dist_test = np.zeros((fold, 1))
        for idx in range(fold):
            train_list, test_list = utils.split_mm_balance([self.EEG_list], fold=fold, fold_idx=idx+1)
            W_train, F_train, _ = self.GCCA(train_list[0])
            corr_train[idx,:], dist_train[idx], tsc_train[idx] = self.avg_corr_coe(train_list[0], W_train)
            corr_test[idx,:], dist_test[idx], tsc_test[idx] = self.avg_corr_coe(test_list[0], W_train)
        if self.signifi_level:
            if self.pool:
                corr_trials = self.permutation_test(test_list[0], W_train, block_len=1)
                corr_trials = np.sort(abs(corr_trials), axis=None)
                sig_idx = -int(self.n_permu*self.p_value*n_components)
                print('Significance level: {}'.format(corr_trials[sig_idx]))
            else:
                corr_trials = self.permutation_test(test_list[0], W_train, block_len=20*self.fs)
                corr_trials = np.sort(abs(corr_trials), axis=0)
                sig_idx = -int(self.n_permu*self.p_value)
                print('Significance level of each component: {}'.format(corr_trials[sig_idx,:]))
        if self.message:
            print('Average ISC of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train, axis=0)))
            print('Average ISC of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test, axis=0)))
        return corr_train, corr_test, tsc_train, tsc_test, dist_train, dist_test, W_train, F_train
    

class CorrelatedComponentAnalysis:
    def __init__(self, EEG_list, fs, L=1, offset=0, fold=10, n_components=5, regularization='lwcov', message=True, signifi_level=True, pool=True, n_permu=1000, p_value=0.05):
        '''
        EEG_list: list of EEG data, each element is a T(#sample)xDx(#channel) array
        fs: Sampling rate
        L: If use (spatial-) temporal filter, the number of taps
        offset: If use (spatial-) temporal filter, the offset of the time lags
        fold: Number of folds for cross-validation
        n_components: Number of components to be returned
        regularization: Regularization of the estimated covariance matrix
        message: If print message
        signifi_level: If calculate significance level
        pool: If pool the significance level of all components
        n_permu: Number of permutations for significance level calculation
        p_value: P-value for significance level calculation
        '''
        self.EEG_list = EEG_list
        self.fs = fs
        self.L = L
        self.offset = offset
        self.fold = fold
        self.n_components = n_components
        self.regularization = regularization
        self.message = message
        self.signifi_level = signifi_level
        self.pool = pool
        self.n_permu = n_permu
        self.p_value = p_value

    def corr_component(self, EEG, W_train=None):
        '''
        Input:
        EEG: EEG data with shape (T, D, N) [T: # sample, D: # channel, N: # subjects]
        W_train: If not None, then goes to test mode.
        Outputs:
        ISC: inter-subject correlation defined in Parra's work, assuming w^T X_n^T X_n w is equal for all subjects
        W: weights
        '''
        _, _, N = EEG.shape
        X_list = [utils.block_Hankel(EEG[:,:,n], self.L, self.offset) for n in range(N)]
        X = np.stack(X_list, axis=2)
        _, D, _ = X.shape
        Rw = np.zeros([D,D])
        for n in range(N):
            if self.regularization == 'lwcov':
                Rw += LedoitWolf().fit(X[:,:,n]).covariance_
            else:
                Rw += np.cov(np.transpose(X[:,:,n])) # Inside np.cov: observations in the columns
        if self.regularization == 'lwcov':
            Rt = N**2*LedoitWolf().fit(np.average(X, axis=2)).covariance_
        else:
            Rt = N**2*np.cov(np.transpose(np.average(X, axis=2)))
        Rb = (Rt - Rw)/(N-1)
        if W_train is not None:
            W = W_train
            ISC = np.diag((np.transpose(W)@Rb@W)/(np.transpose(W)@Rw@W))
        else:
            ISC, W = eigh(Rb, Rw, subset_by_index=[D-self.n_components,D-1])
            ISC = np.squeeze(np.fliplr(np.expand_dims(ISC, axis=0)))
            W = np.fliplr(W)
        return ISC, W

    def forward_model(self, EEG, W):
        '''
        Input:
        EEG: EEG data with shape (T, D, N) [T: # sample, D: # channel, N: # subjects]
        L: If use (spatial-) temporal filter, the number of taps
        offset: If use (spatial-) temporal filter, the offset of the time lags
        W: weights with shape (DL, n_components)
        Outputs:
        F: forward model (D, n_components)
        '''
        _, _, N = EEG.shape
        X_list = [utils.block_Hankel(EEG[:,:,n], self.L, self.offset) for n in range(N)]
        X = np.concatenate(tuple(X_list), axis=0)
        X_transformed = X @ W
        F_redun = (lstsq(X_transformed, X)[0]).T
        F = utils.F_organize(F_redun, self.L, self.offset)
        return F

    def cross_val(self):
        fold = self.fold
        n_components = self.n_components
        corr_train = np.zeros((fold, n_components))
        corr_test = np.zeros((fold, n_components))
        tsc_train = np.zeros((fold, 1))
        tsc_test = np.zeros((fold, 1))
        isc_train = np.zeros((fold, n_components))
        isc_test = np.zeros((fold, n_components))
        GCCA = GeneralizedCCA(self.EEG_list, self.fs, self.L, self.offset, n_components=self.n_components)
        for idx in range(fold):
            train_list, test_list = utils.split_mm_balance([self.EEG_list], fold=fold, fold_idx=idx+1)
            isc_train[idx,:], W_train = self.corr_component(train_list[0])
            isc_test[idx, :], _ = self.corr_component(test_list[0], W_train)
            corr_train[idx,:], _, tsc_train[idx] = GCCA.avg_corr_coe(train_list[0], W_train)
            corr_test[idx,:], _, tsc_test[idx] = GCCA.avg_corr_coe(test_list[0], W_train)
        if self.signifi_level:
            if self.pool:
                corr_trials = GCCA.permutation_test(test_list[0], W_train, block_len=1)
                corr_trials = np.sort(abs(corr_trials), axis=None)
                sig_idx = -int(self.n_permu*self.p_value*n_components)
                print('Significance level: {}'.format(corr_trials[sig_idx]))
            else:
                corr_trials = GCCA.permutation_test(test_list[0], W_train, block_len=20*self.fs)
                corr_trials = np.sort(abs(corr_trials), axis=0)
                sig_idx = -int(self.n_permu*self.p_value)
                print('Significance level of each component: {}'.format(corr_trials[sig_idx,:]))
        if self.message:
            print('Average ISC of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train, axis=0)))
            print('Average ISC of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test, axis=0)))
        return corr_train, corr_test, tsc_train, tsc_test, isc_train, isc_test, W_train


class StimulusInformedGCCA:
    def __init__(self, nested_datalist, fs, Llist, offsetlist, fold=10, n_components=5, regularization='lwcov', message=True, ISC=True, pool=True, n_permu=1000, p_value=0.05):
        '''
        nested_datalist: [EEG_list, Stim_list], where
            EEG_list: list of EEG data, each element is a T(#sample)xDx(#channel) array
            Stim_list: list of stimulus, each element is a T(#sample)xDy(#feature dim) array
        fs: Sampling rate
        Llist: [L_EEG, L_Stim], where
            L_EEG/L_Stim: If use (spatial-) temporal filter, the number of taps
        offsetlist: [offset_EEG, offset_Stim], where
            offset_EEG/offset_Stim: If use (spatial-) temporal filter, the offset of the time lags
        fold: Number of folds for cross-validation
        n_components: Number of components to be returned
        regularization: Regularization of the estimated covariance matrix
        message: If print message
        signifi_level: If calculate significance level
        pool: If pool the significance level of all components
        n_permu: Number of permutations for significance level calculation
        p_value: P-value for significance level calculation
        '''
        self.nested_datalist = nested_datalist
        self.fs = fs
        self.Llist = Llist
        self.offsetlist = offsetlist
        self.n_components = n_components
        self.fold = fold
        self.regularization = regularization
        self.message = message
        self.ISC = ISC
        self.pool = pool
        self.n_permu = n_permu
        self.p_value = p_value

    def SI_GCCA(self, datalist, rho):
        EEG, Stim = datalist
        # EEG = EEG/LA.norm(EEG)
        # Stim = Stim/LA.norm(Stim)
        if np.ndim(EEG) == 2:
            EEG = np.expand_dims(EEG, axis=2)
        T, D_eeg, N = EEG.shape
        _, D_stim = Stim.shape
        L_EEG, L_Stim = self.Llist
        dim_list = [D_eeg*L_EEG]*N + [D_stim*L_Stim]
        offset_EEG, offset_Stim = self.offsetlist
        EEG_list = [utils.block_Hankel(EEG[:,:,n], L_EEG, offset_EEG) for n in range(N)]
        EEG_Hankel = np.concatenate(tuple(EEG_list), axis=1)
        Stim_Hankel = utils.block_Hankel(Stim, L_Stim, offset_Stim)
        X = np.concatenate((EEG_Hankel, Stim_Hankel), axis=1)
        Rxx = np.cov(X, rowvar=False)
        Dxx = np.zeros_like(Rxx)
        dim_accumu = 0
        for dim in dim_list:
            if self.regularization == 'lwcov':
                Rxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim] = LedoitWolf().fit(X[:,dim_accumu:dim_accumu+dim]).covariance_
            Dxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim] = Rxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim]
            dim_accumu = dim_accumu + dim
        try:
            lam, W = utils.transformed_GEVD(Dxx, Rxx, rho, dim_list[-1], self.n_components)
            Lam = np.diag(lam)
            Rxx[:,-D_stim*L_Stim:] = Rxx[:,-D_stim*L_Stim:]*rho
        except:
            print("Numerical issue exists for eigh. Use eig instead.")
            Rxx[:,-D_stim*L_Stim:] = Rxx[:,-D_stim*L_Stim:]*rho
            lam, W = eig(Dxx, Rxx)
            idx = np.argsort(lam)
            lam = np.real(lam[idx]) # rank eigenvalues
            W = np.real(W[:, idx]) # rearrange eigenvectors accordingly
            lam = lam[:self.n_components]
            Lam = np.diag(lam)
            W = W[:,:self.n_components]
        # Right scaling
        Rxx[-D_stim*L_Stim:, :] = Rxx[-D_stim*L_Stim:, :]*rho
        W = W @ sqrtm(LA.inv(Lam.T @ W.T @ Rxx * T @ W @ Lam))
        # Forward models
        F = T * Dxx @ W
        # Organize weights of different modalities
        Wlist = utils.W_organize(W, datalist, self.Llist)
        Flist = utils.W_organize(F, datalist, self.Llist)
        Fstack = utils.F_organize(Flist[0], L_EEG, offset_EEG, avg=True)
        return Wlist, Fstack, lam
    
    def avg_corr_coe(self, datalist, Wlist):
        '''
        Calculate the pairwise average correlation.
        Inputs:
        datalist: data of different modalities (a list) E.g, [EEG_stack, Stim]
        Wlist: weights of different modalities (a list)
        Output:
        avg_corr: average pairwise correlation
        '''
        n_components = self.n_components
        Llist = self.Llist
        offsetlist = self.offsetlist
        if self.ISC and (np.ndim(datalist[0]) != 2): # calculate avg correlation across only EEG views, unless there is only one EEG subject (CCA)
            GCCA = GeneralizedCCA(datalist[0], self.fs, Llist[0], offsetlist[0], n_components=n_components)
            avg_corr, avg_ChDist, avg_TSC = GCCA.avg_corr_coe(datalist[0], Wlist[0])
        else:
            avg_corr = np.zeros(n_components)
            corr_mtx_list = []
            n_mod = len(datalist)
            for component in range(n_components):
                X_trans_list = []
                for i in range(n_mod):
                    W = Wlist[i]
                    rawdata = datalist[i]
                    L = Llist[i]
                    offset = offsetlist[i]
                    if np.ndim(W) == 3:
                        w = W[:,:,component]
                        w = np.expand_dims(w, axis=1)
                        data_trans = [np.expand_dims(utils.block_Hankel(rawdata[:,:,n],L,offset),axis=2) for n in range(rawdata.shape[2])]
                        data = np.concatenate(tuple(data_trans), axis=2)
                        X_trans = np.einsum('tdn,dln->tln', data, w)
                        X_trans = np.squeeze(X_trans, axis=1)
                    if np.ndim(W) == 2:
                        w = W[:,component]
                        data = utils.block_Hankel(rawdata,L,offset)
                        X_trans = data@w
                        X_trans = np.expand_dims(X_trans, axis=1)
                    X_trans_list.append(X_trans)
                X_trans_all = np.concatenate(tuple(X_trans_list), axis=1)
                if self.regularization=='lwcov':
                    cov_mtx = LedoitWolf().fit(X_trans_all).covariance_
                    cov_diag = np.expand_dims(np.diag(cov_mtx).shape, axis = 1)
                    corr_mtx = cov_mtx/np.sqrt(cov_diag)/np.sqrt(cov_diag.T)
                else:
                    corr_mtx = np.corrcoef(X_trans_all, rowvar=False)
                N = X_trans_all.shape[1]
                corr_mtx_list.append(corr_mtx)
                avg_corr[component] = np.sum(corr_mtx-np.eye(N))/N/(N-1)
            corr_mtx_stack = np.stack(corr_mtx_list, axis=2)
            Squared_corr = np.sum(np.square(corr_mtx_stack[:,:,:3]), axis=2)
            avg_TSC = np.sum(Squared_corr-3*np.eye(N))/N/(N-1)
            Chordal_dist = np.sqrt(3-Squared_corr)
            avg_ChDist = np.sum(Chordal_dist)/N/(N-1)
        return avg_corr, avg_ChDist, avg_TSC
    
    def rho_sweep(self, sweep_list, message=True):
        best = np.Inf
        for i in sweep_list:
            rho = 10**i
            _, _, _, _, dist_train, dist_test, _, _ = self.cross_val(rho, signifi_level=False)
            avg_dist_train = np.average(dist_train)
            avg_dist_test = np.average(dist_test)
            if message:
                print('Average ISD across different training sets when rho=10**{}: {}'.format(i, avg_dist_train))
                print('Average ISD across different test sets when rho=10**{}: {}'.format(i, avg_dist_test))
            if avg_dist_test < best:
                rho_best = rho
                best = avg_dist_test
        return rho_best

    def permutation_test(self, datalist, Wlist, block_len):
        n_components = self.n_components
        corr_coe_topK = np.empty((0, n_components))
        for i in tqdm(range(self.n_permu)):
            datalist_shuffled = []
            for data in datalist:
                if np.ndim(data) == 2:
                    datalist_shuffled.append(utils.shuffle_2D(data, block_len))
                elif np.ndim(data) == 3:
                    datalist_shuffled.append(utils.shuffle_3D(data, block_len))
            corr_coe, _, _ = self.avg_corr_coe(datalist_shuffled, Wlist)
            corr_coe_topK = np.concatenate((corr_coe_topK, np.expand_dims(corr_coe[:n_components], axis=0)), axis=0)
        return corr_coe_topK

    def cross_val(self, rho, signifi_level=True):
        n_components = self.n_components
        fold = self.fold
        corr_train = np.zeros((fold, n_components))
        corr_test = np.zeros((fold, n_components))
        tsc_train = np.zeros((fold, 1))
        tsc_test = np.zeros((fold, 1))
        dist_train = np.zeros((fold, 1))
        dist_test = np.zeros((fold, 1))
        for idx in range(fold):
            train_list, test_list = utils.split_mm_balance(self.nested_datalist, fold=fold, fold_idx=idx+1)
            Wlist_train, F_train, _ = self.SI_GCCA(train_list, rho)
            corr_train[idx,:], dist_train[idx], tsc_train[idx] = self.avg_corr_coe(train_list, Wlist_train)
            corr_test[idx,:], dist_test[idx], tsc_test[idx] = self.avg_corr_coe(test_list, Wlist_train)
        if signifi_level:
            if self.pool:
                corr_trials = self.permutation_test(test_list, Wlist_train, block_len=1)
                corr_trials = np.sort(abs(corr_trials), axis=None)
                sig_idx = -int(self.n_permu*self.p_value*n_components)
                print('Significance level: {}'.format(corr_trials[sig_idx]))
            else:
                corr_trials = self.permutation_test(test_list, Wlist_train, block_len=20*self.fs)
                corr_trials = np.sort(abs(corr_trials), axis=0)
                sig_idx = -int(self.n_permu*self.p_value)
                print('Significance level of each component: {}'.format(corr_trials[sig_idx,:]))
        if self.message:
            print('Average ISC of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train, axis=0)))
            print('Average ISC of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test, axis=0)))
        return corr_train, corr_test, tsc_train, tsc_test, dist_train, dist_test, Wlist_train, F_train


