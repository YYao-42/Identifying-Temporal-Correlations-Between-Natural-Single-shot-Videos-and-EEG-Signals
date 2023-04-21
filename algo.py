import numpy as np
import copy
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
from numpy import linalg as LA
from scipy.linalg import eig, eigh, sqrtm, lstsq
from scipy.stats import pearsonr
import utils


class LeastSquares:
    '''
    Note:
    When working with a forward model, the encoder maps the stimuli (or the latent variables) to the EEG data. We need to take more past samples of the stimuli into account. Therefore the offset should be zero or a small number to compensate for the possible misalignment.
    When working with a backward model, the decoder maps the EEG data (or the latent variables) to the stimuli. We need to take more future samples of the EEG data into account. Therefore the offset should be L-1 or a slightly smaller number than L-1. 
    '''
    def __init__(self, EEG_list, Stim_list, fs, decoding, L_EEG=1, offset_EEG=0, L_Stim=1, offset_Stim=0, fold=10, message=True, signifi_level=True, pool=True, n_permu=1000, p_value=0.05):
        self.EEG_list = EEG_list
        self.Stim_list = Stim_list
        self.fs = fs
        self.decoding = decoding
        self.L_EEG = L_EEG
        self.offset_EEG = offset_EEG
        self.L_Stim = L_Stim
        self.offset_Stim = offset_Stim
        self.fold = fold
        self.message = message
        self.signifi_level = signifi_level
        self.pool = pool
        self.n_permu = n_permu
        self.p_value = p_value
        if decoding:
            if np.ndim(Stim_list[0]) == 1:
                self.n_components = 1
            else:
                self.n_components = Stim_list[0].shape[1]
        else:
            self.n_components = EEG_list[0].shape[1]

    def encoder(self, EEG, Stim, W_f=None):
        '''
        Inputs:
        EEG: T(#sample)xD_eeg(#channel) array
        stim: T(#sample)xD_stim(#feature dim) array
        W_f: only used in test mode
        Output:
        W_f: (D_stim*L_stim)xD_eeg array
        '''
        self.n_components = EEG.shape[1] # in case functions in other classes call this function and the EEG is different from the one in the initialization
        Stim_Hankel = utils.block_Hankel(Stim, self.L_Stim, self.offset_Stim)
        if W_f is not None: # Test Mode
            pass
        else:
            W_f = lstsq(Stim_Hankel, EEG)[0]
        filtered_Stim = Stim_Hankel@W_f
        mse = np.mean((filtered_Stim-EEG)**2)
        corr_pvalue = [pearsonr(EEG[:,k], filtered_Stim[:,k]) for k in range(self.n_components)]
        corr = np.array([corr_pvalue[k][0] for k in range(self.n_components)])
        return W_f, mse, corr

    def decoder(self, EEG, Stim, W_b=None):
        '''
        Inputs:
        EEG: T(#sample)xD_eeg(#channel) array
        stim: T(#sample)xD_stim(#feature dim) array
        W_b: only used in test mode
        Output:
        W_b: (D_eeg*L_eeg)xD_stim array
        '''
        if np.ndim(Stim) == 1:
            Stim = np.expand_dims(Stim, axis=1)
        self.n_components = Stim.shape[1] # in case functions in other classes call this function and the Stim is different from the one in the initialization
        EEG_Hankel = utils.block_Hankel(EEG, self.L_EEG, self.offset_EEG)
        if W_b is not None: # Test Mode
            pass
        else:
            W_b = lstsq(EEG_Hankel, Stim)[0]
        filtered_EEG = EEG_Hankel@W_b
        mse = np.mean((filtered_EEG-Stim)**2)
        corr_pvalue = [pearsonr(filtered_EEG[:,k], Stim[:,k]) for k in range(self.n_components)]
        corr = np.array([corr_pvalue[k][0] for k in range(self.n_components)])
        return W_b, mse, corr

    def permutation_test(self, X, Y, W_fb, block_len):
        corr_coe_topK = np.zeros((self.n_permu, self.n_components))
        for i in tqdm(range(self.n_permu)):
            X_shuffled = utils.shuffle_2D(X, block_len)
            Y_shuffled = utils.shuffle_2D(Y, block_len)
            if self.decoding:
                _, _, corr_coe_topK[i,:] = self.decoder(X_shuffled, Y_shuffled, W_b=W_fb)
            else:
                _, _, corr_coe_topK[i,:] = self.encoder(X_shuffled, Y_shuffled, W_f=W_fb)
        return corr_coe_topK

    def cross_val(self):
        fold = self.fold
        mse_train = np.zeros((fold, self.n_components))
        mse_test = np.zeros((fold, self.n_components))
        corr_train = np.zeros((fold, self.n_components))
        corr_test = np.zeros((fold, self.n_components))
        for idx in range(fold):
            EEG_train, EEG_test, Sti_train, Sti_test = utils.split_balance(self.EEG_list, self.Stim_list, fold=fold, fold_idx=idx+1)
            if self.decoding:
                W_train, mse_train[idx,:], corr_train[idx,:] = self.decoder(EEG_train, Sti_train)
                _, mse_test[idx,:], corr_test[idx,:] = self.decoder(EEG_test, Sti_test, W_b=W_train)
            else:
                W_train, mse_train[idx,:], corr_train[idx,:] = self.encoder(EEG_train, Sti_train)
                _, mse_test[idx], corr_test[idx] = self.encoder(EEG_test, Sti_test, W_f=W_train)
        if self.signifi_level:
            if self.pool:
                corr_trials = self.permutation_test(EEG_test, Sti_test, W_fb=W_train, block_len=1)
                corr_trials = np.sort(abs(corr_trials), axis=None)
                sig_idx = -int(self.n_permu*self.p_value*self.n_components)
                print('Significance level: {}'.format(corr_trials[sig_idx]))
            else:
                corr_trials = self.permutation_test(EEG_test, Sti_test, W_fb=W_train, block_len=1)
                corr_trials = np.sort(abs(corr_trials), axis=0)
                sig_idx = -int(self.n_permu*self.p_value)
                print('Significance level of each component: {}'.format(corr_trials[sig_idx,:]))
        if self.message:
            print('Average correlation across {} training folds: {}'.format(self.fold, np.average(corr_train, axis=0)))
            print('Average correlation across {} test folds: {}'.format(self.fold, np.average(corr_test, axis=0)))
        return mse_train, mse_test, corr_train, corr_test, W_train


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

    def fit(self, X, Y, V_A=None, V_B=None, Lam=None):
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
        # mtx_X and mtx_Y should be centered according to the definition. But since we calculate the correlation coefficients, it does not matter.
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
        return corr_coe, TSC, ChDist, p_value, V_A, V_B, Lam

    def permutation_test(self, X, Y, V_A, V_B, Lam, block_len):
        corr_coe_topK = np.zeros((self.n_permu, self.n_components))
        for i in tqdm(range(self.n_permu)):
            X_shuffled = utils.shuffle_2D(X, block_len)
            Y_shuffled = utils.shuffle_2D(Y, block_len)
            corr_coe_topK[i,:], _, _, _, _, _, _ = self.fit(X_shuffled, Y_shuffled, V_A=V_A, V_B=V_B, Lam=Lam)
        return corr_coe_topK

    def forward_model(self, X, V_A):
        '''
        Inputs:
        X: observations (one subject) TxD
        V_A: filters/backward models DLxK
        L: time lag (if temporal-spatial)
        Output:
        F: forward model
        '''
        X_block_Hankel = utils.block_Hankel(X, self.L_EEG, self.offset_EEG)
        Rxx = np.cov(X_block_Hankel, rowvar=False)
        F_redun = Rxx@V_A@LA.inv(V_A.T@Rxx@V_A)
        F = utils.F_organize(F_redun, self.L_EEG, self.offset_EEG)
        return F

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
            # EEG_train, EEG_test, Sti_train, Sti_test = split(EEG_list, feature_list, fold=fold, fold_idx=idx+1)
            EEG_train, EEG_test, Sti_train, Sti_test = utils.split_balance(self.EEG_list, self.Stim_list, fold=fold, fold_idx=idx+1)
            corr_train[idx,:], tsc_train[idx], dist_train[idx], _, V_A_train, V_B_train, Lam = self.fit(EEG_train, Sti_train)
            corr_test[idx,:], tsc_test[idx], dist_test[idx], _, _, _, _ = self.fit(EEG_test, Sti_test, V_A=V_A_train, V_B=V_B_train, Lam=Lam)
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
        return corr_train, corr_test, tsc_train, tsc_test, dist_train, dist_test, V_A_train, V_B_train


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

    def fit(self, X_stack):
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
        X_center = copy.deepcopy(X)
        Rxx = np.cov(X, rowvar=False)
        Dxx = np.zeros_like(Rxx)
        for n in range(N):
            X_center[:,n*D*L:(n+1)*D*L] -= np.mean(X_center[:,n*D*L:(n+1)*D*L], axis=0, keepdims=True)
            if self.regularization == 'lwcov':
                Rxx[n*D*L:(n+1)*D*L, n*D*L:(n+1)*D*L] = LedoitWolf().fit(X[:, n*D*L:(n+1)*D*L]).covariance_
            Dxx[n*D*L:(n+1)*D*L, n*D*L:(n+1)*D*L] = Rxx[n*D*L:(n+1)*D*L, n*D*L:(n+1)*D*L]
        lam, W = eigh(Dxx, Rxx, subset_by_index=[0,self.n_components-1]) # automatically ascend
        Lam = np.diag(lam)
        # Right scaling
        W = W @ sqrtm(LA.inv(Lam.T @ W.T @ Rxx * T @ W @ Lam))
        # Shared subspace
        S = X_center@W@Lam
        # Forward models
        F_redun = T * Dxx @ W
        # Reshape W as (DL*n_components*N)
        W_stack = np.reshape(W, (N,D*L,-1))
        W_stack = np.transpose(W_stack, [1,0,2])
        F_redun_stack = np.reshape(F_redun, (N,D*L,-1))
        F_redun_stack = np.transpose(F_redun_stack, [1,0,2])
        F_stack = utils.F_organize(F_redun_stack, L, self.offset, avg=True)
        return W_stack, S, F_stack, lam
    
    def forward_model(self, EEG, W_stack, S=None):
        '''
        Input:
        EEG: EEG data with shape (T, D, N) [T: # sample, D: # channel, N: # subjects]
        W: weights with shape (DL, n_components)
        S: shared subspace with shape (T, n_components); if not None, then calculate forward model from the shared subspace
        Outputs:
        F: forward model (D, n_components)
        '''
        _, _, N = EEG.shape
        X_list = [utils.block_Hankel(EEG[:,:,n], self.L, self.offset) for n in range(N)]
        X_list_center = [X_list[n] - np.mean(X_list[n], axis=0, keepdims=True) for n in range(N)]
        X_stack = np.stack(X_list_center, axis=2)
        if S is not None:
            F_T = np.mean(np.einsum('kt, tdn -> kdn', S.T, X_stack), axis=2)
            F_redun = F_T.T
        else:
            X = np.concatenate(tuple(X_list_center), axis=0)
            X_list_trans = [X_stack[:,:,n]@W_stack[:,n,:] for n in range(N)]
            X_transformed = np.concatenate(tuple(X_list_trans), axis=0)
            F_redun = (lstsq(X_transformed, X)[0]).T
        F = utils.F_organize(F_redun, self.L, self.offset)
        return F

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
            W_train, _, F_train, _ = self.fit(train_list[0])
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
    

class CorrelatedComponentAnalysis(GeneralizedCCA):
    def fit(self, EEG, W_train=None):
        '''
        Input:
        EEG: EEG data with shape (T, D, N) [T: # sample, D: # channel, N: # subjects]
        W_train: If not None, then goes to test mode.
        Outputs:
        ISC: inter-subject correlation defined in Parra's work, assuming w^T X_n^T X_n w is equal for all subjects
        W: weights
        '''
        T, _, N = EEG.shape
        X_list = [utils.block_Hankel(EEG[:,:,n], self.L, self.offset) for n in range(N)]
        X = np.stack(X_list, axis=2)
        X_center = X - np.mean(X, axis=0, keepdims=True)
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
            S = None
            F = None
        else:
            ISC, W = eigh(Rb, Rw, subset_by_index=[D-self.n_components,D-1])
            ISC = np.squeeze(np.fliplr(np.expand_dims(ISC, axis=0)))
            W = np.fliplr(W)
            # right scaling
            Lam = np.diag(1/(ISC*(N-1)+1))
            W = W @ sqrtm(LA.inv(Lam.T @ W.T @ Rt * T @ W @ Lam))
            # shared subspace
            S = np.sum(X_center, axis=2) @ W @ Lam
            # Forward models
            F_redun = T * Rw @ W / N
            F = utils.F_organize(F_redun, self.L, self.offset)
        return ISC, W, S, F

    def forward_model(self, EEG, W, S=None):
        '''
        Input:
        EEG: EEG data with shape (T, D, N) [T: # sample, D: # channel, N: # subjects]
        W: weights with shape (DL, n_components)
        S: shared subspace with shape (T, n_components); if not None, then calculate forward model from the shared subspace
        Outputs:
        F: forward model (D, n_components)
        '''
        _, _, N = EEG.shape
        X_list = [utils.block_Hankel(EEG[:,:,n], self.L, self.offset) for n in range(N)]
        X_list_center = [X_list[n] - np.mean(X_list[n], axis=0, keepdims=True) for n in range(N)]
        if S is not None:
            X_stack = np.stack(X_list_center, axis=2)
            F_T = np.mean(np.einsum('kt, tdn -> kdn', S.T, X_stack), axis=2)
            F_redun = F_T.T
        else:
            X = np.concatenate(tuple(X_list_center), axis=0)
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
        for idx in range(fold):
            train_list, test_list = utils.split_mm_balance([self.EEG_list], fold=fold, fold_idx=idx+1)
            isc_train[idx,:], W_train, _, F_train = self.fit(train_list[0])
            isc_test[idx, :], _, _, _ = self.fit(test_list[0], W_train)
            corr_train[idx,:], _, tsc_train[idx] = self.avg_corr_coe(train_list[0], W_train)
            corr_test[idx,:], _, tsc_test[idx] = self.avg_corr_coe(test_list[0], W_train)
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
        return corr_train, corr_test, tsc_train, tsc_test, isc_train, isc_test, W_train, F_train


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

    def fit(self, datalist, rho):
        EEG, Stim = datalist
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
        # Shared subspace
        S = self.shared_subspace(X, W, Lam, N, D_eeg*L_EEG, D_stim*L_Stim, rho)
        # Forward models
        F = T * Dxx @ W
        # Organize weights of different modalities
        Wlist = utils.W_organize(W, datalist, self.Llist)
        Flist = utils.W_organize(F, datalist, self.Llist)
        Fstack = utils.F_organize(Flist[0], L_EEG, offset_EEG, avg=True)
        return Wlist, S, Fstack, lam

    def shared_subspace(self, X, W, Lam, N, DL, DL_Stim, rho):
        W_rho = copy.deepcopy(W)
        W_rho[-DL_Stim:,:] = W[-DL_Stim:,:]*rho
        X_center = copy.deepcopy(X)
        for n in range(N):
            X_center[:,n*DL:(n+1)*DL] -= np.mean(X_center[:,n*DL:(n+1)*DL], axis=0, keepdims=True)
        X_center[:, -DL_Stim:] -= np.mean(X_center[:, -DL_Stim:], axis=0, keepdims=True)
        S = X_center@W_rho@Lam
        return S

    def forward_model(self, EEG, Wlist, S=None):
        '''
        Input:
        EEG: EEG data with shape (T, D, N) [T: # sample, D: # channel, N: # subjects]
        Wlist: [Weeg, Wstim]
        S: shared subspace with shape (T, n_components); if not None, then calculate forward model from the shared subspace
        Note: if S is not None, then EEG must be the one used in the training stage. So it is equivalent to the S generated by self.fit(). This can be used as a sanity check.
        Outputs:
        F: forward model (D, n_components)
        '''
        W = Wlist[0]
        if np.ndim(EEG) == 2:
            EEG = np.expand_dims(EEG, axis=2)
            W = np.expand_dims(W, axis=1)
        _, _, N = EEG.shape
        X_list = [utils.block_Hankel(EEG[:,:,n], self.Llist[0], self.offsetlist[0]) for n in range(N)]
        X_list_center = [X_list[n] - np.mean(X_list[n], axis=0, keepdims=True) for n in range(N)]
        X_stack = np.stack(X_list_center, axis=2)
        if S is not None:
            F_T = np.mean(np.einsum('kt, tdn -> kdn', S.T, X_stack), axis=2)
            F_redun = F_T.T
        else:
            X = np.concatenate(tuple(X_list_center), axis=0)
            X_list_trans = [X_stack[:,:,n]@W[:,n,:] for n in range(N)]
            X_transformed = np.concatenate(tuple(X_list_trans), axis=0)
            F_redun = (lstsq(X_transformed, X)[0]).T
        F = utils.F_organize(F_redun, self.Llist[0], self.offsetlist[0])
        return F

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
                    W = np.squeeze(Wlist[i])
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
            Wlist_train, _, F_train, _ = self.fit(train_list, rho)
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


class StimulusInformedCorrCA(StimulusInformedGCCA):
    def fit(self, datalist, rho):
        EEG, Stim = datalist
        if np.ndim(EEG) == 2:
            EEG = np.expand_dims(EEG, axis=2)
        T, D_eeg, N = EEG.shape
        _, D_stim = Stim.shape
        L_EEG, L_Stim = self.Llist
        dim_list = [D_eeg*L_EEG] + [D_stim*L_Stim]
        offset_EEG, offset_Stim = self.offsetlist
        EEG_list = [utils.block_Hankel(EEG[:,:,n], L_EEG, offset_EEG) for n in range(N)]
        EEG_Hankel = np.stack(EEG_list, axis=2)
        EEG_center = EEG_Hankel - np.mean(EEG_Hankel, axis=0, keepdims=True)
        Stim_Hankel = utils.block_Hankel(Stim, L_Stim, offset_Stim)
        Stim_center = Stim_Hankel - np.mean(Stim_Hankel, axis=0, keepdims=True)
        Rxx = np.zeros((dim_list[0]+dim_list[1], dim_list[0]+dim_list[1]))
        Rw = np.zeros((dim_list[0], dim_list[0]))
        for n in range(N):
            if self.regularization == 'lwcov':
                cov_eeg = LedoitWolf().fit(EEG_Hankel[:,:,n]).covariance_
            else:
                cov_eeg = np.cov(np.transpose(EEG_Hankel[:,:,n]))
            Rw += cov_eeg
            cov_es = EEG_center[:,:,n].T @ Stim_center / T
            Rxx[0:dim_list[0], dim_list[0]:] += cov_es
            Rxx[dim_list[0]:, 0:dim_list[0]] += cov_es.T
        if self.regularization == 'lwcov':
            Rt = N**2*LedoitWolf().fit(np.average(EEG_Hankel, axis=2)).covariance_
            cov_stim = LedoitWolf().fit(Stim_Hankel).covariance_
        else:
            Rt = N**2*np.cov(np.transpose(np.average(EEG_Hankel, axis=2)))
            cov_stim = np.cov(np.transpose(Stim_Hankel))
        Rxx[0:dim_list[0], 0:dim_list[0]] = Rt
        Rxx[dim_list[0]:, dim_list[0]:] = cov_stim
        Dxx = np.zeros_like(Rxx)
        Dxx[0:dim_list[0], 0:dim_list[0]] = Rw
        Dxx[dim_list[0]:, dim_list[0]:] = cov_stim
        try:
            lam, W = utils.transformed_GEVD(Dxx, Rxx, rho, dim_list[-1], self.n_components)
            Lam = np.diag(lam)
            Rxx[:,dim_list[0]:] = Rxx[:,dim_list[0]:]*rho
        except:
            print("Numerical issue exists for eigh. Use eig instead.")
            Rxx[:,dim_list[0]:] = Rxx[:,dim_list[0]:]*rho
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
        Weeg = W[0:dim_list[0],:]
        Weeg_expand = np.repeat(np.expand_dims(Weeg, axis=1), N, axis=1)
        Wstim = W[dim_list[0]:,:]
        # Forward models
        F_redun = T * Dxx @ W / N
        F = utils.F_organize(F_redun[0:dim_list[0],:], L_EEG, offset_EEG)
        # Shared subspace
        S = self.shared_subspace(EEG_center, Stim_center, Weeg, Wstim, Lam, rho)
        Wlist = [Weeg_expand, Wstim]
        return Wlist, S, F, lam

    def shared_subspace(self, EEG_center, Stim_center, Weeg, Wstim, Lam, rho):
        S = (np.sum(np.einsum('tdn,dk->tkn', EEG_center, Weeg), axis=2) + rho*Stim_center@Wstim) @ Lam
        return S

    def forward_model(self, EEG, Wlist, S=None):
        '''
        Input:
        EEG: EEG data with shape (T, D, N) [T: # sample, D: # channel, N: # subjects]
        Wlist: [Weeg_expand, Wstim]
        S: shared subspace with shape (T, n_components); if not None, then calculate forward model from the shared subspace
        Note: if S is not None, then EEG must be the one used in the training stage. So it is equivalent to the S generated by self.fit(). This can be used as a sanity check.
        Outputs:
        F: forward model (D, n_components)
        '''
        if np.ndim(EEG) == 2:
            EEG = np.expand_dims(EEG, axis=2)
        W = Wlist[0][:,0,:]
        _, _, N = EEG.shape
        X_list = [utils.block_Hankel(EEG[:,:,n], self.Llist[0], self.offsetlist[0]) for n in range(N)]
        X_list_center = [X_list[n] - np.mean(X_list[n], axis=0, keepdims=True) for n in range(N)]
        if S is not None:
            X_stack = np.stack(X_list_center, axis=2)
            F_T = np.mean(np.einsum('kt, tdn -> kdn', S.T, X_stack), axis=2)
            F_redun = F_T.T
        else:
            X = np.concatenate(tuple(X_list_center), axis=0)
            X_transformed = X @ W
            F_redun = (lstsq(X_transformed, X)[0]).T
        F = utils.F_organize(F_redun, self.Llist[0], self.offsetlist[0])
        return F


class LSGCCA:
    def __init__(self, EEG_list, Stim_list, fs, L_EEG, offset_EEG, L_Stim, offset_Stim, id_sub, corrca=False, fold=10, n_components=5, regularization='lwcov', message=True, signifi_level=True, pool=True, n_permu=1000, p_value=0.05):
        self.EEG_list = EEG_list
        self.Stim_list = Stim_list
        self.fs = fs
        self.L_EEG = L_EEG
        self.offset_EEG = offset_EEG
        self.L_Stim = L_Stim
        self.offset_Stim = offset_Stim
        self.id_sub = id_sub
        self.corrca = corrca
        self.fold = fold
        self.n_components = n_components
        self.regularization = regularization
        self.message = message
        self.signifi_level = signifi_level
        self.pool = pool
        self.n_permu = n_permu
        self.p_value = p_value

    def correlation(self, EEG, Stim, We, Ws):
        EEG_Hankel = utils.block_Hankel(EEG, self.L_EEG, self.offset_EEG)
        Stim_Hankel = utils.block_Hankel(Stim, self.L_Stim, self.offset_Stim)
        filtered_EEG = EEG_Hankel @ We
        filtered_Stim = Stim_Hankel @ Ws
        corr_pvalue = [pearsonr(filtered_EEG[:,k], filtered_Stim[:,k]) for k in range(self.n_components)]
        corr_coe = np.array([corr_pvalue[k][0] for k in range(self.n_components)])
        p_value = np.array([corr_pvalue[k][1] for k in range(self.n_components)])
        return corr_coe, p_value

    def permutation_test(self, EEG, Stim, We, Ws, block_len):
        corr_coe_topK = np.zeros((self.n_permu, self.n_components))
        for i in tqdm(range(self.n_permu)):
            EEG_shuffled = utils.shuffle_2D(EEG, block_len)
            Stim_shuffled = utils.shuffle_2D(Stim, block_len)
            corr_coe_topK[i,:], _, = self.correlation(EEG_shuffled, Stim_shuffled, We, Ws)
        return corr_coe_topK

    def to_latent_space(self):
        self.nested_train = []
        self.nested_test = []
        self.nested_We_train = []
        self.nested_S = []
        self.nested_F_train = []
        for idx in range(self.fold):
            train_list, test_list = utils.split_mm_balance([self.EEG_list, self.Stim_list], fold=self.fold, fold_idx=idx+1)
            if self.corrca:
                CorrCA = CorrelatedComponentAnalysis(self.EEG_list, self.fs, self.L_EEG, self.offset_EEG, n_components=self.n_components, regularization=self.regularization)
                _, We_train, S, F_train = CorrCA.fit(train_list[0])
                We_train = np.expand_dims(We_train, axis=1)
                We_train = np.repeat(We_train, train_list[0].shape[2], axis=1)
            else:
                GCCA = GeneralizedCCA(self.EEG_list, self.fs,self.L_EEG, self.offset_EEG, n_components=self.n_components, regularization=self.regularization)
                We_train, S, F_train, _ = GCCA.fit(train_list[0])
            self.nested_train.append(train_list)
            self.nested_test.append(test_list)
            self.nested_We_train.append(We_train)
            self.nested_S.append(S)
            self.nested_F_train.append(F_train)

    def cross_val(self):
        fold = self.fold
        n_components = self.n_components
        corr_train = np.zeros((fold, n_components))
        corr_test = np.zeros((fold, n_components))
        for idx in range(fold):
            train_list = self.nested_train[idx]
            test_list = self.nested_test[idx]
            We_train = self.nested_We_train[idx]
            S = self.nested_S[idx]
            F_train = self.nested_F_train[idx]
            # obtain the stimulus filters by least square regression
            LS = LeastSquares(self.EEG_list, self.Stim_list, self.fs, decoding=False, L_Stim=self.L_Stim, offset_Stim=self.offset_Stim)
            Ws_train, _, _ = LS.encoder(S, train_list[1])
            corr_train[idx,:], _ = self.correlation(train_list[0][:,:,self.id_sub], train_list[1], We_train[:,self.id_sub,:], Ws_train)
            corr_test[idx,:], _ = self.correlation(test_list[0][:,:,self.id_sub], test_list[1], We_train[:,self.id_sub,:], Ws_train)
        if self.signifi_level:
            if self.pool:
                corr_trials = self.permutation_test(test_list[0][:,:,self.id_sub], test_list[1], We_train[:,self.id_sub,:], Ws_train, block_len=1)
                corr_trials = np.sort(abs(corr_trials), axis=None)
                sig_idx = -int(self.n_permu*self.p_value*n_components)
                print('Significance level: {}'.format(corr_trials[sig_idx]))
            else:
                corr_trials = self.permutation_test(test_list[0][:,:,self.id_sub], test_list[1], We_train[:,self.id_sub,:], Ws_train, block_len=1)
                corr_trials = np.sort(abs(corr_trials), axis=0)
                sig_idx = -int(self.n_permu*self.p_value)
                print('Significance level of each component: {}'.format(corr_trials[sig_idx,:]))
        if self.message:
            print('Average correlation coefficients of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train, axis=0)))
            print('Average correlation coefficients of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test, axis=0)))
        return corr_train, corr_test, We_train, Ws_train, F_train