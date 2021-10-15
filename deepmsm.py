from typing import Optional, Union, Callable, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from itertools import chain


from deeptime.base import Model, Transformer
from deeptime.base_torch import DLEstimatorMixin
from deeptime.util.torch import map_data
from deeptime.markov.tools.analysis import pcca_memberships

CLIP_VALUE = 1.

def symeig_reg(mat, epsilon: float = 1e-6, mode='regularize', eigenvectors=True) \
        -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r""" Solves a eigenvector/eigenvalue decomposition for a hermetian matrix also if it is rank deficient.

    Parameters
    ----------
    mat : torch.Tensor
        the hermetian matrix
    epsilon : float, default=1e-6
        Cutoff for eigenvalues.
    mode : str, default='regularize'
        Whether to truncate eigenvalues if they are too small or to regularize them by taking the absolute value
        and adding a small positive constant. :code:`trunc` leads to truncation, :code:`regularize` leads to epsilon
        being added to the eigenvalues after taking the absolute value
    eigenvectors : bool, default=True
        Whether to compute eigenvectors.

    Returns
    -------
    (eigval, eigvec) : Tuple[torch.Tensor, Optional[torch.Tensor]]
        Eigenvalues and -vectors.
    """
    assert mode in sym_inverse.valid_modes, f"Invalid mode {mode}, supported are {sym_inverse.valid_modes}"

    if mode == 'regularize':
        identity = torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)
        mat = mat + epsilon * identity

    # Calculate eigvalues and potentially eigvectors
    eigval, eigvec = torch.symeig(mat, eigenvectors=True)

    if eigenvectors:
        eigvec = eigvec.transpose(0, 1)

    if mode == 'trunc':
        # Filter out Eigenvalues below threshold and corresponding Eigenvectors
        mask = eigval > epsilon
        eigval = eigval[mask]
        if eigenvectors:
            eigvec = eigvec[mask]
    elif mode == 'regularize':
        # Calculate eigvalues and eigvectors
        eigval = torch.abs(eigval)
    elif mode == 'clamp':
        eigval = torch.clamp_min(eigval, min=epsilon)

    else:
        raise RuntimeError("Invalid mode! Should have been caught by the assertion.")

    if eigenvectors:
        return eigval, eigvec
    else:
        return eigval, eigvec


def sym_inverse(mat, epsilon: float = 1e-6, return_sqrt=False, mode='regularize', return_both=False):
    """ Utility function that returns the inverse of a matrix, with the
    option to return the square root of the inverse matrix.

    Parameters
    ----------
    mat: numpy array with shape [m,m]
        Matrix to be inverted.
    epsilon : float
        Cutoff for eigenvalues.
    return_sqrt: bool, optional, default = False
        if True, the square root of the inverse matrix is returned instead
    mode: str, default='trunc'
        Whether to truncate eigenvalues if they are too small or to regularize them by taking the absolute value
        and adding a small positive constant. :code:`trunc` leads to truncation, :code:`regularize` leads to epsilon
        being added to the eigenvalues after taking the absolute value
    return_both: bool, default=False
        Whether to return the sqrt and its inverse or simply the inverse
    Returns
    -------
    x_inv: numpy array with shape [m,m]
        inverse of the original matrix
    """
    eigval, eigvec = symeig_reg(mat, epsilon, mode)

    # Build the diagonal matrix with the filtered eigenvalues or square
    # root of the filtered eigenvalues according to the parameter
    if return_sqrt:
        diag_inv = torch.diag(torch.sqrt(1. / eigval))
        if return_both:
            diag = torch.diag(torch.sqrt(eigval))
    else:
        diag_inv = torch.diag(1. / eigval)
        if return_both:
            diag = torch.diag(eigval)
    if not return_both:
        return torch.chain_matmul(eigvec.t(), diag_inv, eigvec)
    else:
        return torch.chain_matmul(eigvec.t(), diag_inv, eigvec), torch.chain_matmul(eigvec.t(), diag, eigvec)


sym_inverse.valid_modes = ('trunc', 'regularize', 'clamp')


def covariances(x: torch.Tensor, y: torch.Tensor, remove_mean: bool = True):
    """Computes instantaneous and time-lagged covariances matrices.

    Parameters
    ----------
    x : (T, n) torch.Tensor
        Instantaneous data.
    y : (T, n) torch.Tensor
        Time-lagged data.
    remove_mean: bool, default=True
        Whether to remove the mean of x and y.

    Returns
    -------
    cov_00 : (n, n) torch.Tensor
        Auto-covariance matrix of x.
    cov_0t : (n, n) torch.Tensor
        Cross-covariance matrix of x and y.
    cov_tt : (n, n) torch.Tensor
        Auto-covariance matrix of y.

    See Also
    --------
    deeptime.covariance.Covariance : Estimator yielding these kind of covariance matrices based on raw numpy arrays
                                     using an online estimation procedure.
    """

    assert x.shape == y.shape, "x and y must be of same shape"
    batch_size = x.shape[0]

    if remove_mean:
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

    # Calculate the cross-covariance
    y_t = y.transpose(0, 1)
    x_t = x.transpose(0, 1)
    cov_01 = 1 / (batch_size - 1) * torch.matmul(x_t, y)
    # Calculate the auto-correlations
    cov_00 = 1 / (batch_size - 1) * torch.matmul(x_t, x)
    cov_11 = 1 / (batch_size - 1) * torch.matmul(y_t, y)

    return cov_00, cov_01, cov_11


valid_score_methods = ('VAMP1', 'VAMP2', 'VAMPE')


class U_layer(torch.nn.Module):
    ''' Neural network layer which implements the reweighting vector of the reversible deep MSM. 
        The trainable weights reweight each configuration in the state space spanned by a VAMPnet to 
        a learned stationary distribution.

        Parameters
        ----------
        output_dim : int
            The output size of the VAMPnet.
        activation : function
            Activation function, where the trainable parameters are passed through. The function should map to the positive real axis

    '''

    def __init__(self, output_dim, activation):
        super(U_layer, self).__init__()

        self.M = output_dim
        # using 0.5414 with softplus results in u being constant, expecting a trajectory in equilibrium
        self.alpha = torch.Tensor(1, self.M).fill_(1.)
        self.u_kernel = torch.nn.Parameter(data=self.alpha, requires_grad=True)
        self.acti = activation

    def forward(self, chi_t, chi_tau, return_u=False, return_mu=False):
        r'''Call function of the layer. It maps the trainable parameters to the reweighting vector u and estimates
        the correlation function in equilibrium.
        
        Parameters
        ---------
        chi_t : torch.Tensor with shape [T x n], where $T$ is the number of frames and n the size of the feature space.
                Configurations at time $t$ mapped on the feature space. The function should represent a fuzzy clustering, i.e. each element
                is positive and the vector is normalized to 1 when summing all feature values.
        chi_tau : torch.Tensor with the same shape as chi_t.
                Configurations at time $t+\tau$ passed through the same functions as chi_t.
        return_u : bool, default=False.
                Whether to return the reweighting vector $u$. Necessary for building consequtive coarse-graining layers.
        return_mu: bool, default=False.
                Whether to return the stationary distribution $\mu$. Necessary when working with observables.
                
        Returns
        ---------
        v : torch.Tensor with shape [1,n].
                Necessary vector for normalizing the parameters for the S layer.
        C_00 : torch.Tensor with shape [n,n].
                Covariance matrix at time $t$
        C_11 : torch.Tensor with shape [n,n].
                Covariance matrix of the reweighted feature functions at time $t+\tau$
        C_01 : torch.Tensor with shape [n,n].
                Cross-correlation matrix between the feature vector at time $t$ with the reweighted one at time $t+\tau$
        Sigma : torch.Tensor with shape [n,n].
                Cross-correlation matrix between the feature wector at time $t+\tau$ and its reweighted form. 
                Necessary to estimate the transition matrix out of S.
        u : torch.Tensor with shape [1,n]. Only if return_u=True.
                The reweighting vector.
        mu : torch.Tensor with shape [T]. Only if return_mu=True.
                The stationary distribution of time shifed configurations.
                
        '''
        batchsize = chi_t.shape[0]
        # note: corr_tau is the correlation matrix of the time-shifted data
        # presented in the paper at page 6, "Normalization of transition density"
        corr_tau = 1. / batchsize * torch.matmul(chi_tau.T, chi_tau)
        chi_mean = torch.mean(chi_tau, dim=0, keepdim=True)

        kernel_u = self.acti(self.u_kernel)

        # u is the normalized and transformed kernel of this layer
        u = kernel_u / torch.sum(chi_mean * kernel_u, dim=1, keepdim=True)

        v = torch.matmul(corr_tau, u.T)
        # estimate the stationary distribution of x_t+tau
        mu = 1. / batchsize * torch.matmul(chi_tau, u.T)

        Sigma = torch.matmul((chi_tau * mu).T, chi_tau)

        # estimate the stationary distribtuion for x_t
        chi_mean_t = torch.mean(chi_t, dim=0, keepdim=True)

        gamma = chi_tau * (torch.matmul(chi_tau, u.T))

        C_00 = 1. / batchsize * torch.matmul(chi_t.T, chi_t)
        C_11 = 1. / batchsize * torch.matmul(gamma.T, gamma)
        C_01 = 1. / batchsize * torch.matmul(chi_t.T, gamma)
        ret = [
            v,
            C_00,
            C_11,
            C_01,
            Sigma,
        ]
        if return_u:
            ret += [
                u
            ]
        if return_mu:
            ret += [
                mu
            ]
            
        return ret


class S_layer(torch.nn.Module):
    ''' Neural network layer which implements the symmetric trainable matrix S of the reversible deep MSM. 
        The matrix S represents the transition matrix and a normalization matrix necessary to normalize the learned
        transition density.

        Parameters
        ----------
        output_dim : int
            The output size of the VAMPnet.
        activation : function
            Activation function, where the trainable parameters are passed through. The function should map to the positive real axis.
        renorm : bool, default=True
            Whether a elemtwise positive matrix S should be enforced. Necessary for a deep (reversible) MSM, but not for a Koopman model.

    '''

    def __init__(self, output_dim, activation, renorm=True):
        super(S_layer, self).__init__()

        self.M = output_dim

        self.alpha = torch.Tensor(self.M, self.M).fill_(0.1)
        self.S_kernel = torch.nn.Parameter(data=self.alpha, requires_grad=True)
        self.acti = activation
        self.renorm = renorm

    def forward(self, v, C_00, C_11, C_01, Sigma, return_K=False, return_S=False):
        r'''Call function of the layer. It maps the trainable parameters to the symmetric matrix S and estimates the VAMP-E score
        of the model.
        
        Parameters
        ---------
        v : torch.Tensor with shape [1,n].
                Necessary vector for normalizing the parameters for the S layer. It is part of the output of the u-layer.
        C_00 : torch.Tensor with shape [n,n]. It is part of the output of the u-layer.
                Covariance matrix at time $t$
        C_11 : torch.Tensor with shape [n,n]. It is part of the output of the u-layer.
                Covariance matrix of the reweighted feature functions at time $t+\tau$
        C_01 : torch.Tensor with shape [n,n]. It is part of the output of the u-layer.
                Cross-correlation matrix between the feature vector at time $t$ with the reweighted one at time $t+\tau$
        Sigma : torch.Tensor with shape [n,n]. It is part of the output of the u-layer.
                Cross-correlation matrix between the feature wector at time $t+\tau$ and its reweighted form. 
                Necessary to estimate the transition matrix out of S.
        return_K : bool, default=False.
                Whether to return the estimated transition matrix learned via S.
        return_S : bool, default=False.
                Whether to return the matrix S. Necessary for further calculations like coarse-graining.
                
        Returns
        ---------
        VAMP-E matrix : torch.Tensor with shape [n,n].
                The trace of the matrix is the VAMP-E score.
        K : torch.Tensor with shape [n,n]. Only if return_K=True.
                Transition matrix propagating the state space in time.
        S : torch.Tensor with shape [n,n]. Only if return_S=True.
                The learned matrix S.
                
        '''
        batchsize = v.shape[0]

        # transform the kernel weights
        kernel_w = self.acti(self.S_kernel)

        # enforce symmetry
        W1 = kernel_w + kernel_w.T

        # normalize the weights
        norm = W1 @ v

        w2 = (1 - torch.squeeze(norm)) / torch.squeeze(v)
        S_temp = W1 + torch.diag(w2)
        if self.renorm:

            # if (S_temp < 0).sum() > 0:  # check if actually non-negativity is violated

            # make sure that the largest value of norm is < 1
            quasi_inf_norm = lambda x: torch.sum((x ** 20)) ** (1. / 20)
            #             print(norm, quasi_inf_norm(norm))
            W1 = W1 / quasi_inf_norm(norm)
            norm = W1 @ v

            w2 = (1 - torch.squeeze(norm)) / torch.squeeze(v)
            S_temp = W1 + torch.diag(w2)

        S = S_temp

        # calculate K
        K = S @ Sigma

        # VAMP-E matrix for the computation of the loss
        VampE_matrix = S.T @ C_00 @ S @ C_11 - 2 * S.T @ C_01

        # stack outputs so that the first dimension is = batchsize, keras requirement

        ret = [VampE_matrix]
        
        if return_K:
            ret += [K]
        if return_S:
            ret += [S]

        return ret

    
class Coarse_grain(torch.nn.Module):
    r'''Layer to coarse grain a state space. The layer can be used with deep reversible MSM, but also for simple VAMPnets.
        
        Parameters
        ---------
        input_dim : int.
                The number of dimension of the previous layer which should be coarse grained to output_dim.
        output_dim : int.
                Number of dimension the system should be coarse grained to. Should be strictly smaller than input_dim.
       
        '''
    def __init__(self, input_dim, output_dim):
        super(Coarse_grain, self).__init__()

        self.N = input_dim
        self.M = output_dim
        
        self.alpha = torch.ones((self.N, self.M))
        self.weight = torch.nn.Parameter(data=self.alpha, requires_grad=True)
        
    def forward(self, x):
        '''
        Call function of the layer, which transforms x from input_dim dimensions to output_dim. Where each input state is probabilisticly assigned
        to each output state.
        
        Parameters
        ----------
        x : torch.Tensor of shape [T, input_dim].
            Fuzzy state assigned from a VAMPnet with a softmax output function, where T are the number of frames.
        
        Returns
        ---------
        y : torch.Tensor of shape [T, output_dim].
            Fuzzy state assignment but now with output_dim states.
        '''
        kernel = torch.softmax(self.weight, dim=1)
        
        ret = x @ kernel

        return ret
    
    def get_softmax(self):
        ''' 
        Helper function to plot the coarse-graining matrix.
        
        Returns:
            M : torch.Tensor of shape [input_dim, output_dim].
                Element (M)_ij is the probability that state i belongs to state j in the coarse grained representation.
        
        '''
        return torch.softmax(self.weight, dim=1)
    
    def get_cg_uS(self, chi_n, chi_tau_n, u_n, S_n, renorm=True, 
                  return_chi=False, return_K=False):
        r'''
        Coarse graining function, when using a deep reversible MSM. Since the coarse-grained representation should be consistant with 
        respect to the learned stationary distribution and transition density in the larger space, $u_m$ and $S_m$ of the coarse grained
        space can be estimated given $u_n$ and $S_n$ of the original feature space given the coarse-grain matrix.
        
        Parameters
        ---------
        chi_n : torch.Tensor of shape [T,n].
                The feature functions of the original space at time $t$.
        chi_tau_n : torch.Tensor of shape [T,n].
                The feature functions of the original space at time $t+\tau$.
        u_n :   torch.Tensor of shape [1,n].
                The reweighting vector of the original space.
        S_n :   torch.Tensor of shape [n,n].
                The symmetric matrix S of the original space.
        renorm : bool, default=True.
                Should be the same as for the original S-layer. Enforces positive elements in S.
        return_chi : bool, default=False. 
                Whether to return the new feature functions, $u_m$, and $S_m$. Necessary if a consequtive coarse-grain layer is implemented.
        return_K : bool, default=False.
                Whether to return the transition matrix in the coarse-grained state space.
                
        Returns
        ---------
        VampE_matrix : torch.Tensor of shape [m,m].
                Taking the trace of the VAMP-E matrix yields the VAMP-E score in the coarse grained space.
        chi_m : torch.Tensor of shape [T,m]. Only if return_chi=True.
                The feature functions of the coarse grained space at time $t$.
        chi_tau_m : torch.Tensor of shape [T,m]. Only if return_chi=True.
                The feature functions of the coarse grained space at time $t+\tau$.
        u_m :   torch.Tensor of shape [1,m]. Only if return_chi=True.
                The reweighting vecor in the coarse-grained space.
        S_m :   torch.Tensor of shape [m,m]. Only if return_chi=True.
                The matrix S in the coarse-grained space.
        K :     torch.Tensor of shape [m,m]. Only if return_K=True.
                Transition matrix in the coarse-grained space.
        
        '''
        
        batchsize = chi_n.shape[0]
        M = torch.softmax(self.weight, dim=1)
        
        chi_t_m = chi_n @ M
        chi_tau_m = chi_tau_n @ M
        
        # estimate the pseudo inverse of M
        U, S_vec, V = torch.svd(M)
        s_nonzero = S_vec > 0
        s_zero = S_vec <= 0
        S_star = torch.cat((1/S_vec[s_nonzero], S_vec[s_zero]))
        U_star = torch.cat((U[:,s_nonzero], U[:,s_zero]), dim=1)
        V_star = torch.cat((V[:,s_nonzero], V[:,s_zero]), dim=1)
        G = V_star @ torch.diag(S_star) @ U_star.T
        
        # estimate the new u and S
        u_m = (G @ u_n.T).T
        # renormalize
        u_m = torch.relu(u_m)
        chi_mean = torch.mean(chi_tau_m, dim=0, keepdim=True)
        u_m = u_m / torch.sum(chi_mean * u_m, dim=1, keepdim=True)
        
        W1 = G @ S_n @ G.T
        W1 = torch.relu(W1)
        #renormalize
        batchsize = chi_n.shape[0]
        corr_tau = 1./batchsize * torch.matmul(chi_tau_m.T, chi_tau_m)
        v = torch.matmul(corr_tau, u_m.T)
        norm = W1 @ v
        
        
        w2 = (1 - torch.squeeze(norm)) / torch.squeeze(v)
        S_temp = W1 + torch.diag(w2)
        if renorm:
            

            # make sure that the largest value of norm is < 1
            quasi_inf_norm = lambda x: torch.sum((x**20))**(1./20)
            W1 = W1 / quasi_inf_norm(norm)
            norm = W1 @ v

            w2 = (1 - torch.squeeze(norm)) / torch.squeeze(v)
            S_temp = W1 + torch.diag(w2)
                
                
        S_m = S_temp
        
        
        # estimate the VAMP-E matrix and other helpful instances
        mu = 1./batchsize * torch.matmul(chi_tau_m, u_m.T)
        Sigma =  torch.matmul((chi_tau_m * mu).T, chi_tau_m)

        gamma = chi_tau_m * (torch.matmul(chi_tau_m, u_m.T))

        C_00 = 1./batchsize * torch.matmul(chi_t_m.T, chi_t_m)
        C_11 = 1./batchsize * torch.matmul(gamma.T, gamma)
        C_01 = 1./batchsize * torch.matmul(chi_t_m.T, gamma)
        
        
        K = S_m @ Sigma

        # VAMP-E matrix for the computation of the loss
        VampE_matrix = S_m.T @ C_00 @ S_m @ C_11 - 2*S_m.T @ C_01
        ret = [VampE_matrix]
        if return_chi:
            ret += [
                chi_t_m, 
                chi_tau_m, 
                u_m,
                S_m,
            ]
        if return_K:
            ret += [
                K
            ]
            
        return ret
    
def vampe_loss_rev(chi_t, chi_tau, ulayer, slayer, return_mu=False, return_Sigma=False, return_K=False, return_S=False):
    '''
    VAMP-E score for a reversible deep MSM.
    
    Parameters
    ---------
    chi_t : torch.Tensor of shape [T,n].
            The fuzzy state assignment for all $T$ time frames at time $t$
    chi_tau : torch.Tensor of shape [T,n].
            The fuzzy state assignment for all $T$ time frames at time $t+\tau$
    ulayer : torch.nn.Module.
            The layer which implements the reweighting vector $u$.
    slayer : torch.nn.Module.
            The layer which implements the symmetric matrix $S$.
    return_mu : bool, default=False.
            Whether the stationary distribution should be returned.
    return_Sigma : bool, default=False.
            Whether the cross-correlation matrix should be returned.
    return_K : bool, default=False.
            Whether the transition matrix should be returned.
    return_S : bool, default=False.
            Whether the matrix S should be returned.
            
    Returns
    ---------
    vampe : torch.Tensor of shape [1,1] 
            VAMP-E score.
    K : torch.Tensor of shape [n,n]. Only if return_K=True.
            Transition matrix.
    S : torch.Tensor of shape [n,n]. Only if return_S=True.
            Symmetric matrix S.
    '''
    
    ret2 = []
    
    output_u = ulayer(chi_t, chi_tau, return_mu=return_mu)
    if return_mu:
        ret2.append(output_u[-1])
        
    if return_Sigma:
        ret2.append(output_u[4])
    
    output_S = slayer(*output_u[:5], return_K=return_K, return_S=return_S)
    vampe = torch.trace(output_S[0])
    ret1 = [-vampe]
    if return_K:        
        ret1.append(output_S[1])
    if return_S:
        ret1.append(output_S[-1])
    
    ret = ret1 + ret2
    
    return ret
    
def vampe_loss_rev_only_S(v, C_00, C_11, C_01, Sigma, slayer, return_K=False, return_S=False):
    
    output_S = slayer(v, C_00, C_11, C_01, Sigma, return_K=return_K, return_S=return_S)
#     print(K)
    vampe = torch.trace(output_S[0])
    ret = [-vampe]
    if return_K:
        ret.append(output_S[1])
    if return_S:
        ret.append(output_S[-1])
    return ret

def get_process_eigval(S, Sigma, state1, state2, epsilon=1e-6, mode='regularize'):  
    ''' state can be int or list of int'''
    Sigma_sqrt_inv, Sigma_sqrt = sym_inverse(Sigma, epsilon, return_sqrt=True, mode=mode, return_both=True)
    
    S_similar = Sigma_sqrt @ S @ Sigma_sqrt

    eigval_all, eigvec_all = torch.symeig(S_similar, eigenvectors=True)
    eigvecs_K = Sigma_sqrt_inv @ eigvec_all

    # Find the relevant process which is changing most between state1 and state2
    process_id = torch.argmin(eigvecs_K[state1,:]*eigvecs_K[state2,:], dim=1).detach()
    
    return eigval_all[process_id]

def obs_its_loss(S, Sigma, state1, state2, exp_value, lam, epsilon=1e-6, mode='regularize'):
    
    obs_value = get_process_eigval(S, Sigma, state1, state2, epsilon=epsilon, mode=mode)
    error = torch.sum(lam*torch.abs(exp_value - obs_value))
    
    return error, obs_value

def obs_ev(obs_value, mu):
    
    exp_value_estimated = torch.sum(obs_value * mu, dim=0)
    
    return exp_value_estimated


def obs_ev_loss(obs_value, mu, exp_value, lam):
    
    exp_value_estimated = obs_ev(obs_value, mu)
    
    error = torch.sum(lam*torch.abs(exp_value - exp_value_estimated))
    
    return error, exp_value_estimated

def obs_ac(obs_value, mu, chi, K, Sigma):

    state_weight = mu*chi

    pi = torch.sum(state_weight, dim=0) # prob to be in a state
    # obs value within a state, the weighting factor needs to be normalized for each state
    ai = torch.sum(state_weight[:,None,:]*obs_value[:,:,None], dim=0) / torch.sum(state_weight, dim=0)[None,:]
    # prob to observe an unconditional jump state i to j
    X = Sigma @ K
    a_sim = torch.sum(ai * torch.matmul(X, ai.T).T, dim=1)
    
    return a_sim

def obs_ac_loss(obs_value, mu, chi, K, Sigma, exp_value, lam):

    
    a_sim = obs_ac(obs_value, mu, chi, K, Sigma)
        
    error = torch.sum(lam*torch.abs(a_sim - exp_value))
           
    return error, a_sim



class DeepMSMModel(Transformer, Model):
    r"""
    A VAMPNet model which can be fit to data optimizing for one of the implemented VAMP scores.

    Parameters
    ----------
    lobe : torch.nn.Module
        One of the lobes of the VAMPNet. See also :class:`deeptime.util.torch.MLP`.
    lobe_timelagged : torch.nn.Module, optional, default=None
        The timelagged lobe. Can be left None, in which case the lobes are shared.
    dtype : data type, default=np.float32
        The data type for which operations should be performed. Leads to an appropriate cast within fit and
        transform methods.
    device : device, default=None
        The device for the lobe(s). Can be None which defaults to CPU.

    See Also
    --------
    VAMPNet : The corresponding estimator.
    """

    def __init__(self, lobe: nn.Module, ulayer, slayer, cg_list=None, mask=torch.nn.Identity(),
                 dtype=np.float32, device=None, epsilon=1e-6, mode='regularize'):
        super().__init__()
        self._lobe = lobe
        self._ulayer = ulayer
        self._slayer = slayer
        if cg_list is not None:
            self._cg_list = cg_list
        self.mask = mask
        if dtype == np.float32:
            self._lobe = self._lobe.float()
        elif dtype == np.float64:
            self._lobe = self._lobe.double()
        self._dtype = dtype
        self._device = device
        self._epsilon = epsilon
        self._mode = mode
        
    def transform(self, data, **kwargs):
        self._lobe.eval()
        net = self._lobe
        out = []
        for data_tensor in map_data(data, device=self._device, dtype=self._dtype):
            out.append(net(self.mask(data_tensor)).cpu().numpy())
        return out if len(out) > 1 else out[0]
    
    def get_mu(self, data_t):
        self._lobe.eval()
        net = self._lobe
        with torch.no_grad():
            x_t = net(self.mask(torch.Tensor(data_t).to(self._device)))
            mu = self._ulayer(x_t, x_t, return_mu=True)[-1] # use dummy x_0
        return mu.detach().to('cpu').numpy()
    
    def get_transition_matrix(self, data_0, data_t):
        self._lobe.eval()
        net = self._lobe
        with torch.no_grad():
            x_0 = net(self.mask(torch.Tensor(data_0).to(self._device)))
            x_t = net(self.mask(torch.Tensor(data_t).to(self._device)))
            _, K = vampe_loss_rev(x_0, x_t, self._ulayer, self._slayer, return_K=True)
            
        K = K.to('cpu').numpy().astype('float64') 
        # Converting to double precision destroys the normalization
        T = K / K.sum(axis=1)[:, None]
        return T
    
    def timescales(self, data_0, data_t, tau):
        
        T = self.get_transition_matrix(data_0, data_t)
        eigvals = np.linalg.eigvals(T)
        eigvals_sort = np.sort(eigvals)[:-1] # remove eigenvalue 1
        its = - tau/np.log(np.abs(eigvals_sort[::-1]))
        
        return its
    
    def get_transition_matrix_cg(self, data_0, data_t, idx=0):
        
        self._lobe.eval()
        net = self._lobe
        with torch.no_grad():
            chi_t = net(self.mask(torch.Tensor(data_0).to(self._device)))
            chi_tau = net(self.mask(torch.Tensor(data_t).to(self._device)))
            v, C00, Ctt, C0t, Sigma, u_n = self._ulayer(chi_t, chi_tau, return_u=True)
            _, S_n = self._slayer(v, C00, Ctt, C0t, Sigma, return_S=True)
                
            
            for cg_id in range(idx+1):
                _ , chi_t, chi_tau, u_n, S_n, K = self._cg_list[cg_id].get_cg_uS(chi_t, chi_tau, u_n, S_n, return_chi=True, return_K=True)
            
        K = K.to('cpu').numpy().astype('float64') 
        # Converting to double precision destroys the normalization
        T = K / K.sum(axis=1)[:, None]
        return T
    
    def timescales_cg(self, data_0, data_t, tau, idx=0):
        
        T = self.get_transition_matrix_cg(data_0, data_t, idx=idx)
        eigvals = np.linalg.eigvals(T)
        eigvals_sort = np.sort(eigvals)[:-1] # remove eigenvalue 1
        its = - tau/np.log(np.abs(eigvals_sort[::-1]))
        
        return its
    
    def observables(self, data_0, data_t, data_ev=None, data_ac=None, state1=None, state2=None):
        return_mu = False
        return_K = False
        return_S = False
        if data_ev is not None:
            return_mu = True
        if data_ac is not None:
            return_mu = True
            return_K = True
        if state1 is not None:
            return_S = True
        self._lobe.eval()
        net = self._lobe
        with torch.no_grad():
            x_0 = net(self.mask(torch.Tensor(data_0).to(self._device)))
            x_t = net(self.mask(torch.Tensor(data_t).to(self._device)))
            output_u = self._ulayer(x_0, x_t, return_mu=return_mu)
            if return_mu:
                mu = output_u[5]
            Sigma = output_u[4]
            output_S = self._slayer(*output_u[:5], return_K=return_K, return_S=return_S)
            if return_K:
                K = output_S[1]
            if return_S:
                S = output_S[-1]
            ret = []
            if data_ev is not None:
                x_ev = torch.Tensor(data_ev).to(self._device)
                ev_est = obs_ev(x_ev,mu)
                ret.append(ev_est.detach().to('cpu').numpy())
            if data_ac is not None:
                x_ac = torch.Tensor(data_ac).to(self._device)
                ac_est = obs_ac(x_ac, mu, x_t, K, Sigma)
                ret.append(ac_est.detach().to('cpu').numpy())
            if state1 is not None:
                its_est = get_process_eigval(S, Sigma, state1, state2, epsilon=self._epsilon, mode=self._mode)
                ret.append(its_est.detach().to('cpu').numpy())
        return ret

class DeepMSM(DLEstimatorMixin, Transformer):
    r""" Implementation of VAMPNets :cite:`vnet-mardt2018vampnets` which try to find an optimal featurization of
    data based on a VAMP score :cite:`vnet-wu2020variational` by using neural networks as featurizing transforms
    which are equipped with a loss that is the negative VAMP score. This estimator is also a transformer
    and can be used to transform data into the optimized space. From there it can either be used to estimate
    Markov state models via making assignment probabilities crisp (in case of softmax output distributions) or
    to estimate the Koopman operator using the :class:`VAMP <deeptime.decomposition.VAMP>` estimator.

    Parameters
    ----------
    lobe : torch.nn.Module
        A neural network module which maps input data to some (potentially) lower-dimensional space.
    lobe_timelagged : torch.nn.Module, optional, default=None
        Neural network module for timelagged data, in case of None the lobes are shared (structure and weights).
    device : torch device, default=None
        The device on which the torch modules are executed.
    optimizer : str or Callable, default='Adam'
        An optimizer which can either be provided in terms of a class reference (like `torch.optim.Adam`) or
        a string (like `'Adam'`). Defaults to Adam.
    learning_rate : float, default=5e-4
        The learning rate of the optimizer.
    score_method : str, default='VAMP2'
        The scoring method which is used for optimization.
    score_mode : str, default='regularize'
        The mode under which inverses of positive semi-definite matrices are estimated. Per default, the matrices
        are perturbed by a small constant added to the diagonal. This makes sure that eigenvalues are not too
        small. For a complete list of modes, see :meth:`sym_inverse`.
    epsilon : float, default=1e-6
        The strength of the regularization under which matrices are inverted. Meaning depends on the score_mode,
        see :meth:`sym_inverse`.
    dtype : dtype, default=np.float32
        The data type of the modules and incoming data.
    shuffle : bool, default=True
        Whether to shuffle data during training after each epoch.

    See Also
    --------
    deeptime.decomposition.VAMP

    References
    ----------
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: vnet-
    """
    _MUTABLE_INPUT_DATA = True

    def __init__(self, lobe: nn.Module, output_dim: int, coarse_grain: list = None, mask=None,
                 device=None, optimizer: Union[str, Callable] = 'Adam', learning_rate: float = 5e-4,
                 score_mode: str = 'regularize', epsilon: float = 1e-6,
                 dtype=np.float32, shuffle: bool = True):
        super().__init__()
        
        self.lobe = lobe
        self.output_dim = output_dim
        self.coarse_grain = coarse_grain
        self.ulayer = U_layer(output_dim=output_dim, activation=torch.nn.ReLU()).to(device)
        self.slayer = S_layer(output_dim=output_dim, activation=torch.nn.ReLU(), renorm=True).to(device)
        if self.coarse_grain is not None:
            self.cg_list = []
            self.cg_opt_list = []
            for i, dim_out in enumerate(self.coarse_grain):
                if i==0:
                    dim_in = self.output_dim
                else:
                    dim_in = self.coarse_grain[i-1]
                self.cg_list.append(Coarse_grain(dim_in, dim_out).to(device))
                self.cg_opt_list.append(torch.optim.Adam(self.cg_list[-1].parameters(), lr=0.1))
        else:
            self.cg_list=None
        if mask is not None:
            self.mask = mask
            self.optimizer_mask = torch.optim.Adam(self.mask.parameters(), lr=self.learning_rate)
        else:
            self.mask = torch.nn.Identity()
            self.optimizer_mask = None
        self.score_mode = score_mode
        self._step = 0
        self.shuffle = shuffle
        self._epsilon = epsilon
        self.device = device
        self.learning_rate = learning_rate
        self.dtype = dtype
        self.setup_optimizer(optimizer, list(self.lobe.parameters()))
        self.optimizer_u = torch.optim.Adam(self.ulayer.parameters(), lr=self.learning_rate*10)
        self.optimizer_s = torch.optim.Adam(self.slayer.parameters(), lr=self.learning_rate*100)
        self.optimizer_lobe = torch.optim.Adam(self.lobe.parameters(), lr=self.learning_rate)
        self.optimimzer_all = torch.optim.Adam(chain(self.ulayer.parameters(), self.slayer.parameters(), self.lobe.parameters()), lr=self.learning_rate)
        self._train_scores = []
        self._validation_scores = []
        self._train_vampe = []
        self._train_ev = []
        self._train_ac = []
        self._train_its = []
        self._validation_vampe = []
        self._validation_ev = []
        self._validation_ac = []
        self._validation_its = []

    @property
    def train_scores(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._train_scores)
    @property
    def train_vampe(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._train_vampe)
    @property
    def train_ev(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.concatenate(self._train_ev).reshape(-1,self._train_ev[0].shape[0])
    @property
    def train_ac(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.concatenate(self._train_ac).reshape(-1,self._train_ac[0].shape[0])
    @property
    def train_its(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.concatenate(self._train_its).reshape(-1,self._train_its[0].shape[0])
    
    @property
    def validation_scores(self) -> np.ndarray:
        r""" The collected validation scores. First dimension contains the step, second dimension the score.
        Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._validation_scores)
    @property
    def validation_vampe(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._validation_vampe)
    @property
    def validation_ev(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.concatenate(self._validation_ev).reshape(-1,self._validation_ev[0].shape[0])
    @property
    def validation_ac(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.concatenate(self._validation_ac).reshape(-1,self._validation_ac[0].shape[0])
    @property
    def validation_its(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.concatenate(self._validation_its).reshape(-1,self._validation_its[0].shape[0])
    @property
    def epsilon(self) -> float:
        r""" Regularization parameter for matrix inverses.

        :getter: Gets the currently set parameter.
        :setter: Sets a new parameter. Must be non-negative.
        :type: float
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        assert value >= 0
        self._epsilon = value

    @property
    def score_method(self) -> str:
        r""" Property which steers the scoring behavior of this estimator.

        :getter: Gets the current score.
        :setter: Sets the score to use.
        :type: str
        """
        return self._score_method

    @score_method.setter
    def score_method(self, value: str):
        if value not in valid_score_methods:
            raise ValueError(f"Tried setting an unsupported scoring method '{value}', "
                             f"available are {valid_score_methods}.")
        self._score_method = value

    @property
    def lobe(self) -> nn.Module:
        r""" The instantaneous lobe of the VAMPNet.

        :getter: Gets the instantaneous lobe.
        :setter: Sets a new lobe.
        :type: torch.nn.Module
        """
        return self._lobe

    @lobe.setter
    def lobe(self, value: nn.Module):
        self._lobe = value
        if self.dtype == np.float32:
            self._lobe = self._lobe.float()
        else:
            self._lobe = self._lobe.double()
        self._lobe = self._lobe.to(device=self.device)
    
    def forward(self, data):
        
        if data.get_device():
            data = data.to(device=self.device)
        
        return self.lobe(self.mask(data))
                                              
    def partial_fit(self, data, mask: bool = False, train_score_callback: Callable[[int, torch.Tensor], None] = None,
                   tb_writer=None):
        r""" Performs a partial fit on data. This does not perform any batching.

        Parameters
        ----------
        data : tuple or list of length 2, containing instantaneous and timelagged data
            The data to train the lobe(s) on.
        train_score_callback : callable, optional, default=None
            An optional callback function which is evaluated after partial fit, containing the current step
            of the training (only meaningful during a :meth:`fit`) and the current score as torch Tensor.

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """

        if self.dtype == np.float32:
            self._lobe = self._lobe.float()
        elif self.dtype == np.float64:
            self._lobe = self._lobe.double()

        self.lobe.train()
        assert isinstance(data, (list, tuple)) and len(data) == 2, \
            "Data must be a list or tuple of batches belonging to instantaneous " \
            "and respective time-lagged data."

        batch_0, batch_t = data[0], data[1]

        if isinstance(data[0], np.ndarray):
            batch_0 = torch.from_numpy(data[0].astype(self.dtype)).to(device=self.device)
        if isinstance(data[1], np.ndarray):
            batch_t = torch.from_numpy(data[1].astype(self.dtype)).to(device=self.device)

        self.optimizer_lobe.zero_grad()
        self.optimizer_u.zero_grad()
        self.optimizer_s.zero_grad()
        if self.optimizer_mask is not None and mask:
            self.optimizer_mask.zero_grad()
        x_0 = self.forward(batch_0)
        x_t = self.forward(batch_t)
        
        loss_value = -vampe_loss_rev(x_0, x_t, self.ulayer, self.slayer)[0]
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(chain(self.lobe.parameters(), self.mask.parameters(), self.ulayer.parameters(), self.slayer.parameters()), CLIP_VALUE)
        if self.mask is not None and mask:
            self.optimizer_mask.step()
        self.optimizer_lobe.step()
        self.optimizer_u.step()
        self.optimizer_s.step()

        if train_score_callback is not None:
            lval_detached = loss_value.detach()
            train_score_callback(self._step, -lval_detached)
        if tb_writer is not None:
            tb_writer.add_scalars('Loss', {'train': loss_value.item()}, self._step)
            tb_writer.add_scalars('VAMPE', {'train': -loss_value.item()}, self._step)
        self._train_scores.append((self._step, (-loss_value).item()))
        self._step += 1

        return self
                                              
    def validate(self, validation_data: Tuple[torch.Tensor]) -> torch.Tensor:
        r""" Evaluates the currently set lobe(s) on validation data and returns the value of the configured score.

        Parameters
        ----------
        validation_data : Tuple of torch Tensor containing instantaneous and timelagged data
            The validation data.

        Returns
        -------
        score : torch.Tensor
            The value of the score.
        """
        self.lobe.eval()

        with torch.no_grad():
            val = self.forward(validation_data[0])
            val_t = self.forward(validation_data[1])
            score_value = vampe_loss_rev(val, val_t, self.ulayer, self.slayer)[0]
            return score_value

                                              
    def fit(self, data_loader: torch.utils.data.DataLoader, n_epochs=1, validation_loader=None,
            train_mode='all', mask=False,
            train_score_callback: Callable[[int, torch.Tensor], None] = None,
            validation_score_callback: Callable[[int, torch.Tensor], None] = None,
            tb_writer=None, **kwargs):
        r""" Fits a VampNet on data.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data to use for training. Should yield a tuple of batches representing
            instantaneous and time-lagged samples.
        n_epochs : int, default=1
            The number of epochs (i.e., passes through the training data) to use for training.
        validation_loader : torch.utils.data.DataLoader, optional, default=None
            Validation data, should also be yielded as a two-element tuple.
        train_mode : str, default='all'
            'all': training for lobe, u, and s
            'us' : training for u and s
            's'  : training for s
        train_score_callback : callable, optional, default=None
            Callback function which is invoked after each batch and gets as arguments the current training step
            as well as the score (as torch Tensor).
        validation_score_callback : callable, optional, default=None
            Callback function for validation data. Is invoked after each epoch if validation data is given
            and the callback function is not None. Same as the train callback, this gets the 'step' as well as
            the score.
        **kwargs
            Optional keyword arguments for scikit-learn compatibility

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
        self._step = 0        
        
        # and train
        if train_mode=='all':
            for epoch in range(n_epochs):
                for batch_0, batch_t in data_loader:
                    self.partial_fit((batch_0, batch_t), mask=mask,
                                     train_score_callback=train_score_callback, tb_writer=tb_writer)
                if validation_loader is not None:
                    with torch.no_grad():
                        scores = []
                        for val_batch in validation_loader:
                            scores.append(
                                self.validate((val_batch[0], val_batch[1]))
                            )
                        mean_score = torch.mean(torch.stack(scores))
                        self._validation_scores.append((self._step, mean_score.item()))
                        if tb_writer is not None:
                            tb_writer.add_scalars('Loss', {'valid': mean_score.item()}, self._step)
                            tb_writer.add_scalars('VAMPE',{'valid': -mean_score.item()}, self._step)
                        if validation_score_callback is not None:
                            validation_score_callback(self._step, mean_score)
        else:
            chi_t, chi_tau = [], []
            with torch.no_grad():
                for batch_0, batch_t in data_loader:
                    chi_t.append(self.forward(batch_0).detach())
                    chi_tau.append(self.forward(batch_t).detach())
                x_0 = torch.cat(chi_t, dim=0)
                x_t = torch.cat(chi_tau, dim=0)
                if validation_loader is not None:
                    chi_val_t, chi_val_tau = [], []
                    for batch_0, batch_t in validation_loader:
                        chi_val_t.append(self.forward(batch_0).detach())
                        chi_val_tau.append(self.forward(batch_t).detach())
                    x_val_0 = torch.cat(chi_val_t, dim=0)
                    x_val_t = torch.cat(chi_val_tau, dim=0)
            if train_mode=='us' or train_mode=='u':
                for epoch in range(n_epochs):
                    self.optimizer_u.zero_grad()
                    if train_mode=='us':
                        self.optimizer_s.zero_grad()
    
                    loss_value = -vampe_loss_rev(x_0, x_t, self.ulayer, self.slayer)[0]
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(chain(self.ulayer.parameters(), self.slayer.parameters()), CLIP_VALUE)
                    self.optimizer_u.step()
                    if train_mode=='us':
                        self.optimizer_s.step()
                    
                    if train_score_callback is not None:
                        lval_detached = loss_value.detach()
                        train_score_callback(self._step, -lval_detached)
                    self._train_scores.append((self._step, (-loss_value).item()))
                    if tb_writer is not None:
                        tb_writer.add_scalars('Loss', {'train': loss_value.item()}, self._step)
                        tb_writer.add_scalars('VAMPE', {'train': -loss_value.item()}, self._step)
                    if validation_loader is not None:
                        with torch.no_grad():
                            score_val = vampe_loss_rev(x_val_0, x_val_t, self.ulayer, self.slayer)[0]
                            self._validation_scores.append((self._step, score_val.item()))
                            if tb_writer is not None:
                                tb_writer.add_scalars('Loss', {'valid': score_val.item()}, self._step)
                                tb_writer.add_scalars('VAMPE', {'valid': -score_val.item()}, self._step)
                            if validation_score_callback is not None:
                                validation_score_callback(self._step, score_val)
                    self._step += 1
            if train_mode=='s':
                with torch.no_grad():
                    v, C_00, C_11, C_01, Sigma = self.ulayer(x_0, x_t)
                    v_val, C_00_val, C_11_val, C_01_val, Sigma_val = self.ulayer(x_val_0, x_val_t)
                for epoch in range(n_epochs):
                    self.optimizer_s.zero_grad()
                    
                    loss_value = -vampe_loss_rev_only_S(v, C_00, C_11, C_01, Sigma, self.slayer)[0]
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.slayer.parameters(), CLIP_VALUE)
                    self.optimizer_s.step()

                    if train_score_callback is not None:
                        lval_detached = loss_value.detach()
                        train_score_callback(self._step, -lval_detached)
                    self._train_scores.append((self._step, (-loss_value).item()))
                    if tb_writer is not None:
                        tb_writer.add_scalars('Loss', {'train': loss_value.item()}, self._step)
                        tb_writer.add_scalars('VAMPE', {'train': -loss_value.item()}, self._step)
                    if validation_loader is not None:
                        with torch.no_grad():
                            score_val = vampe_loss_rev_only_S(v_val, C_00_val, C_11_val, C_01_val, Sigma_val, self.slayer)[0]
                            self._validation_scores.append((self._step, score_val.item()))
                            if tb_writer is not None:
                                tb_writer.add_scalars('Loss', {'valid': score_val.item()}, self._step)
                                tb_writer.add_scalars('VAMPE', {'valid': -score_val.item()}, self._step)
                            if validation_score_callback is not None:
                                validation_score_callback(self._step, score_val)
                    self._step += 1                        
            
        return self
    
    def fit_routine(self, data_loader: torch.utils.data.DataLoader, n_epochs=1, validation_loader=None,
            rel=1e-4, reset_u=False, max_iter=100, mask=False,
            train_score_callback: Callable[[int, torch.Tensor], None] = None,
            validation_score_callback: Callable[[int, torch.Tensor], None] = None, tb_writer=None, **kwargs):
        r""" Fits a VampNet on data.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data to use for training. Should yield a tuple of batches representing
            instantaneous and time-lagged samples.
        n_epochs : int, default=1
            The number of epochs (i.e., passes through the training data) to use for training.
        validation_loader : torch.utils.data.DataLoader, optional, default=None
            Validation data, should also be yielded as a two-element tuple.
        train_mode : str, default='all'
            'all': training for lobe, u, and s
            'us' : training for u and s
            's'  : training for s
        train_score_callback : callable, optional, default=None
            Callback function which is invoked after each batch and gets as arguments the current training step
            as well as the score (as torch Tensor).
        validation_score_callback : callable, optional, default=None
            Callback function for validation data. Is invoked after each epoch if validation data is given
            and the callback function is not None. Same as the train callback, this gets the 'step' as well as
            the score.
        **kwargs
            Optional keyword arguments for scikit-learn compatibility

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
        self._step = 0
        
        # and train
        for g in self.optimizer_lobe.param_groups:
            lr_chi = g['lr']
            g['lr'] = lr_chi/10 
        if self.optimizer_mask is not None:
            for g in self.optimizer_mask.param_groups:
                lr_mask = g['lr']
                g['lr'] = lr_mask/10
        for g in self.optimizer_u.param_groups:
            lr_u = g['lr']
        for g in self.optimizer_s.param_groups:
            lr_s = g['lr']                
        for epoch in range(n_epochs):
            for g in self.optimizer_u.param_groups:
                g['lr'] = lr_u/10
            for g in self.optimizer_s.param_groups:
                g['lr'] = lr_s/10
            for batch_0, batch_t in data_loader:
                self.partial_fit((batch_0, batch_t), mask=mask,
                                 train_score_callback=train_score_callback, tb_writer=tb_writer)
            
            
        
            chi_t, chi_tau = [], []
            with torch.no_grad():
                for batch_0, batch_t in data_loader:
                    chi_t.append(self.forward(batch_0).detach())
                    chi_tau.append(self.forward(batch_t).detach())
                x_0 = torch.cat(chi_t, dim=0)
                x_t = torch.cat(chi_tau, dim=0)
                score_value_before = vampe_loss_rev(x_0, x_t, self.ulayer, self.slayer)[0].detach()
            flag = True
            # reduce the learning rate of u and S
            for g in self.optimizer_u.param_groups:
                g['lr'] = lr_u/2
            for g in self.optimizer_s.param_groups:
                g['lr'] = lr_s/2
            counter = 0
            print('Score before loop', score_value_before.item())
            if reset_u:
                cov_00, cov_0t, cov_tt = covariances(x_0, x_t, remove_mean=False)
                cov_00_inv = sym_inverse(cov_00, epsilon=self.epsilon, mode=self.score_mode).to('cpu').numpy()

                K_vamp = (cov_00_inv @ cov_0t.to('cpu').numpy())

                # estimate pi, the stationary distribution vector
                eigv, eigvec = np.linalg.eig(K_vamp.T)
                ind_pi = np.argmin((eigv-1)**2)

                pi_vec = np.real(eigvec[:,ind_pi])
                pi = pi_vec / np.sum(pi_vec, keepdims=True)
                print('pi', pi)
                # reverse the consruction of u 
                u_optimal = cov_00_inv @ pi
                print('u optimal', u_optimal)

                # u_kernel = np.log(np.exp(np.abs(u_optimal))-1) # if softplus
                # for relu
                u_kernel = np.abs(u_optimal)

                with torch.no_grad():
                    for param in self.ulayer.parameters():

                        param.copy_(torch.Tensor(u_kernel[None,:]))  
            while flag:


                self.optimizer_u.zero_grad()
                self.optimizer_s.zero_grad()

                score = vampe_loss_rev(x_0, x_t, self.ulayer, self.slayer)[0]
                loss_value = -score
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(chain(self.ulayer.parameters(), self.slayer.parameters()), CLIP_VALUE)
                self.optimizer_u.step()
                self.optimizer_s.step()
                if (score-score_value_before) < rel and counter > 0:
                    flag=False
                counter+=1
                if counter > max_iter:
                    print('Reached max number of iterations')
                    flag=False
                score_value_before = score
            print('and after: ', score.item())   
            if validation_loader is not None:
                with torch.no_grad():
                    scores = []
                    for val_batch in validation_loader:
                        scores.append(
                            self.validate((val_batch[0], val_batch[1]))
                        )
                    mean_score = torch.mean(torch.stack(scores))
                    self._validation_scores.append((self._step, mean_score.item()))
                    if tb_writer is not None:
                        tb_writer.add_scalars('Loss', {'valid': mean_score.item()}, self._step)
                        tb_writer.add_scalars('VAMPE', {'valid': -mean_score.item()}, self._step)
                    if validation_score_callback is not None:
                        validation_score_callback(self._step, mean_score)                                
        for g in self.optimizer_lobe.param_groups:
            g['lr'] = lr_chi   
        if self.optimizer_mask is not None:
            for g in self.optimizer_mask.param_groups:
                g['lr'] = lr_mask
        return self
    
    def fit_cg(self, data_loader: torch.utils.data.DataLoader, n_epochs=1, validation_loader=None,
            train_mode='single', idx=0,
            train_score_callback: Callable[[int, torch.Tensor], None] = None,
            validation_score_callback: Callable[[int, torch.Tensor], None] = None, tb_writer=None, **kwargs):
        r""" Fits a VampNet on data.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data to use for training. Should yield a tuple of batches representing
            instantaneous and time-lagged samples.
        n_epochs : int, default=1
            The number of epochs (i.e., passes through the training data) to use for training.
        validation_loader : torch.utils.data.DataLoader, optional, default=None
            Validation data, should also be yielded as a two-element tuple.
        train_mode : str, default='all'
            'all': training u, and s and all coarse graining matrices
            'single' : training for coarse graining matrix of layer idx
        train_score_callback : callable, optional, default=None
            Callback function which is invoked after each batch and gets as arguments the current training step
            as well as the score (as torch Tensor).
        validation_score_callback : callable, optional, default=None
            Callback function for validation data. Is invoked after each epoch if validation data is given
            and the callback function is not None. Same as the train callback, this gets the 'step' as well as
            the score.
        **kwargs
            Optional keyword arguments for scikit-learn compatibility

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
        self._step = 0        
        
        # and train
        chi_t, chi_tau = [], []
        with torch.no_grad():
            for batch_0, batch_t in data_loader:
                chi_t.append(self.forward(batch_0).detach())
                chi_tau.append(self.forward(batch_t).detach())
            chi_t = torch.cat(chi_t, dim=0)
            chi_tau = torch.cat(chi_tau, dim=0)
            if validation_loader is not None:
                chi_val_t, chi_val_tau = [], []
                for batch_0, batch_t in validation_loader:
                    chi_val_t.append(self.forward(batch_0).detach())
                    chi_val_tau.append(self.forward(batch_t).detach())
                chi_val_t = torch.cat(chi_val_t, dim=0)
                chi_val_tau = torch.cat(chi_val_tau, dim=0)
        if train_mode=='all':
            
                
            for epoch in range(n_epochs):
                self.optimizer_u.zero_grad()
                self.optimizer_s.zero_grad()
                for opt in self.cg_opt_list:
                    opt.zero_grad()
                
                v, C00, Ctt, C0t, Sigma, u_n = self.ulayer(chi_t, chi_tau, return_u=True)
                matrix, S_n = self.slayer(v, C00, Ctt, C0t, Sigma, return_S=True)
                
                chi_t_n, chi_tau_n = chi_t, chi_tau
                loss_value = torch.trace(matrix)
                for cg_id in range(len(self.coarse_grain)):
                    matrix_cg , chi_t_n, chi_tau_n, u_n, S_n = self.cg_list[cg_id].get_cg_uS(chi_t_n, chi_tau_n, u_n, S_n, return_chi=True)
                    loss_value += torch.trace(matrix_cg)
                    
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(chain(self.ulayer.parameters(), self.slayer.parameters()), CLIP_VALUE)
                for lay_cg in self.cg_list:
                    torch.nn.utils.clip_grad_norm_(lay_cg.parameters(), CLIP_VALUE)
                self.optimizer_u.step()
                self.optimizer_s.step()
                for opt in self.cg_opt_list:
                    opt.step()
                 
                if train_score_callback is not None:
                    lval_detached = loss_value.detach()
                    train_score_callback(self._step, -lval_detached)
                self._train_scores.append((self._step, (-loss_value).item()))
                if tb_writer is not None:
                    tb_writer.add_scalars('Loss', {'cg_train': loss_value.item()}, self._step)
                    tb_writer.add_scalars('VAMPE', {'cg_train': -loss_value.item()}, self._step)
                
                if validation_loader is not None:
                    with torch.no_grad():
                        v, C00, Ctt, C0t, Sigma, u_n = self.ulayer(chi_val_t, chi_val_tau, return_u=True)
                        matrix, S_n = self.slayer(v, C00, Ctt, C0t, Sigma, return_S=True)
                        chi_val_t_n, chi_val_tau_n = chi_val_t, chi_val_tau
                        loss_value = torch.trace(matrix)
                        for cg_id in range(len(self.coarse_grain)):
                            matrix_cg , chi_val_t_n, chi_val_tau_n, u_n, S_n = self.cg_list[cg_id].get_cg_uS(chi_val_t_n, chi_val_tau_n, u_n, S_n, return_chi=True)
                            loss_value += torch.trace(matrix_cg)
                        score_val = -loss_value
                        self._validation_scores.append((self._step, score_val.item()))
                        if tb_writer is not None:
                            tb_writer.add_scalars('Loss', {'cg_valid': -score_val.item()}, self._step)
                            tb_writer.add_scalars('VAMPE', {'cg_valid': score_val.item()}, self._step)
                        if validation_score_callback is not None:
                            validation_score_callback(self._step, score_val)
                self._step += 1
        elif train_mode=='single':
            with torch.no_grad():
                v, C00, Ctt, C0t, Sigma, u_n = self.ulayer(chi_t, chi_tau, return_u=True)
                _, S_n = self.slayer(v, C00, Ctt, C0t, Sigma, return_S=True)
                
                if idx>0:
                    for cg_id in range(idx):
                        _ , chi_t, chi_tau, u_n, S_n = self.cg_list[cg_id].get_cg_uS(chi_t, chi_tau, u_n, S_n, return_chi=True)
                if validation_loader is not None:
                    v_val, C00_val, Ctt_val, C0t_val, Sigma_val, u_n_val = self.ulayer(chi_val_t, chi_val_tau, return_u=True)
                    _, S_n_val = self.slayer(v_val, C00_val, Ctt_val, C0t_val, Sigma_val, return_S=True)
                    for cg_id in range(idx):
                        _ , chi_val_t, chi_val_tau, u_n_val, S_n_val = self.cg_list[cg_id].get_cg_uS(chi_val_t, chi_val_tau, u_n_val, S_n_val, return_chi=True)
            for epoch in range(n_epochs):
                self.cg_opt_list[idx].zero_grad()
                matrix_cg = self.cg_list[idx].get_cg_uS(chi_t, chi_tau, u_n, S_n, return_chi=False)[0]
                loss_value = torch.trace(matrix_cg)
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(self.cg_list[idx].parameters(), CLIP_VALUE)
                self.cg_opt_list[idx].step()
                
                if train_score_callback is not None:
                    lval_detached = loss_value.detach()
                    train_score_callback(self._step, -lval_detached)
                self._train_scores.append((self._step, (-loss_value).item()))
                if tb_writer is not None:
                    tb_writer.add_scalars('Loss', {'cg_train': loss_value.item()}, self._step)
                    tb_writer.add_scalars('VAMPE', {'cg_train': -loss_value.item()}, self._step)
                if validation_loader is not None:
                    with torch.no_grad():
                        matrix_cg = self.cg_list[idx].get_cg_uS(chi_val_t, chi_val_tau, u_n_val, S_n_val, return_chi=False)[0]

                        score_val = -torch.trace(matrix_cg)
                
                        self._validation_scores.append((self._step, score_val.item()))
                        if tb_writer is not None:
                                tb_writer.add_scalars('Loss', {'cg_valid': score_val.item()}, self._step)
                                tb_writer.add_scalars('VAMPE', {'cg_valid': -score_val.item()}, self._step)
                        if validation_score_callback is not None:
                            validation_score_callback(self._step, score_val)
            
                self._step += 1                        
            
        return self
    
    def partial_fit_obs(self, data, data_ev, data_ac, exp_ev=None, exp_ac=None, exp_its=None,
                                    lam_ev=None, lam_ac=None, lam_its=None, 
                                    its_state1=None, its_state2=None, mask=False,
                        train_score_callback: Callable[[int, torch.Tensor], None] = None, tb_writer=None):
        r""" Performs a partial fit on data. This does not perform any batching.

        Parameters
        ----------
        data : tuple or list of length 2, containing instantaneous and timelagged data
            The data to train the lobe(s) on.
        train_score_callback : callable, optional, default=None
            An optional callback function which is evaluated after partial fit, containing the current step
            of the training (only meaningful during a :meth:`fit`) and the current score as torch Tensor.

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """

        if self.dtype == np.float32:
            self._lobe = self._lobe.float()
        elif self.dtype == np.float64:
            self._lobe = self._lobe.double()

        self.lobe.train()
        assert isinstance(data, (list, tuple)) and len(data) == 2, \
            "Data must be a list or tuple of batches belonging to instantaneous " \
            "and respective time-lagged data."

        batch_0, batch_t = data[0], data[1]

        if isinstance(data[0], np.ndarray):
            batch_0 = torch.from_numpy(data[0].astype(self.dtype))
        if isinstance(data[1], np.ndarray):
            batch_t = torch.from_numpy(data[1].astype(self.dtype))
        
        return_mu = False
        return_K = False
        return_Sigma = False
        return_S = False
        
        if exp_ev is not None:
            return_mu = True
            batch_ev = data_ev
            if isinstance(data_ev, np.ndarray):
                batch_ev = torch.from_numpy(data_ev.astype(self.dtype)).to(device=self.device)
            if isinstance(exp_ev, np.ndarray):
                exp_ev = torch.from_numpy(exp_ev.astype(self.dtype)).to(device=self.device)
            if isinstance(lam_ev, np.ndarray):
                lam_ev = torch.from_numpy(lam_ev.astype(self.dtype)).to(device=self.device)
        if exp_ac is not None:
            return_mu = True
            return_K = True
            return_Sigma = True
            batch_ac = data_ac
            if isinstance(data_ac, np.ndarray):
                batch_ac = torch.from_numpy(data_ac.astype(self.dtype)).to(device=self.device)
            if isinstance(exp_ac, np.ndarray):
                exp_ac = torch.from_numpy(exp_ac.astype(self.dtype)).to(device=self.device)
            if isinstance(lam_ac, np.ndarray):
                lam_ac = torch.from_numpy(lam_ac.astype(self.dtype)).to(device=self.device)
        if exp_its is not None:
            return_S = True
            return_Sigma =True
            if isinstance(exp_its, np.ndarray):
                exp_its = torch.from_numpy(exp_its.astype(self.dtype)).to(device=self.device)
            if isinstance(lam_its, np.ndarray):
                lam_its = torch.from_numpy(lam_its.astype(self.dtype)).to(device=self.device)
                
        
        self.optimizer_lobe.zero_grad()
        self.optimizer_u.zero_grad()
        self.optimizer_s.zero_grad()
        if mask and self.optimizer_mask is not None:
            self.optimizer_mask.zero_grad()
        x_0 = self.forward(batch_0)
        x_t = self.forward(batch_t)
        
            
        output_loss = vampe_loss_rev(x_0, x_t, self.ulayer, self.slayer, 
                                     return_mu=return_mu, return_Sigma=return_Sigma, return_K=return_K, return_S=return_S)
        vampe_loss = output_loss[0] 
        loss_value = - vampe_loss# vampe loss
        counter=1
        if return_K:
            K = output_loss[counter]
            counter += 1
        if return_S:
            S = output_loss[counter]
            counter += 1
        if return_mu:
            mu = output_loss[counter]
        if return_Sigma:
            Sigma = output_loss[-1]
        if exp_ev is not None:
            loss_ev, est_ev = obs_ev_loss(batch_ev, mu, exp_ev, lam_ev)
            loss_value += loss_ev
        if exp_ac is not None:
            loss_ac, est_ac = obs_ac_loss(batch_ac, mu, x_t, K, Sigma, exp_ac, lam_ac)
            loss_value += loss_ac
        if exp_its is not None:
            loss_its, est_its = obs_its_loss(S, Sigma, its_state1, its_state2, exp_its, lam_its, epsilon=self.epsilon, mode=self.score_mode)
            loss_value += loss_its
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(chain(self.lobe.parameters(), self.mask.parameters(), self.ulayer.parameters(), self.slayer.parameters()), CLIP_VALUE)
        self.optimizer_lobe.step()
        self.optimizer_u.step()
        self.optimizer_s.step()
        if mask and self.optimizer_mask is not None:
            self.optimizer_mask.step()

        if train_score_callback is not None:
            lval_detached = loss_value.detach()
            train_score_callback(self._step, lval_detached)
        self._train_scores.append((self._step, (loss_value).item()))
        self._train_vampe.append((self._step, (vampe_loss).item()))
        if tb_writer is not None:
            tb_writer.add_scalars('Loss', {'train': loss_value.item()}, self._step)
            tb_writer.add_scalars('VAMPE', {'train': vampe_loss.item()}, self._step)
        if exp_ev is not None:
            self._train_ev.append(np.concatenate(([self._step], (est_ev).detach().to('cpu').numpy())))
            if tb_writer is not None:
                for i in range(est_ev.shape[0]):
                    tb_writer.add_scalars('EV', {'train_'+str(i+1): est_ev[i].item()}, self._step)
        if exp_ac is not None:
            self._train_ac.append(np.concatenate(([self._step], (est_ac).detach().to('cpu').numpy())))
            if tb_writer is not None:
                for i in range(est_ac.shape[0]):
                    tb_writer.add_scalars('AC', {'train_'+str(i+1): est_ac[i].item()}, self._step)
        if exp_its is not None:
            self._train_its.append(np.concatenate(([self._step], (est_its).detach().to('cpu').numpy())))
            if tb_writer is not None:
                for i in range(est_its.shape[0]):
                    tb_writer.add_scalars('ITS', {'train_'+str(i+1): est_its[i].item()}, self._step)
        self._step += 1

        return self
                                              
    def validate_obs(self, validation_data: Tuple[torch.Tensor], val_data_ev=None, val_data_ac=None,
                    exp_ev=None, exp_ac=None, exp_its=None,
                    lam_ev=None, lam_ac=None, lam_its=None, 
                    its_state1=None, its_state2=None) -> torch.Tensor:
        r""" Evaluates the currently set lobe(s) on validation data and returns the value of the configured score.

        Parameters
        ----------
        validation_data : Tuple of torch Tensor containing instantaneous and timelagged data
            The validation data.

        Returns
        -------
        score : torch.Tensor
            The value of the score.
        """
        self.lobe.eval()
        return_mu = False
        return_K = False
        return_Sigma = False
        return_S = False
        
        if exp_ev is not None:
            return_mu = True
            batch_ev = val_data_ev
            if isinstance(val_data_ev, np.ndarray):
                batch_ev = torch.from_numpy(val_data_ev.astype(self.dtype)).to(device=self.device)
            if isinstance(exp_ev, np.ndarray):
                exp_ev = torch.from_numpy(exp_ev.astype(self.dtype)).to(device=self.device)
            if isinstance(lam_ev, np.ndarray):
                lam_ev = torch.from_numpy(lam_ev.astype(self.dtype)).to(device=self.device)
        if exp_ac is not None:
            return_mu = True
            return_K = True
            return_Sigma = True
            batch_ac = val_data_ac
            if isinstance(val_data_ac, np.ndarray):
                batch_ac = torch.from_numpy(val_data_ac.astype(self.dtype)).to(device=self.device)
            if isinstance(exp_ac, np.ndarray):
                exp_ac = torch.from_numpy(exp_ac.astype(self.dtype)).to(device=self.device)
            if isinstance(lam_ac, np.ndarray):
                lam_ac = torch.from_numpy(lam_ac.astype(self.dtype)).to(device=self.device)
        if exp_its is not None:
            return_S = True
            return_Sigma =True
            if isinstance(exp_its, np.ndarray):
                exp_its = torch.from_numpy(exp_its.astype(self.dtype)).to(device=self.device)
            if isinstance(lam_its, np.ndarray):
                lam_its = torch.from_numpy(lam_its.astype(self.dtype)).to(device=self.device)
        with torch.no_grad():
            val = self.forward(validation_data[0])
            val_t = self.forward(validation_data[1])
            output_loss = vampe_loss_rev(val, val_t, self.ulayer, self.slayer, 
                                     return_mu=return_mu, return_Sigma=return_Sigma, return_K=return_K, return_S=return_S)
            vampe_loss = output_loss[0]
            score_value = -vampe_loss # vampe loss
            ret = [score_value, vampe_loss]
            counter=1
            if return_K:
                K = output_loss[counter]
                counter += 1
            if return_S:
                S = output_loss[counter]
                counter += 1
            if return_mu:
                mu = output_loss[counter]
            if return_Sigma:
                Sigma = output_loss[-1]
            if exp_ev is not None:
                loss_ev, est_ev = obs_ev_loss(batch_ev, mu, exp_ev, lam_ev)
                score_value += loss_ev
                ret.append(est_ev)
            if exp_ac is not None:
                loss_ac, est_ac = obs_ac_loss(batch_ac, mu, val_t, K, Sigma, exp_ac, lam_ac)
                score_value += loss_ac
                ret.append(est_ac)
            if exp_its is not None:
                loss_its, est_its = obs_its_loss(S, Sigma, its_state1, its_state2, exp_its, lam_its, epsilon=self.epsilon, mode=self.score_mode)
                score_value += loss_its
                ret.append(est_its)
            return ret

                                              
    def fit_obs(self, data_loader: torch.utils.data.DataLoader, n_epochs=1, validation_loader=None,
            train_mode = 'all', exp_ev=None, exp_ac=None, exp_its=None,
            lam_ev=None, lam_ac=None, lam_its=None, its_state1=None, its_state2=None, mask=False,
            train_score_callback: Callable[[int, torch.Tensor], None] = None,
            validation_score_callback: Callable[[int, torch.Tensor], None] = None, tb_writer=None, **kwargs):
        r""" Fits a VampNet on data.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data to use for training. Should yield a tuple of batches representing
            instantaneous and time-lagged samples.
        n_epochs : int, default=1
            The number of epochs (i.e., passes through the training data) to use for training.
        validation_loader : torch.utils.data.DataLoader, optional, default=None
            Validation data, should also be yielded as a two-element tuple.
        train_mode : str, default='all'
            'all': training for lobe, u, and s
            'us' : training for u and s
            's'  : training for s
        train_score_callback : callable, optional, default=None
            Callback function which is invoked after each batch and gets as arguments the current training step
            as well as the score (as torch Tensor).
        validation_score_callback : callable, optional, default=None
            Callback function for validation data. Is invoked after each epoch if validation data is given
            and the callback function is not None. Same as the train callback, this gets the 'step' as well as
            the score.
        **kwargs
            Optional keyword arguments for scikit-learn compatibility

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
        self._step = 0        
        if exp_ev is not None:
            if isinstance(exp_ev, list):
                exp_ev = np.array(exp_ev)
            if isinstance(lam_ev, list):
                lam_ev = np.array(lam_ev)
            if isinstance(exp_ev, np.ndarray):
                exp_ev = torch.from_numpy(exp_ev.astype(self.dtype)).to(device=self.device)
            if isinstance(lam_ev, np.ndarray):
                lam_ev = torch.from_numpy(lam_ev.astype(self.dtype)).to(device=self.device)
        if exp_ac is not None:
            if isinstance(exp_ac, list):
                exp_ac = np.array(exp_ac)
            if isinstance(lam_ac, list):
                lam_ac = np.array(lam_ac)
            if isinstance(exp_ac, np.ndarray):
                exp_ac = torch.from_numpy(exp_ac.astype(self.dtype)).to(device=self.device)
            if isinstance(lam_ac, np.ndarray):
                lam_ac = torch.from_numpy(lam_ac.astype(self.dtype)).to(device=self.device)
        if exp_its is not None:
            if isinstance(exp_its, list):
                exp_its = np.array(exp_its)
            if isinstance(lam_its, list):
                lam_its = np.array(lam_its)
            if isinstance(exp_its, np.ndarray):
                exp_its = torch.from_numpy(exp_its.astype(self.dtype)).to(device=self.device)
            if isinstance(lam_its, np.ndarray):
                lam_its = torch.from_numpy(lam_its.astype(self.dtype)).to(device=self.device)
        # and train
        if train_mode=='all':
            for epoch in range(n_epochs):
                for batch in data_loader:
                    batch_0, batch_t = batch[0], batch[1]
                    if exp_ev is not None:
                        batch_ev = batch[2].to(device=self.device)
                    else:
                        batch_ev = None
                    if exp_ac is not None:
                        batch_ac = batch[-1].to(device=self.device)
                    else:
                        batch_ac = None
                    self.partial_fit_obs((batch_0, batch_t),
                                         data_ev=batch_ev, data_ac=batch_ac,
                                     exp_ev=exp_ev, exp_ac=exp_ac, exp_its=exp_its,
                                     lam_ev=lam_ev, lam_ac=lam_ac, lam_its=lam_its, 
                                     its_state1=its_state1, its_state2=its_state2, mask=mask,
                                     train_score_callback=train_score_callback, tb_writer=tb_writer)
                if validation_loader is not None:
                    with torch.no_grad():
                        scores = []
                        scores_vampe = []
                        idx_ac = 2
                        if exp_ev is not None:
                            scores_ev = []
                            idx_ac = 3
                        if exp_ac is not None:
                            scores_ac = []
                        if exp_its is not None:
                            scores_its = []
                        for val_batch in validation_loader:
                            if exp_ev is not None:
                                data_val_ev = val_batch[2].to(device=self.device)
                            else:
                                data_val_ev = None
                            if exp_ac is not None:
                                data_val_ac = val_batch[-1].to(device=self.device)
                            else:
                                data_val_ac = None
                            all_scores= self.validate_obs((val_batch[0].to(device=self.device), val_batch[1].to(device=self.device)),
                                             val_data_ev=data_val_ev, val_data_ac=data_val_ac,
                                             exp_ev=exp_ev, exp_ac=exp_ac, exp_its=exp_its,
                                             lam_ev=lam_ev, lam_ac=lam_ac, lam_its=lam_its, 
                                             its_state1=its_state1, its_state2=its_state2)
                            scores.append(all_scores[0])
                            scores_vampe.append(all_scores[1])
                            if exp_ev is not None:
                                scores_ev.append(all_scores[2])
                            if exp_ac is not None:
                                scores_ac.append(all_scores[idx_ac])
                            if exp_its is not None:
                                scores_its.append(all_scores[-1])
                            
                        mean_score = torch.mean(torch.stack(scores))
                        self._validation_scores.append((self._step, mean_score.item()))
                        mean_vampe = torch.mean(torch.stack(scores_vampe))
                        self._validation_vampe.append((self._step, mean_vampe.item()))
                        if tb_writer is not None:
                            tb_writer.add_scalars('Loss', {'valid': mean_score.item()}, self._step)
                            tb_writer.add_scalars('VAMPE', {'valid': mean_vampe.item()}, self._step)
                        if exp_ev is not None:
                            mean_ev = torch.mean(torch.stack(scores_ev), dim=0)
                            self._validation_ev.append(np.concatenate(([self._step], (mean_ev).detach().to('cpu').numpy())))
                            if tb_writer is not None:
                                for i in range(mean_ev.shape[0]):
                                    tb_writer.add_scalars('EV', {'valid_'+str(i+1): mean_ev[i].item()}, self._step)
                        if exp_ac is not None:
                            mean_ac = torch.mean(torch.stack(scores_ac), dim=0)
                            self._validation_ac.append(np.concatenate(([self._step], (mean_ac).detach().to('cpu').numpy())))
                            if tb_writer is not None:
                                for i in range(mean_ac.shape[0]):
                                    tb_writer.add_scalars('AC', {'valid_'+str(i+1): mean_ac[i].item()}, self._step)
                        if exp_its is not None:
                            mean_its = torch.mean(torch.stack(scores_its), dim=0)
                            self._validation_its.append(np.concatenate(([self._step], (mean_its).detach().to('cpu').numpy())))
                            if tb_writer is not None:
                                for i in range(mean_its.shape[0]):
                                    tb_writer.add_scalars('ITS', {'valid_'+str(i+1): mean_its[i].item()}, self._step)
                        if validation_score_callback is not None:
                            validation_score_callback(self._step, mean_score)
        else:
            return_mu = False
            return_K = False
            return_Sigma = False
            return_S = False
            chi_t, chi_tau = [], []
            if exp_ev is not None:
                data_ev = []
                return_mu = True
            if exp_ac is not None:
                data_ac = []
                return_mu = True
                return_K = True
                return_Sigma = True
            if exp_its is not None:
                return_S = True
                return_Sigma = True
            with torch.no_grad():
                for batch in data_loader:
                    batch_0, batch_t = batch[0], batch[1]
                    chi_t.append(self.forward(batch_0).detach())
                    chi_tau.append(self.forward(batch_t).detach())
                    if exp_ev is not None:
                        data_ev.append(batch[2].to(device=self.device))
                    if exp_ac is not None:
                        data_ac.append(batch[-1].to(device=self.device))
                x_0 = torch.cat(chi_t, dim=0)
                x_t = torch.cat(chi_tau, dim=0)
                if exp_ev is not None:
                    x_ev = torch.cat(data_ev, dim=0)
                if exp_ac is not None:
                    x_ac = torch.cat(data_ac, dim=0)
                if validation_loader is not None:
                    chi_val_t, chi_val_tau = [], []
                    if exp_ev is not None:
                        data_val_ev = []
                    if exp_ac is not None:
                        data_val_ac = []
                    for batch in validation_loader:
                        batch_0, batch_t = batch[0], batch[1]
                        chi_val_t.append(self.forward(batch_0).detach())
                        chi_val_tau.append(self.forward(batch_t).detach())
                        if exp_ev is not None:
                            data_val_ev.append(batch[2].to(device=self.device))
                        if exp_ac is not None:
                            data_val_ac.append(batch[-1].to(device=self.device))
                    x_val_0 = torch.cat(chi_val_t, dim=0)
                    x_val_t = torch.cat(chi_val_tau, dim=0)
                    if exp_ev is not None:
                        x_val_ev = torch.cat(data_val_ev, dim=0)
                    if exp_ac is not None:
                        x_val_ac = torch.cat(data_val_ac, dim=0)
            if train_mode=='us' or train_mode=='u':
                for epoch in range(n_epochs):
                    self.optimizer_u.zero_grad()
                    if train_mode=='us':
                        self.optimizer_s.zero_grad()
    
                    output_loss = vampe_loss_rev(x_0, x_t, self.ulayer, self.slayer, 
                                     return_mu=return_mu, return_Sigma=return_Sigma, return_K=return_K, return_S=return_S)
                    vampe_loss = output_loss[0]
                    loss_value = -vampe_loss # vampe loss
                    counter=1
                    if return_K:
                        K = output_loss[counter]
                        counter += 1
                    if return_S:
                        S = output_loss[counter]
                        counter += 1
                    if return_mu:
                        mu = output_loss[counter]
                    if return_Sigma:
                        Sigma = output_loss[-1]
                    if exp_ev is not None:
                        loss_ev, est_ev = obs_ev_loss(x_ev, mu, exp_ev, lam_ev) 
                        loss_value += loss_ev
                        self._train_ev.append(np.concatenate(([self._step], (est_ev).detach().to('cpu').numpy())))
                        if tb_writer is not None:
                            for i in range(est_ev.shape[0]):
                                tb_writer.add_scalars('EV', {'train_'+str(i+1): est_ev[i].item()}, self._step)
                    if exp_ac is not None:
                        loss_ac, est_ac = obs_ac_loss(x_ac, mu, x_t, K, Sigma, exp_ac, lam_ac) 
                        loss_value += loss_ac
                        self._train_ac.append(np.concatenate(([self._step], (est_ac).detach().to('cpu').numpy())))
                        if tb_writer is not None:
                            for i in range(est_ac.shape[0]):
                                tb_writer.add_scalars('AC', {'train_'+str(i+1): est_ac[i].item()}, self._step)
                    if exp_its is not None:
                        loss_its, est_its = obs_its_loss(S, Sigma, its_state1, its_state2, exp_its, lam_its, epsilon=self.epsilon, mode=self.score_mode)
                        loss_value += loss_its
                        self._train_its.append(np.concatenate(([self._step], (est_its).detach().to('cpu').numpy())))
                        if tb_writer is not None:
                            for i in range(est_its.shape[0]):
                                tb_writer.add_scalars('ITS', {'train_'+str(i+1): est_its[i].item()}, self._step)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(chain(self.ulayer.parameters(), self.slayer.parameters()), CLIP_VALUE)
                    self.optimizer_u.step()
                    if train_mode=='us':
                        self.optimizer_s.step()
                    
                    if train_score_callback is not None:
                        lval_detached = loss_value.detach()
                        train_score_callback(self._step, lval_detached)
                    self._train_scores.append((self._step, (loss_value).item()))
                    self._train_vampe.append((self._step, (vampe_loss).item()))
                    if tb_writer is not None:
                        tb_writer.add_scalars('Loss', {'train': loss_value.item()}, self._step)
                        tb_writer.add_scalars('VAMPE', {'train': vampe_loss.item()}, self._step)
                    if validation_loader is not None:
                        with torch.no_grad():
                            output_loss = vampe_loss_rev(x_val_0, x_val_t, self.ulayer, self.slayer, 
                                     return_mu=return_mu, return_Sigma=return_Sigma, return_K=return_K, return_S=return_S)
                            vampe_loss = output_loss[0]
                            score_val = -vampe_loss # vampe loss
                            counter=1
                            if return_K:
                                K = output_loss[counter]
                                counter += 1
                            if return_S:
                                S = output_loss[counter]
                                counter += 1
                            if return_mu:
                                mu = output_loss[counter]
                            if return_Sigma:
                                Sigma = output_loss[-1]
                            if exp_ev is not None:
                                loss_ev, est_ev = obs_ev_loss(x_val_ev, mu, exp_ev, lam_ev)
                                score_val += loss_ev
                                self._validation_ev.append(np.concatenate(([self._step], (est_ev).detach().to('cpu').numpy())))
                                if tb_writer is not None:
                                    for i in range(est_ev.shape[0]):
                                        tb_writer.add_scalars('EV', {'valid_'+str(i+1): est_ev[i].item()}, self._step)
                            if exp_ac is not None:
                                loss_ac, est_ac = obs_ac_loss(x_val_ac, mu, x_val_t, K, Sigma, exp_ac, lam_ac)
                                score_val += loss_ac
                                self._validation_ac.append(np.concatenate(([self._step], (est_ac).detach().to('cpu').numpy())))
                                if tb_writer is not None:
                                    for i in range(est_ac.shape[0]):
                                        tb_writer.add_scalars('AC', {'valid_'+str(i+1): est_ac[i].item()}, self._step)
                            if exp_its is not None:
                                loss_its, est_its = obs_its_loss(S, Sigma, its_state1, its_state2, exp_its, lam_its, epsilon=self.epsilon, mode=self.score_mode)
                                score_val += loss_its
                                self._validation_its.append(np.concatenate(([self._step], (est_its).detach().to('cpu').numpy())))
                                if tb_writer is not None:
                                    for i in range(est_its.shape[0]):
                                        tb_writer.add_scalars('ITS', {'valid_'+str(i+1): est_its[i].item()}, self._step)
                            self._validation_scores.append((self._step, score_val.item()))
                            self._validation_vampe.append((self._step, vampe_loss.item()))
                            if tb_writer is not None:
                                tb_writer.add_scalars('Loss', {'valid': score_val.item()}, self._step)
                                tb_writer.add_scalars('VAMPE', {'valid': vampe_loss.item()}, self._step)
                            if validation_score_callback is not None:
                                validation_score_callback(self._step, score_val)
                    self._step += 1
            if train_mode=='s':
                with torch.no_grad():
                    output_u = self.ulayer(x_0, x_t, return_mu=return_mu)
                    output_val_u = self.ulayer(x_val_0, x_val_t, return_mu=return_mu)
                    if return_mu:
                        mu = output_u[-1]
                        mu_val = output_val_u[-1]
                    if return_Sigma:
                        Sigma = output_u[4]
                        Sigma_val = output_u[4]
                for epoch in range(n_epochs):
                    self.optimizer_s.zero_grad()
                    
                    output_loss = vampe_loss_rev_only_S(*output_u[:5] , self.slayer, return_K=return_K, return_S=return_S)
                    vampe_loss = output_loss[0]
                    loss_value = - vampe_loss
                    if return_K:
                        K = output_loss[1]
                    if return_S:
                        S = output_loss[-1]
                    if exp_ac is not None:
                        loss_ac, est_ac = obs_ac_loss(x_ac, mu, x_t, K, Sigma, exp_ac, lam_ac)
                        loss_value += loss_ac
                        self._train_ac.append(np.concatenate(([self._step], (est_ac).detach().to('cpu').numpy())))
                        if tb_writer is not None:
                            for i in range(est_ac.shape[0]):
                                tb_writer.add_scalars('AC', {'train_'+str(i+1): est_ac[i].item()}, self._step)
                    if exp_its is not None:
                        loss_its, est_its = obs_its_loss(S, Sigma, its_state1, its_state2, exp_its, lam_its, epsilon=self.epsilon, mode=self.score_mode)
                        loss_value += loss_its
                        self._train_its.append(np.concatenate(([self._step], (est_its).detach().to('cpu').numpy())))
                        if tb_writer is not None:
                            for i in range(est_its.shape[0]):
                                tb_writer.add_scalars('ITS', {'train_'+str(i+1): est_its[i].item()}, self._step)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.slayer.parameters(), CLIP_VALUE)
                    self.optimizer_s.step()

                    if train_score_callback is not None:
                        lval_detached = loss_value.detach()
                        train_score_callback(self._step, lval_detached)
                    self._train_scores.append((self._step, (loss_value).item()))
                    self._train_vampe.append((self._step, (vampe_loss).item()))
                    if tb_writer is not None:
                        tb_writer.add_scalars('Loss', {'train': loss_value.item()}, self._step)
                        tb_writer.add_scalars('VAMPE', {'train': vampe_loss.item()}, self._step)
                    if validation_loader is not None:
                        with torch.no_grad():
                            output_loss = vampe_loss_rev_only_S(*output_val_u[:5], self.slayer, return_K=return_K, return_S=return_S)
                            vampe_loss = output_loss[0]
                            score_val = -vampe_loss
                            if return_K:
                                K = output_loss[1]
                            if return_S:
                                S = output_loss[-1]
                            if exp_ac is not None:
                                loss_ac, est_ac = obs_ac_loss(x_val_ac, mu_val, x_val_t, K, Sigma_val, exp_ac, lam_ac)
                                score_val += loss_ac
                                self._validation_ac.append(np.concatenate(([self._step], (est_ac).detach().to('cpu').numpy())))
                                if tb_writer is not None:
                                    for i in range(est_ac.shape[0]):
                                        tb_writer.add_scalars('AC', {'valid_'+str(i+1): est_ac[i].item()}, self._step)
                            if exp_its is not None:
                                loss_its, est_its = obs_its_loss(S, Sigma_val, its_state1, its_state2, exp_its, lam_its, epsilon=self.epsilon, mode=self.score_mode)
                                score_val += loss_its
                                self._validation_its.append(np.concatenate(([self._step], (est_its).detach().to('cpu').numpy())))
                                if tb_writer is not None:
                                    for i in range(est_its.shape[0]):
                                        tb_writer.add_scalars('ITS', {'valid_'+str(i+1): est_its[i].item()}, self._step)
                            self._validation_scores.append((self._step, score_val.item()))
                            self._validation_vampe.append((self._step, vampe_loss.item()))
                            if tb_writer is not None:
                                tb_writer.add_scalars('Loss', {'valid': score_val.item()}, self._step)
                                tb_writer.add_scalars('VAMPE', {'valid': vampe_loss.item()}, self._step)
                            if validation_score_callback is not None:
                                validation_score_callback(self._step, score_val)
                    self._step += 1                        
            
        return self
    
    def transform(self, data, instantaneous: bool = True, **kwargs):
        r""" Transforms data through the instantaneous or time-shifted network lobe.

        Parameters
        ----------
        data : numpy array or torch tensor
            The data to transform.
        instantaneous : bool, default=True
            Whether to use the instantaneous lobe or the time-shifted lobe for transformation.
        **kwargs
            Ignored kwargs for api compatibility.

        Returns
        -------
        transform : array_like
            List of numpy array or numpy array containing transformed data.
        """
        model = self.fetch_model()
        return model.transform(data, **kwargs)

    def fetch_model(self) -> DeepMSMModel:
        r""" Yields the current model. """
        return DeepMSMModel(self.lobe, self.ulayer, self.slayer, self.cg_list, self.mask, dtype=self.dtype, device=self.device)
        
        
    def set_rev_var(self, data_loader: torch.utils.data.DataLoader, S=False):
        
        with torch.no_grad():
            chi_t, chi_tau = [], []
            for batch_0, batch_t in data_loader:
                chi_t.append(self.forward(batch_0).detach())
                chi_tau.append(self.forward(batch_t).detach())
            
        chi_t = torch.cat(chi_t, dim=0)
        chi_tau = torch.cat(chi_tau, dim=0)
        
        cov_00, cov_0t, cov_tt = covariances(chi_t, chi_tau, remove_mean=False)
        cov_00_inv = sym_inverse(cov_00, epsilon=self.epsilon, mode=self.score_mode).to('cpu').numpy()
        K_vamp = (cov_00_inv @ cov_0t.to('cpu').numpy())
        # estimate pi, the stationary distribution vector
        eigv, eigvec = np.linalg.eig(K_vamp.T)
        ind_pi = np.argmin((eigv-1)**2)

        pi_vec = np.real(eigvec[:,ind_pi])
        pi = pi_vec / np.sum(pi_vec, keepdims=True)
        print('pi', pi)
        # reverse the consruction of u 
        u_optimal = cov_00_inv @ pi
        print('u optimal', u_optimal)
        
        # u_kernel = np.log(np.exp(np.abs(u_optimal))-1) # if softplus
        # for relu
        u_kernel = np.abs(u_optimal)
        
        with torch.no_grad():
            for param in self.ulayer.parameters():
            
                param.copy_(torch.Tensor(u_kernel[None,:]))  
            
        if S:
            with torch.no_grad():
                _, _, _, _, Sigma = self.ulayer(chi_t, chi_tau)
                Sigma = Sigma

                sigma_inv = sym_inverse(Sigma, epsilon=self.epsilon, mode=self.score_mode).detach().to('cpu').numpy()
            # reverse the construction of S
            S_nonrev = K_vamp @ sigma_inv
            S_rev_add = 1/2 * (S_nonrev + S_nonrev.T)
            
            kernel_S = S_rev_add / 2.
            # for softplus
            # kernel_S = np.log(np.exp(np.abs(kernel_S))-1)
            # for relu
            kernel_S = np.abs(kernel_S)
            
            with torch.no_grad():
                for param in self.slayer.parameters():

                    param.copy_(torch.Tensor(kernel_S))
    
    def reset_u_S(self, data_loader: torch.utils.data.DataLoader, reset_opt=False):
        
        with torch.no_grad():
            chi_t, chi_tau = [], []
            for batch_0, batch_t in data_loader:
                chi_t.append(self.forward(batch_0).detach())
                chi_tau.append(self.forward(batch_t).detach())
            
        chi_t = torch.cat(chi_t, dim=0)
        chi_tau = torch.cat(chi_tau, dim=0)
        
        u_kernel = np.ones(self.output_dim)
        K_vamp = np.ones((self.output_dim, self.output_dim)) + np.diag(np.ones(self.output_dim))
        K_vamp = K_vamp / np.sum(K_vamp, axis=1, keepdims=True)
        with torch.no_grad():
            for param in self.ulayer.parameters():
            
                param.copy_(torch.Tensor(u_kernel[None,:]))  
            
        with torch.no_grad():
            _, _, _, _, Sigma = self.ulayer(chi_t, chi_tau)
            Sigma = Sigma

            sigma_inv = sym_inverse(Sigma, epsilon=self.epsilon, mode=self.score_mode).detach().to('cpu').numpy()
        # reverse the construction of S
        S_nonrev = K_vamp @ sigma_inv
        S_rev_add = 1/2 * (S_nonrev + S_nonrev.T)

        kernel_S = S_rev_add / 2.
        # for softplus
        # kernel_S = np.log(np.exp(np.abs(kernel_S))-1)
        # for relu
        kernel_S = np.abs(kernel_S)

        with torch.no_grad():
            for param in self.slayer.parameters():

                param.copy_(torch.Tensor(kernel_S))
        if reset_opt:
            self.optimizer_u = torch.optim.Adam(self.ulayer.parameters(), lr=self.learning_rate*10)
            self.optimizer_s = torch.optim.Adam(self.slayer.parameters(), lr=self.learning_rate*100)
    
#     def reset_u_S(self):
        
        
#         u_kernel = np.ones(self.output_dim)
        
#         with torch.no_grad():
#             for param in self.ulayer.parameters():
            
#                 param.copy_(torch.Tensor(u_kernel[None,:]))  
#         S_kernel = np.ones((self.output_dim, self.output_dim))
#         with torch.no_grad():
#             for param in self.slayer.parameters():

#                 param.copy_(torch.Tensor(S_kernel))
    
    def reset_opt_u_S(self, lr=1):
        self.optimizer_u = torch.optim.Adam(self.ulayer.parameters(), lr=self.learning_rate*lr)
        self.optimizer_s = torch.optim.Adam(self.slayer.parameters(), lr=self.learning_rate*10*lr)
        
    def initialize_cg_layer(self, idx: int, data_loader: torch.utils.data.DataLoader, factor: float = 1.0):

        '''Initilize the coarse_layer[idx] with the pcca_memberships'''

        assert self.coarse_grain is not None , f"The estimator has no coarse-graining layers"

        assert idx<len(self.coarse_grain), f"The chosen idx of the coarse graining layer {idx} does not exist"

        # First estimate the values of u and S before the coarse layer idx
        with torch.no_grad():
            chi_t, chi_tau = [], []
            for batch_0, batch_t in data_loader:
                chi_t.append(self.forward(batch_0).detach())
                chi_tau.append(self.forward(batch_t).detach())
            chi_t = torch.cat(chi_t, dim=0)
            chi_tau = torch.cat(chi_tau, dim=0)

            v, C00, Ctt, C0t, Sigma, u_n = self.ulayer(chi_t, chi_tau, return_u=True)
            _, K_n, S_n = self.slayer(v, C00, Ctt, C0t, Sigma, return_S=True, return_K=True)

            for cg_id in range(idx):
                _, chi_t, chi_tau, u_n, S_n, K_n = self.cg_list[cg_id].get_cg_uS(chi_t, chi_tau, u_n, S_n, return_chi=True, return_K=True)

            T = K_n.to('cpu').numpy().astype('float64')
            # renormalize because of the type casting
            T = T / T.sum(axis=1)[:, None]
            try:
                # use the estimated transition matrix to get the pcca membership
                mem = pcca_memberships(T, self.coarse_grain[idx])
                mem = np.log(mem)
            except ValueError:
                print('PCCA was not successful try different initialization strategy')
                eigvals, eigvecs = np.linalg.eig(T)
                sort_id = np.argsort(eigvals)
                eigvals = eigvals[sort_id]
                eigvecs = eigvecs[:,sort_id]
                size = self.cg_list[idx].M
                mem = eigvecs[:,-size:]
                for j in range(size):
                    ind = np.argwhere(mem[:,j]<0)
                    mem[ind,j] = mem[ind,j] / np.abs(np.min(mem[:,j]))
                    ind = np.argwhere(mem[:,j]>0)
                    mem[ind,j] = mem[ind,j] / np.abs(np.max(mem[:,j]))
                mem[:,-1] = mem[:,-1]/size
            # since they will be past through a softmax take log
            initial_values = mem * factor
            # set the parameters of cg_layer[idx] to the estimated values
            self.cg_list[idx].weight.copy_(torch.Tensor(initial_values))
#             assert np.allclose(mem,self.cg_list[idx].get_softmax().to('cpu').numpy().astype('float64'),rtol=1e-6), 'The estimated values does not match the assigned one'

        return
    def reset_cg(self, idx=0, lr=0.1):
        with torch.no_grad():
            self.cg_list[idx].weight.copy_(torch.ones((self.cg_list[idx].N, self.cg_list[idx].M)))
        self.cg_opt_list[idx] = torch.optim.Adam(self.cg_list[idx].parameters(), lr=lr)
        
        return
    def state_dict(self):
        
        ret = [self.lobe.state_dict(), self.ulayer.state_dict(), self.slayer.state_dict()]
        
        if self.coarse_grain is not None:
            for cglayer in self.cg_list:
                ret.append(cglayer.state_dict())
        if len(self.mask.state_dict())>0:
            ret.append(self.mask.state_dict())
        return ret
    
    
    def load_state_dict(self, dict_lobe, dict_u, dict_S, cg_dicts=None, mask_dict=None):
        
        self.lobe.load_state_dict(dict_lobe)
        self.ulayer.load_state_dict(dict_u)
        self.slayer.load_state_dict(dict_S)
        
        if cg_dicts is not None:
            assert len(self.cg_list)==len(cg_dicts), 'The number of coarse grain layer dictionaries does not match the number of coarse grain layers'
            for i, cglayer in enumerate(self.cg_list):
                cglayer.load_state_dict(cg_dicts[i])
        if mask_dict is not None:
            assert isinstance(self.mask, nn.Module), 'The mask layer is not a nn.Module'
            self.mask.load_state_dict(mask_dict)
        return
        
    def save_params(self, paths: str):
        
        list_dict = self.state_dict()
        
        np.savez(paths, 
        dict_lobe=list_dict[0],
        dict_u=list_dict[1],
        dict_s=list_dict[2],
                *list_dict[3:])
        
        return print('Saved parameters at: '+paths)
    
    def load_params(self, paths: str):
        
        dicts = np.load(paths, allow_pickle=True)
        
        
        if len(dicts.keys())>3: # it is expected to be coarse-grain layers
            cg_dicts = []
            for i in range(len(self.cg_list)):
                cg_dicts.append(dicts['arr_'+str(i)].item())
            if len(dicts.keys())>(3+i+1):
                mask_dict = dicts['arr_'+str(i+1)].item()
            else:
                mask_dict = None
        else:
            cg_dicts=None
            mask_dict=None
        self.load_state_dict(dicts['dict_lobe'].item(), dicts['dict_u'].item(), dicts['dict_s'].item(), cg_dicts=cg_dicts, mask_dict=mask_dict)
        
        return