import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Mean_std_layer(nn.Module):
    """ Custom Linear layer for substracting the mean and dividing by the std"""
    def __init__(self, size_in, mean=None, std=None):
        super().__init__()
        self.size_in = size_in
        if mean is None:
            mean = torch.zeros((1,size_in))
        self.weights_mean = nn.Parameter(mean, requires_grad=False)  # nn.Parameter is a Tensor that's a module parameter.
        if std is None:
            std = torch.ones((1,size_in))
        self.weights_std = nn.Parameter(std, requires_grad=False)

    def forward(self, x):
        y = (x-self.weights_mean)/self.weights_std
        return y  
    
    def set_both(self, mean, std):
        new_params = [mean, std]
        with torch.no_grad():
            for i, param in enumerate(self.parameters()):
                new_param = new_params[i]
                param.copy_(torch.Tensor(new_param[None,:]))
                
                
                

class Mask(torch.nn.Module):
    ''' Attention mask either independent from the time point (mask_const=True) or dependent.
    If dependent the attention is estimated via a NN with depth and width given as input, which are 
    otherwise ignored. 
    The attention mechanism assumes that distances are used. skip_res is number of residues skiped when estimating
    the distance. 
    
    ---------
    Inputs
    ---------
    input_size: int
            the dimension of the input array, which are expected to be distances
    mask_const: bool
            if True a constant mask is used independent from the actual time frame, which results
            in trainable weights of vector with the dimension of the number of windows
            if False a time dependent attention mechanism is used with a NN with depth and width
    depth: int
            the number of hidden layers for the attention mechanism if mask_const=True.
    width: int
            the width of hidden layers for the attention mechanism if mask_const=True.
    patchsize: int
            The size of the window for the attention. The larger the number the smoother the attention
            along the residue chain. a patchsize=1 corresponds to no smoothing.
    fac: bool
            if True the window weight is scaled with the number of windows. This circumvents if large windows
            are used that the product of all weights does not scale to fast to zero. The weights
            are then distributed around 1 instead of 1/number_weights.
    noise: float
            The amount of noise used to scale the input with, which depends on the attention value.
            This should prevent the classification network afterwards to use information originating 
            from residues with a low attention weight.            
            
    
    
    '''
    def __init__(self, input_size, mask_const, depth=0, width=100 , patchsize=4, fac=True,
                noise=0., device='cpu'):
        super(Mask, self).__init__()
        
#         self.alpha = torch.Tensor(1, input_size, N, heads).fill_(0)

        skip_res = 3
        self.noise = noise
        self.n_residues = int(-1/2 + np.sqrt(1/4+input_size*2) + skip_res)
        self.device = device
        self.bs_per_res = [[] for _ in range(patchsize)]
        
        self.residues_1 = []
        self.residues_2 = []
        self.patchsize = patchsize
        self.number_weights = self.n_residues + (patchsize-1)
        # estimate the pairs
        for n1 in range(self.n_residues):
            for i in range(patchsize):
                self.bs_per_res[i].append(n1+i)
            
        for n1 in range(self.n_residues-skip_res):
            for n2 in range(n1+skip_res, self.n_residues):
                self.residues_1.append(n1)
                self.residues_2.append(n2)
                        
                        
        self.mask_const = mask_const
        
        if mask_const:        
            self.alpha = torch.randn((1, self.number_weights)) * 0.5
            self.weight = torch.nn.Parameter(data=self.alpha, requires_grad=True)
        else:
            nodes = [input_size]
            for i in range(depth):
                nodes.append(width)
            
            
            self.hfc = [nn.Linear(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
            self.softmax = nn.Linear(nodes[-1], self.number_weights, bias=True)
            self.layers = nn.ModuleList(self.hfc)
            
        if fac:
            self.fac = self.number_weights
        else:
            self.fac = 1.
    def forward(self, x):
        
        # weights for each residue
        weights_for_res = self.get_softmax(x) 
        
            
        # get the weights for each distance
        weight_1 = weights_for_res[:,self.residues_1]
        weight_2 = weights_for_res[:,self.residues_2]

        alpha = weight_1 * weight_2 * self.n_residues**2
        masked_x = x * alpha
        
        if self.noise > 0.: # add noise if enabled to regularize
            max_attention_value = torch.max(alpha, dim=1, keepdim=True)[0].detach()
            shape = alpha.shape
            random_numbers = torch.randn(shape, device=self.device) * self.noise
            masked_x += (1 - alpha/max_attention_value) * random_numbers
        
        return masked_x
    
    def get_softmax(self, x=None):
        if self.mask_const:
            weights_for_res = []
            for i in range(self.patchsize): # get all weights b for each residue
                weights_for_res.append(self.weight[None,:,self.bs_per_res[i]])
                
            weights_for_res = torch.prod(torch.cat(weights_for_res, dim=0), dim=0) # take the product of the b factors
            weights_for_res = F.softmax(weights_for_res, dim=1) # take the softmax over the residues
        else:
            y = x
            for layer in self.hfc:
                y = layer(y)
            y = self.softmax(y)
            y = F.softmax(y, dim=1)*self.fac
#             y = F.elu(y)+1
            weights_for_res = []
            for i in range(self.patchsize): # get all weights b for each residue
                weights_for_res.append(y[None,:,self.bs_per_res[i]])
                
            weights_for_res = torch.prod(torch.cat(weights_for_res, dim=0), dim=0) # take the product of the b factors
#             weights_for_res = F.relu(weights_for_res-0.9)
#             weights_for_res = weights_for_res/torch.sum(weights_for_res, dim=1, keepdims=True)
            weights_for_res = F.softmax(weights_for_res, dim=1) # take the softmax over the residues
            
        
        return weights_for_res
# transform a trajectory which might not fit into memory at once, predict batchwise
def pred_batchwise(lobe, traj, batchsize=10000, device='cpu'):
    
    data_size = traj.shape[0]
    batches = data_size//batchsize
    pred_all = []
    for i in range(batches):
        s = batchsize*i
        e = s+batchsize
        pred_temp = lobe.forward(torch.Tensor(traj[s:e]).to(device)).detach().to('cpu').numpy()
        pred_all.append(pred_temp)
    if batches==0:
        pred_all.append(lobe.forward(torch.Tensor(traj).to(device)).detach().to('cpu').numpy())
    else:
        pred_all.append(lobe.forward(torch.Tensor(traj[e:]).to(device)).detach().to('cpu').numpy())
    
    return np.concatenate(pred_all, axis=0)

# plotting the mask

def plot_mask(data=None, lobe=None, mask=None, mask_const=True, device='cpu', return_values=False, skip=5, vmax=1, top=10):
    if mask_const:
        attention = mask.get_softmax()
        attention_np = attention.detach().to('cpu').numpy().T
        n_residues = attention_np.shape[0]
        plt.imshow(attention_np, vmin=0, vmax=vmax, aspect='auto')
        plt.xlabel('System', fontsize=18)
        plt.ylabel('Input', fontsize=18)
        plt.xticks(np.arange(1),['{}'.format(i) for i in range(1)], fontsize=16)
        plt.yticks(np.arange(0,n_residues,skip),['x{}'.format(i) for i in range(0,n_residues,skip)], fontsize=16)
        plt.show()
    #     plt.savefig('./Figs/2x3_mix_Mask.pdf', bbox_inches='tight')
        if return_values:
            return attention_np
        
    else:
        pred_temp = pred_batchwise(lobe, data, batchsize=10000, device=device)
        arg_sort = np.argsort(pred_temp, axis=0)
        top_x_state = arg_sort[-top:]
        states = pred_temp.shape[1]
        att_atom = []
        for state in range(states):
            frames = top_x_state[:,state]
            attention = mask.get_softmax(torch.Tensor(data[frames]).to(device))
            attention_np = attention.detach().to('cpu').numpy()
            att_atom.append(np.mean(attention_np, axis=0, keepdims=True))
        att_atom = np.concatenate(att_atom)
        n_residues = att_atom.shape[1]
        plt.imshow(att_atom.T, vmin=0, vmax=vmax, aspect='auto')
        plt.xlabel('State', fontsize=18)
        plt.ylabel('Input', fontsize=18)
        plt.xticks(np.arange(states),['{}'.format(i) for i in range(states)], fontsize=16)
        plt.yticks(np.arange(0,n_residues,skip),['x{}'.format(i) for i in range(0,n_residues,skip)], fontsize=16)
        plt.show()
    #     plt.savefig('./Figs/2x3_mix_Mask.pdf', bbox_inches='tight')
        if return_values:
            return att_atom
        
def get_its(data, lags, calculate_K = True, multiple_runs = False):
    
    def get_single_its(data):

        if type(data) == list:
            outputsize = data[0].shape[1]
        else:
            outputsize = data.shape[1]

        single_its = np.zeros((outputsize-1, len(lags)))

        for t, tau_lag in enumerate(lags):
            if calculate_K:
                koopman_op = estimate_koopman_op(data, tau_lag)
            else:
                koopman_op = data[t]
            k_eigvals, k_eigvec = np.linalg.eig(np.real(koopman_op))
            k_eigvals = np.sort(np.absolute(k_eigvals))
            k_eigvals = k_eigvals[:-1]
            single_its[:,t] = (-tau_lag / np.log(k_eigvals))

        return np.array(single_its)


    if not multiple_runs:

        its = get_single_its(data)

    else:

        its = []
        for data_run in data:
            its.append(get_single_its(data_run))

    return its

def get_ck(K, lag):
    n_states = K[0].shape[0]
    steps = len(lag)
    predicted = np.zeros((n_states, n_states, steps))
    estimated = np.zeros((n_states, n_states, steps))

    predicted[:,:,0] =  np.identity(n_states)
    estimated[:,:,0] =  np.identity(n_states)

    for vector, i  in zip(np.identity(n_states), range(n_states)):
        for n in range(1, steps):

            koop = K[0]
            fac = lag[n]//lag[0]
            koop_pred = np.linalg.matrix_power(koop,fac)

            koop_est = K[n]

            predicted[i,:,n]= vector @ koop_pred
            estimated[i,:,n]= vector @ koop_est
        
              
    return [predicted, estimated]

def plot_cg(layer):
    attention = layer.get_softmax()
    attention_np = attention.detach().to('cpu').numpy()
    plt.imshow(attention_np)
    plt.xlabel('From State', fontsize=18)
    plt.ylabel('To State', fontsize=18)
    plt.show()
    
    
def estimate_mu(mu, chi_true, frames):
    ''' Estimates the state probability of a reference model. The stationary distribution
    mu is estimated from the current model, but the state assignment stems from the
    reference model. This makes it comparable over several models.
    '''
    state_prob = np.sum(mu * chi_true[frames], axis=0)
#     plt.plot(state_prob, '.')
#     plt.show()
    return state_prob





class TimeSeriesDataset(object):
    r""" High-level container for time-series data.
    This can be used together with pytorch data tools, i.e., data loaders and other utilities.

    Parameters
    ----------
    data : (T, ...) ndarray
        The dataset with T frames.
    """

    def __init__(self, data):
        self.data = data

    def lag(self, lagtime: int):
        r""" Creates a time lagged dataset out of this one.

        Parameters
        ----------
        lagtime : int
            The lagtime, must be positive.

        Returns
        -------
        dataset : TimeLaggedDataset
            Time lagged dataset.
        """
        return TimeLaggedDataset.from_trajectory(lagtime, self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class TimeLaggedDataset(TimeSeriesDataset):
    r""" High-level container for time-lagged time-series data.
    This can be used together with pytorch data tools, i.e., data loaders and other utilities.

    Parameters
    ----------
    data : iterable of data
        The data which is wrapped into a dataset
    data_lagged : iterable of data
        Corresponding timelagged data. Must be of same length.
    dtype : numpy data type
        The data type to map to when retrieving outputs
    """

    def __init__(self, data, data_lagged, dtype=np.float32):
        super().__init__(data)
        assert len(data) == len(data_lagged), 'data and data lagged must be of same size'
        self.data_lagged = data_lagged
        self.dtype = dtype

    @staticmethod
    def from_trajectory(lagtime: int, data: np.ndarray):
        r""" Creates a time series dataset from a single trajectory by applying a lagtime.

        Parameters
        ----------
        lagtime : int
            Lagtime, must be positive. The effective size of the dataset reduces by the selected lagtime.
        data : (T, d) ndarray
            Trajectory with T frames in d dimensions.

        Returns
        -------
        dataset : TimeSeriesDataset
            The resulting time series dataset.
        """
        assert lagtime > 0, "Lagtime must be positive"
        return TimeLaggedDataset(data[:-lagtime], data[lagtime:], dtype=data.dtype)

    def __getitem__(self, item):
        return self.data[item].astype(self.dtype), self.data_lagged[item].astype(self.dtype)

    def __len__(self):
        return len(self.data)

    
class TimeLaggedDatasetObs(TimeSeriesDataset):
    r""" High-level container for time-lagged time-series data.
    This can be used together with pytorch data tools, i.e., data loaders and other utilities.

    Parameters
    ----------
    data : iterable of data
        The data which is wrapped into a dataset
    data_lagged : iterable of data
        Corresponding timelagged data. Must be of same length.
    data_obs_ev: iterable of data
        Corresponding microscopic observable of type expectation value
    data_obs_ac: iterable of data
        Corresponding microscopic observable of type auto correlation
    dtype : numpy data type
        The data type to map to when retrieving outputs
    """

    def __init__(self, data, data_lagged, data_obs_ev=None, data_obs_ac=None, dtype=np.float32):
        super().__init__(data)
        assert len(data) == len(data_lagged), 'data and data lagged must be of same size'
        self.data_lagged = data_lagged
        self.data_obs_ev = data_obs_ev
        self.data_obs_ac = data_obs_ac
        self.dtype = dtype

    @staticmethod
    def from_trajectory(lagtime: int, data: np.ndarray, data_obs_ev: np.ndarray=None, data_obs_ac: np.ndarray=None):
        r""" Creates a time series dataset from a single trajectory by applying a lagtime.

        Parameters
        ----------
        lagtime : int
            Lagtime, must be positive. The effective size of the dataset reduces by the selected lagtime.
        data : (T, d) ndarray
            Trajectory with T frames in d dimensions.
        data_obs_ev : (T, n) ndarray
            Trajectory of n microscopic observables with T frames. 
        data_obs_ac : (T, n) ndarray
            Trajectory of n microscopic observables with T frames. 

        Returns
        -------
        dataset : TimeLaggedDatasetObs
            The resulting time series dataset.
        """
        assert lagtime > 0, "Lagtime must be positive"
        if data_obs_ev is not None:
            data_obs_ev = data_obs_ev[lagtime:]
        if data_obs_ac is not None:
            data_obs_ac = data_obs_ac[lagtime:]
        return TimeLaggedDatasetObs(data[:-lagtime], data[lagtime:], data_obs_ev, data_obs_ac, dtype=data.dtype)
    
    @staticmethod
    def from_frames(lagtime: int, data: np.ndarray, frames: np.ndarray, 
                    data_obs_ev: np.ndarray=None, data_obs_ac: np.ndarray=None):
        r""" Creates a time series dataset from a single trajectory by applying a lagtime.

        Parameters
        ----------
        lagtime : int
            Lagtime, must be positive. The effective size of the dataset reduces by the selected lagtime.
        data : (T, d) ndarray
            Trajectory with T frames in d dimensions.
        data_obs_ev : (T, n) ndarray
            Trajectory of n microscopic observables with T frames. 
        data_obs_ac : (T, n) ndarray
            Trajectory of n microscopic observables with T frames. 

        Returns
        -------
        dataset : TimeLaggedDatasetObs
            The resulting time series dataset.
        """
        assert lagtime > 0, "Lagtime must be positive"
        if data_obs_ev is not None:
            data_obs_ev = data_obs_ev[frames+lagtime]
        if data_obs_ac is not None:
            data_obs_ac = data_obs_ac[frames+lagtime]
        return TimeLaggedDatasetObs(data[frames], data[frames+lagtime], data_obs_ev, data_obs_ac, dtype=data.dtype)
    
    def __getitem__(self, item):
        if self.data_obs_ev is not None and self.data_obs_ac is not None:
            return self.data[item].astype(self.dtype), self.data_lagged[item].astype(self.dtype), self.data_obs_ev[item].astype(self.dtype), self.data_obs_ac[item].astype(self.dtype)
        elif self.data_obs_ev is not None:
            return self.data[item].astype(self.dtype), self.data_lagged[item].astype(self.dtype), self.data_obs_ev[item].astype(self.dtype)
        elif self.data_obs_ac is not None:
            return self.data[item].astype(self.dtype), self.data_lagged[item].astype(self.dtype), self.data_obs_ac[item].astype(self.dtype)
        else:
            return self.data[item].astype(self.dtype), self.data_lagged[item].astype(self.dtype)

    def __len__(self):
        return len(self.data)