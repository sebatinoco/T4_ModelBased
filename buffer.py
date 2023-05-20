import numpy as np

class Buffer:

    def __init__(self, dim_states, dim_actions, max_size, sample_size, continuous_control, drop_last = False):

        assert sample_size <= max_size, "Sample size cannot be greater than buffer size"
        
        self._buffer_idx     = 0
        self._exps_stored    = 0
        self._buffer_size    = max_size
        self._sample_size    = sample_size
        
        self._dim_actions = dim_actions
        self._continuous_control = continuous_control

        self._s_t_array      = np.zeros((max_size, dim_states))
        self._a_t_array      = np.zeros((max_size, dim_actions))
        self._s_t1_array     = np.zeros((max_size, dim_states))
        
        self._drop_last = drop_last


    def store_transition(self, s_t, a_t, s_t1):
        
        # one hot action
        if not self._continuous_control:
            a_t = np.eye(self._dim_actions)[a_t]
        
        # Add transition to the buffer
        self._s_t_array[self._buffer_idx] = s_t
        self._a_t_array[self._buffer_idx] = a_t
        self._s_t1_array[self._buffer_idx] = s_t1
        
        # update buffer idx
        self._buffer_idx += 1
        if self._buffer_idx == self._buffer_size:
            self._buffer_idx = 0
            
        # update exps_stored idx
        self._exps_stored += 1
        
    
    def get_batches(self):
        assert self._exps_stored + 1 > self._sample_size, "Not enough samples has been stored to start sampling"
            
        upper_idx = self._exps_stored if self._sample_size <= self._exps_stored < self._buffer_size else self._buffer_size

        # number of batches
        n_batches = upper_idx // self._sample_size + 1 
        
        # idxs to split batches
        idxs = [self._sample_size * i for i in range(1, n_batches)]   
        
        # generate batches
        s_t_batches = np.split(self._s_t_array[:self._exps_stored], idxs)
        a_t_batches = np.split(self._a_t_array[:self._exps_stored], idxs)
        s_t1_batches = np.split(self._s_t1_array[:self._exps_stored], idxs)
        
        # drop last batch if wanted
        if self._drop_last:
            s_t_batches = s_t_batches[:-1]
            a_t_batches = a_t_batches[:-1]
            s_t1_batches = s_t1_batches[:-1]
        
        return list(zip(s_t_batches, a_t_batches, s_t1_batches))
            