from src.utils import *
from src.Queue import Queue

class Arm:
    def __init__(self, node_list, d, M_max, state_mode, sample_mode, T_period):
        self.arm = node_list
        self.size = len(node_list)
        self.d = d
        self.index = {u:i for i,u in enumerate(self.arm)}
        self.M_max = M_max
        self.T_period = T_period
        
        self.fn_Psi = fn_Psi[state_mode]
        self.find_closest = find_closest[state_mode]
        
        self.sample_mode = sample_mode
        
        # need to create a dictionary of queues for each state
        self.queue_dict = {}
        
        self.transition_potential = {}
        
        self.mu = np.zeros((self.size*self.d,1))
        self.sigma = np.zeros((self.size*self.d, self.size*self.d))
        
    def preprocess(self, x):
        y = np.zeros((self.size*self.d,))
        
        for u in self.arm:
            y[self.index[u]*self.d: (self.index[u]+1)*self.d] = x[u*self.d: (u+1)*self.d]
            
        return y
    
    def extract(self, x, u):
        if x.shape[0] != self.size*self.d:
            return x[u*self.d: (u+1)*self.d]
        return x[self.index[u]*self.d: (self.index[u]+1)*self.d]
        
    
    def get_markov_matrix(self):
        '''The function `get_markov_matrix` calculates the transition matrix for a Markov chain given a
        dictionary of transition potentials.
        
        Returns
        -------
            a tuple containing two elements. The first element is a list of all states, and the second
        element is the transpose of the Markov transition matrix.
        
        '''
        level_0_states = list(self.transition_potential.keys())
        
        level_1_states = []
        for level_0_state in level_0_states:
            level_1_states += list(self.transition_potential[level_0_state].keys())
        
        all_states = list(set(level_0_states + level_1_states))
        n_states = len(all_states)
        
        T_mtx = np.zeros((n_states, n_states))
        
        # state (i) -> state_ (j)
        for i, state in enumerate(all_states):
            for j, state_ in enumerate(all_states):
                if state in self.transition_potential.keys():
                    if state_ in self.transition_potential[state]:
                        T_mtx[i][j] = self.transition_potential[state][state_]

        denom = np.sum(T_mtx, axis=1)
        denom[denom==0]=1
        W = T_mtx.T/denom
        
        return (all_states, W.T)
        
    def update(self, shock_now, shock_next, time_step=None, joint=False):
        # update queue
        state_now = self.fn_Psi(self.preprocess(shock_now), time_step, self.T_period)
        state_next = self.fn_Psi(self.preprocess(shock_next), time_step, self.T_period)
        
        if joint: # (s,s')
            queue_state = state_now + state_next
        else: # (s)
            queue_state = state_now
            
        if queue_state not in self.queue_dict:
            self.queue_dict[queue_state] = Queue(self.M_max, queue_state, self.arm, self.size, self.d, self.sample_mode)
        
        
        # # ignoring missing value
        # if not np.any(shock_next == 0):    
        #     self.queue_dict[queue_state].enqueue(self.preprocess(shock_next))
        
        self.queue_dict[queue_state].enqueue(self.preprocess(shock_next))
        
        # update transition potential matrix
        if state_now not in self.transition_potential:
            self.transition_potential[state_now] = {}
        
        if state_next not in self.transition_potential[state_now]:
            self.transition_potential[state_now][state_next] = 1

        self.transition_potential[state_now][state_next] +=1
    
    # single-step
    def sample(self, shock_now, time_step=None, joint=False):
        
        if shock_now.shape[0] > self.size*self.d:
            state_now = self.fn_Psi(self.preprocess(shock_now), time_step, self.T_period)
        else:
            state_now = self.fn_Psi(shock_now, time_step, self.T_period)
        # what if the state_now is never seen?
        if state_now not in self.transition_potential.keys():
            # return np.zeros(len((state_now),)) # this might be bad for multi-step forecasting
            # can instead find the state closest to state_now, and sample from that instead
            state_now = self.find_closest(list(self.transition_potential.keys()), state_now, self.T_period)
            # ISSUE: this seems to slow down the algorithm but is necessary for multi-step forecasts
            # FIX: keep the adjacency sparse
            
        if joint: # (s,s')
            # first sample the next state s' from the current state s
            state_vec = list(self.transition_potential[state_now].keys())
            p_vec = np.array(list(self.transition_potential[state_now].values()))
            p_vec = p_vec/np.sum(p_vec)
            state_next = state_vec[np.random.choice(len(p_vec), 1, p=p_vec)[0]]
            
            queue_state = state_now+state_next
            
        else: # (s)
            queue_state = state_now
        
        # sample the shock from the dist of queue (s,s') or (s)
        self.queue_dict[queue_state].update_distribution()
        shock_next = self.queue_dict[queue_state].distribution.sample()
        
        if joint:
            # sample until self.fn_Psi(shock_next) is same as state_next (~2%)
            while self.fn_Psi(shock_next, time_step, self.T_period) != state_next:
                shock_next = self.queue_dict[queue_state].distribution.sample()
            
        return shock_next
    
    def sample_multistep(self, shock_now, q, time_step=None, joint=False):
        shock_mtx =  np.zeros((self.size*self.d,q))
        for m in range(q):
            if time_step is None:
                shock_next = self.sample(shock_now)
            else:
                shock_next = self.sample(shock_now, time_step+m)
            shock_mtx.T[m] = shock_next
            shock_now = shock_next
            
        return shock_mtx