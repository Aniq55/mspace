import numpy as np

class Gaussian:
    def __init__(self, mu=None, sigma=None, n=None):
        self.mu = mu 
        self.sigma = sigma
        self.n = n
        
    def update(self, Q):
        mu = np.mean(Q, axis=1)
        Q_ = Q - np.expand_dims(mu, 1)
        sigma = (1/Q.shape[1])*Q_@Q_.T
        
        self.mu = mu 
        self.sigma = sigma
        
        self.n = mu.shape[0]
    
    def sample(self, independent=False):
        if independent: # independent sampling is a bad idea
            return self.sample_independent()
        return np.random.multivariate_normal(self.mu, self.sigma)
    
    def sample_independent(self):
        return np.random.normal(self.mu, np.ones(self.n).T @ (self.sigma*np.eye(self.n)))
    

class Gaussian2:
    def __init__(self, mu=None, sigma=None, n=None):
        self.mu = mu 
        self.sigma = sigma
        self.n = n
        
    def update(self, Q, A_sigma):
        mu = np.mean(Q, axis=1)
        Q_ = Q - np.expand_dims(mu, 1)
        sigma = (1/Q.shape[1])*Q_@Q_.T
        
        self.mu = mu 
        self.sigma = sigma*A_sigma
        
        self.n = mu.shape[0]
    
    def sample(self, independent=False):
        if independent: # independent sampling is a bad idea
            return self.sample_independent()
        return np.random.multivariate_normal(self.mu, self.sigma)
    
    def sample_independent(self):
        return np.random.normal(self.mu, np.ones(self.n).T @ (self.sigma*np.eye(self.n)))
    
class Mean:
    def __init__(self, mu=None, n=None):
        self.mu = mu 
        self.n = n
        
    def update(self, Q):
        mu = np.mean(Q, axis=1)   
        self.mu = mu 
        self.n = mu.shape[0]
    
    def sample(self):
        return self.mu
    
class Bernoulli:
    def __init__(self, p=None, n=None):
        self.p = p 
        self.n = n
        
    def update(self, Q):
        p = np.mean(Q, axis=1)
        self.p = p
        self.n = p.shape[0]
    
    def sample(self, independent=False):
        return np.array([np.random.binomial(1, p_, 1)[0] for p_ in self.p])





