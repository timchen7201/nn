import numpy as np



class Optimizer:
    def __init__(self,optimizer='sgd',learning_rate=.1, beta= .9):
        self.optimizer = optimizer
        self.beta = beta
        self.learning_rate = learning_rate
        self.velocity = {}


    def update(self,nn):
        if self.optimizer == 'sgd':
            self._sgd(nn)
        elif self.optimizer == 'momentum':
            if not bool(self.velocity):
                self._initial_velocity(nn)
            else:
                self._momentum(nn)
        elif self.optimizer == 'adam':
            if not bool(self.velocity):
                self._initial_velocity(nn)

    def _sgd(self,nn):
        for param in nn.params:
            nn.params[param] -= self.learning_rate*nn.gradients[param]
    
    def _momentum(self,nn):
        for param in nn.params:
            self.velocity = self.velocity[param]*self.beta - \
                             self.learning_rate*nn.gradients[param]
            nn.params[param] += self.velocity
    
    # def _adam(self,nn):
    #     for param in nn.params:

    
    def _initial_velocity(self,nn):
        for param in nn.params:
            self.velocity[param] = np.zeros(nn.params[param])
