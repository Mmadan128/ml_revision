import numpy as np

class LogisticRegression:
    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state

    def fit(self,x,y):
        rgen=np.random.RandomState(self.random_state)    
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=x.shape[1])
        self.b=np.float(0.)
        self.losses_=[]

        for i in range(self.n_iter):
            net_input=self.net_input(x)
            output=self.activation(net_input)
            errors=(y-output)
            self.w_+=self.eta*2.0*x.T.dot(errors)/x.shape[0]
            self.b_+=self.eta*2.0*errors.mean()
            loss=(-y.dot(np.log(output))
                  -((1-y).dot(np.log(1-output)))
                  /x.shape[0])
            self.losses_.append(loss)
        return self
    
    def net_input(self,x):
        return np.dot(x,self.w_)+self.b_
    
    def activation(self,z):
        return 1./(1. +np.exp(-np.clip(z,-250,250)))
    
    def predict(self,z):
        return 1./(1.+np.exp(-np.clip(z,-250,250)))
    
    def predict(self,x):
        return np.where(self.activation(self.net_input(x))>=0.5,1,0)
    