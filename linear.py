
import torch
import numpy as np
import torch.nn as nn 
import matplotlib.pyplot as plt
from torch.nn import Linear
#creating data sets
X=torch.randn(100,1)*10
Y=X+torch.randn(100,1)*3



class LR(nn.Module): 
    def __init__(self,in_features,out_features):
        super().__init__()
        self.linear=Linear(in_features,out_features)
    def forward(self,x):
        pred=self.linear(x)
        return pred


torch.manual_seed(1)
model=LR(1,1)
#to get weight and bias value
def get_params():
    [w,b]=model.parameters()
    return (w.item(),b.item())

def plot_fit(title):
    plt.title=title
    w1,b1=get_params() 
    x1=np.array([-30,30])
    y1=w1*x1+b1
    plt.plot(x1,y1,'r')
    plt.scatter(X,Y)
    plt.show()

criteria=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
epochs=100
losses=[]
for i in range(epochs):
    y_pred=model.forward(X) 
    loss=criteria(y_pred,Y) #calculatin loss value
    print("epoch: ",i, "loss: ",loss.item())
    losses.append(loss)
    optimizer.zero_grad() #to get zero result of gradient
    loss.backward() #taking derivative
    optimizer.step() #updating 

plot_fit("Trained Model")