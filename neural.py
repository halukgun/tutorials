import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn import datasets

n_pts=500
centers=[[-0.5,0.5],[0.5,-0.5]]
X,Y=datasets.make_circles(n_samples=n_pts,random_state=123,noise=0.1,factor=0.2)
x_data=torch.Tensor(X)
y_data=torch.Tensor(Y.reshape(500,1))

def scattering():
    plt.scatter(X[Y==0,0],X[Y==0,1])
    plt.scatter(X[Y==1,0],X[Y==1,1])
    plt.show()



class Model(nn.Module):
    def __init__(self,input,output,hidden1):
        super().__init__()
        self.linear=nn.Linear(input,hidden1)#now we have hidden layer, so we need to include it
        self.linear2=nn.Linear(hidden1,output)
    def forward(self,x):
        x=torch.sigmoid(self.linear(x))
        x=torch.sigmoid(self.linear2(x))
        return x
    def prediction(self,x):
        prediction=self.forward(x)
        if prediction>=0.5:
            return 1
        else:
            return 0

torch.manual_seed(2)
model=Model(2,1,4)

criterion=nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.1)
losses=[]
epochs=1000
for i in range(epochs):
    y_pred=model.forward(x_data)
    loss=criterion(y_pred,y_data)
    losses.append(loss)
    print("Epoch: ",i," loss: ",loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs),losses)
plt.show()
plt.xlabel('epoch')
plt.ylabel('loss')

def plot_boundary(X,Y):
    x_span=np.linspace(min(X[:,0]),max(X[:,0]))
    y_span=np.linspace(min(X[:,1]),max(X[:,1]))
    x1,y1=np.meshgrid(x_span,y_span)#50x50 arrays
    grid=torch.Tensor(np.c_[x1.ravel(),y1.ravel()])#concatenation by column wise these arrays
    pred_func=model.forward(grid)
    z=pred_func.view(x1.shape).detach().numpy()
    plt.contourf(x1,y1,z)
    

plot_boundary(X,Y)
scattering()