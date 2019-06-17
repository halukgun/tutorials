
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import torch.nn as nn

n_points=100
centers=[[-0.5,0.5],[0.5,-0.5]]
X,Y =datasets.make_blobs(n_samples=n_points,random_state=123,centers=centers,cluster_std=0.4)
x_data=torch.Tensor(X)
y_data=torch.Tensor(Y)
def scattering():
    plt.scatter(X[Y==0,0],X[Y==0,1])
    plt.scatter(X[Y==1,0],X[Y==1,1])
    plt.show()
class Model(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.linear=nn.Linear(input_size,output_size)
    def forward(self,x): #for prediction
        pred=torch.sigmoid(self.linear(x))#probability
        return pred
    def prediction(self,x): #to specify point class location on graph
        pred=self.forward(x)
        if pred>=0.5:
            return 1
        else:
            return 0

torch.manual_seed(2)
model=Model(2,1)
def get_params():
    [w,b]=model.parameters()
    w1,w2=w.view(2)
    return(w1.item(),w2.item(),b[0].item())

def plot_fit(Title):
    plt.title=Title
    w1,w2,b=get_params()
    x1=np.array([-2.0,2.0])
    x2=(w1*x1+b)/-w2
    plt.plot(x1,x2,'r')
    scattering()
    plt.show()

plot_fit('Initial Model')
criterion=nn.BCELoss()#computing error
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
epochs=1000
losses=[]
for i in range(epochs):
    pred=model.forward(x_data)
    loss=criterion(pred,y_data)
    print("epoch: ",i," loss: ",loss.item())
    losses.append(loss)
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()

plt.plot(range(epochs),losses)
plt.show()
plot_fit('Trained Model')

def Model_Testing(p1,p2):
    plt.plot(p1.numpy()[0],p1.numpy()[1],'ro')
    plt.plot(p2.numpy()[0],p2.numpy()[1],'ko')
    print("Red point positive prob.= {}".format(model.forward(p1).item()))
    print("Black point positive prob.= {}".format(model.forward(p2).item()))
    print("Red point class {}".format(model.prediction(p1)))
    print("Black point class {}".format(model.prediction(p2)))
    plot_fit('Trained Model')

p1=torch.Tensor([2.0,-2.0])
p2=torch.Tensor([-2.0,2.0])
Model_Testing(p1,p2)