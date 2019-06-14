#%%
import torch
import numpy as np
import matplotlib.pyplot as plt

x=torch.linspace(0,10,5)
y=torch.exp(x)
plt.plot(x.numpy(), y.numpy())
plt.show()


#%%
