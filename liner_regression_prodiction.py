import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
import numpy as np

weight = 0.7
bais = 0.3

X = torch.arange(0,1,0.02).unsqueeze(dim=1)
y = weight * X +bais

train_split = int(0.8 * len(X))
x_train,y_train = X[:train_split], y[:train_split]
x_test,y_test = X[train_split:],y[train_split:]


def plot_prodictions(trian_data = x_train,train_lable = y_train, test_data = x_test,test_lable = y_test,prodictions = None):
    plt.figure(figsize=(10,7))

    plt.scatter(trian_data,train_lable,c="b", s=4)
    plt.scatter(test_data,test_lable,c="g",s=4)

    if(prodictions is not None):
        plt.scatter( test_data,prodictions,c="r",s=4)

    plt.legend(prop={"size": 14});
    plt.show()

class LinerRegressionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))
        self.bais = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x +self.bais
    
module_0 = LinerRegressionModule()
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=module_0.parameters(), # parameters of target model to optimize
                            lr=0.01)

epochs = 200

epoch_count = []
train_loss_values = []
test_loss_values = []

for epoch in range(epochs):
    module_0.train()
    y_pred = module_0(x_train)
    loss = loss_fn(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    module_0.eval()

    with torch.inference_mode():
        test_pred = module_0(x_test)
        test_loss = loss_fn(test_pred,y_test)
        if epoch % 10 ==0:
            print(f'epoch:{epoch},loss:{loss},test_loss:{test_loss}')
        
        if epoch == 199:
            plot_prodictions(prodictions=test_pred)




