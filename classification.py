from sklearn.datasets import make_circles

X, y = make_circles(n_samples=1000,noise=0.03,random_state=42)

import matplotlib.pyplot as plt

#plt.scatter(x=X[:,0],y=X[:,1],c=['b' if label == 1 else 'r' for label in y])

#plt.show()

import torch

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from torch import nn 

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(device)

model = nn.Sequential(
    nn.Linear(in_features= 2,out_features= 5),
    nn.Linear(in_features= 5,out_features= 1)
).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.01)



X_train=X_train.to(device)
X_test=X_test.to(device)
y_train = y_train.to(device)
y_test=y_test.to(device)
loss_fn = loss_fn.to(device)



steps = 100

def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


for step in range(steps):
    model.train()
    y_logits = model(X_train).squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labls

    loss = loss_fn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                    y_train) 
    acc = accuracy_fn(y_true=y_train, 
                        y_pred=y_pred) 

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                                y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if step % 10 == 0:
        print(f"Epoch: {step} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")





