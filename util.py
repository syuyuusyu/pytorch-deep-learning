import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple


def accuracy_fn(y_true:torch.Tensor,y_pred:torch.Tensor) -> float:
    correct =  y_true.eq(y_pred).sum().item()
    return (correct / len(y_true)) * 100

def train_step(module:nn.Module,data_loader:DataLoader,loss_fn = nn.Module,optimizer = torch.optim.Optimizer,accuracy_fn = None, device:torch.device = 'cpu')->Tuple:
    train_loss,train_acc,epoch_loss,epoch_acc = 0,0,[],[]
    for X,y in data_loader:
        module.to(device)
        X,y = X.to(device), y.to(device)        
        y_pred = module(X)
        loss = loss_fn(y_pred,y)
        print(loss)
        if hasattr(module,'l2_regularization'):
            reg = module.l2_regularization()
            loss += reg
        train_loss += loss
        epoch_loss.append(loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1)
        optimizer.step()
        if accuracy_fn:
            acc = accuracy_fn(y,y_pred.argmax(dim=1))
            train_acc += acc
            epoch_acc.append(acc)
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f'train_loss:{train_loss}, trian:acc:{train_acc}')
    return train_loss,train_acc,epoch_loss,epoch_acc

def test_step(module:nn.Module,data_loader:DataLoader,loss_fn = nn.Module,accuracy_fn = None, device:torch.device = 'cpu')->Tuple:
    test_loss, test_acc,epoch_loss,epoch_acc = 0, 0,[],[]
    with torch.inference_mode():
        module.to(device)
        module.eval()
        for X,y in data_loader:
            
            X, y = X.to(device),y.to(device)
            y_pred = module(X)
            #y_pred = y_pred.argmax(dim=1)
            loss = loss_fn(y_pred,y)
            epoch_loss.append(loss)
            test_loss += loss
            if accuracy_fn:
                acc = accuracy_fn(y,y_pred.argmax(dim=1))
                epoch_acc.append(acc)
                test_acc += acc
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
    print(f'test loss {test_loss}, test acc:{test_acc}')
    return test_loss,test_acc,epoch_loss,epoch_acc

import requests
def download_file(url, save_path):
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # 检查是否发生 HTTP 错误

            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)

        print(f"Downloaded and saved as: {save_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

import os
def is_file_in_directory(file_name, directory_path):
    file_path = os.path.join(directory_path, file_name)
    return os.path.isfile(file_path)

import sys
def test():
    print('1111'+sys.executable)
        