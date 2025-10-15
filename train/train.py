import torch
from torch import nn
import numpy as np
import pandas as pd
import copy



def train(epoch,
          models,
          device,
          optimizers,
          criterions,
          train_loader,
          test_loader,
          pre_train_bool=False,
          train_loder_pre_train=None,
          test_loader_pre_train=None,
          epochs_pre_train=10,
          pre_train_return_best=True,
          bv_model_return_best=True
          ):
    if pre_train_bool:
        if train_loder_pre_train is None or test_loader_pre_train is None or epochs_pre_train is None:
            raise ValueError("train_loder_pre_train 、 test_loader_pre_train and epochs_pre_train must be provided for pre-training")
        if len(models) !=2:
            raise ValueError("models must be a list of two models fro pre-training modeland get_bv model")
        if len(optimizers) !=2:
            raise ValueError("optimizers must be a list of two optimizers fro pre-training modeland get_bv model")
        if len(criterions) !=2:
            raise ValueError("criterions must be a list of two criterions fro pre-training modeland get_bv model")
    if isinstance(models,list):
        models=[_.to(device) for _ in models]
    else:
        models=models.to(device)
    if pre_train_bool:
            best_pre_model= pre_train( models[0], 
                                      train_loder_pre_train,
                                       optimizers[0],
                                       device,
                                      criterions[0], 
                                      epochs_pre_train,
                                      return_best=pre_train_return_best
                                      )
            if hasattr(best_pre_model, 'state_dict'):
                partial_params = {k: v for k, v in best_pre_model.state_dict().items() if "con2" not in k}
            else:
                partial_params = {k: v for k, v in best_pre_model.items() if "con2" not in k}
        
            model2=models[1]
            model2.load_state_dict(partial_params,strict=False)
    else:
        model2=models
    optimizer=optimizers[1] if pre_train_bool else optimizers
    criterion=criterions[1] if pre_train_bool else criterions
    model2=  train_bv(epoch, model2, device,optimizer, criterion,train_loader,test_loader,return_best=bv_model_return_best)
    return model2
    
    
            

def pre_train(model,train_loader,optimizer,device,criterion,epochs=100,return_best=True):
    best_acc=0
    best_model = model
    for epoch in range(epochs):
        avg_loss = 0
        for inputs, labels in train_loader:
            # print(inputs.shape)
            model.train()
            bools=((labels.sum(axis=1)!=0).unsqueeze(1).broadcast_to(labels.shape)).to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            outputs_=outputs*bools
            loss = criterion(outputs_, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        # acc_train=get_acc(model,train_loader,device)
        acc_valid=get_acc(model,train_loader,device)
        if return_best:
            if acc_valid>best_acc:
                best_acc=acc_valid
                best_model=copy.deepcopy(model)
        # if (epoch+1)%10==0:

        #     ind,pop1,pop2  = get_re_indANDpop_cor(train_loader,model)
        #     print(f'Epoch [{epoch+1}/{10}], ind: {ind:.4f}, pop1: {pop1:.4f}, pop2: {pop2:.4f}')
        # print(f'Epoch [{epoch+1}/{10}], Loss: {avg_loss/len(train_loader):.4f}')
        # print(f'Epoch [{epoch+1}/{10}], Accuracy_train: {acc_train:.4f}, Accuracy_valid: {acc_valid:.4f}')
    if return_best:
        return best_model
    else:
        return model

def get_acc(model,data_loder,device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # print(len(data_loder))
        for x, y in data_loder:
            x = x.to(device)
            y = y.to(device)
            bool_=(y.sum(axis=1)!=0)
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim=1)
            y_true = torch.argmax(y, dim=1)
            correct += ((y_pred == y_true)*bool_) .sum()
            total += y.shape[0]*y.shape[1]-bool_.sum()
    return correct / total


def train_bv(
              epoch,
              model, 
              device,
              optimizer, 
              criterion,
              data_loader,
              test_loader,
              return_best=True 
):
    model.train()
    fold_loss_values = []
    all_loss = 0    
    best_acc=0
    best_model = model
    for i in range(epoch):
        
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1), target.view(-1))
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} '
                    f'({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_loss = all_loss / len(data_loader)
        fold_loss_values.append(avg_loss)
        print(f'=========> Epoch: {epoch} Average loss: {avg_loss:.4f}')
        # if epoch % test_gap == 0:
            
        acc = test_bv(model, test_loader,device)
        if return_best:
            if acc[0] > best_acc:
                best_acc = acc[0]
                best_model = copy.deepcopy(model)
            # torch.save(model.state_dict(), 'pre_train_model.pth')
        
    if return_best:
        return best_model
    else:
        return model


def test_bv(model,  test_loader,device):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            corr = np.corrcoef(output.view(-1).detach().cpu().numpy(), 
                              target.numpy().reshape(-1))[0, 1]
            print(corr)
            return corr, output.view(-1).detach().cpu().numpy()
        
        
if __name__ == "__main__":
    pass