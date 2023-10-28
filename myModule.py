import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

def myMSE(real, pred):
    if len(real) !=  len(pred):
        print("myMSE: la lunghezza dei due vettori deve essere uguale!")
        return -1
    else:
        p_sum = 0
        for idx in range(len(real)):
            p_sum += pow( real[idx] - pred[idx], 2)
            
        return p_sum / len(real)

def MAPE(real, pred):
    real, pred = np.array(real), np.array(pred)
    
    if len(real) !=  len(pred):
        print("myDifPerc: la lunghezza dei due vettori deve essere uguale!")
        return -1
    
    else:
        mape = np.mean(np.abs((real - pred) / real))
        return mape
        
class Normalizer():
    # una gaussiana ha 2 parametri: mu (indica la media, il centro, x del picco) e la variazione std
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x): 
        self.mu = np.mean(x, axis=(0)) # mean() fa la media dei dati sull'asse specificato
        self.sd = np.std(x, axis=(0)) # deviazione std
        normalized_x = (x - self.mu)/self.sd
        print(self.mu.shape, self.sd.shape)
        return normalized_x
        
    def inverse_transform(self, x):
        return (x*self.sd) + self.mu
    
    def inverse_transform_lin(self, x):
        return (x*self.sd.iloc[0]) + self.mu.iloc[0]
    
def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out): # x, y, windows size, days predicted
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        end_ix = i + n_steps_in # = i + 20
        out_end_ix = end_ix + n_steps_out # = 21
        if out_end_ix > len(input_sequences): break # check if we are beyond the dataset

        seq_x = input_sequences[i:end_ix] # seq_x = inp[0,20]; seq_x = inp[1,21]; seq_x = inp[2,22] etc
        seq_y = output_sequence[end_ix:out_end_ix] # seq_y = out[20:21]=out[21]; seq_y = out[22]; seq_y = out[23] etc
        
        X.append(seq_x), y.append(seq_y)        
    return np.array(X), np.array(y)

class TimeSeriesDataset(Dataset):
 
  def __init__(self, x, y):
    # self.x = torch.tensor(x, dtype=torch.float32)
    # self.x = x.clone().detach().requires_grad_(True).float()
    self.x = x.clone().requires_grad_(True).float()
    self.y = y.clone().requires_grad_(True).float()

  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]

class LSTM(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers, drop=0):
        super().__init__()
        self.input_size = input_size # input size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.hidden_size = hidden_size # neurons in each lstm layer
        self.output_size = output_size # output size

        # LSTM model
        self.fc_1 =  nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=drop) # lstm
        self.dropout = nn.Dropout(drop)

        self.fc_2 = nn.Linear(num_layers * hidden_size, output_size)
     
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self,x):
        # propagate input through LSTM
        x = x.float()
        batchsize = x.shape[0]   
        # layer 1
        x = self.fc_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        h_n = h_n.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        
        # layer 2
        x = self.dropout(x)
        out = self.fc_2(x)       

        return out

def run_epoch(model, optimizer, criterion, scheduler, engine, dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(engine)
        y = y.to(engine)

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr
