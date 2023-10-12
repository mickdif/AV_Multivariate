
import os

from sched import scheduler

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators

print("All libraries loaded")

aggiorna_dati = 0 # 0 no
disegna = 0 # i grafici rallentano il programma, con questa var si possono disabilitare (0)
data_example = 1 # 0 no
salva_su_file = 0 # 0 no

config = {
    "alpha_vantage": {
        "key": "demo",
        "mykey": "URXBRZFHAJXF1NSB",
        "symbol": "IBM",
        "outputsize": "full",
        "key_adjusted_close": "5. adjusted close", # It is considered an industry best practice to use split/dividend-adjusted prices instead of raw prices to model stock price movements. 
    },
    "data": {
        "window_size": 20,
        "days_predicted": 1,
        "train_split_size": 0.9,
    },
    "plots": {
        "xticks_interval": 180, # show a date every xx days
    },
    "model": {
        "input_size": 5, # MOD dcp, rsi, stochrsi, willr, roc 
        "output_size": 1,
        "num_lstm_layers": 2,
        "lstm_size": 32, # MOD erano 32
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 64, # +++ MOD erano 64
        "num_epoch": 100, # +++ MOD erano 100
        "learning_rate": 0.01,# +++ MOD era 0.01
        "scheduler_step_size": 40,
    }
}


# se esistono i file coi dati li apre, altrimenti scarica i dati da AlphaVantage
if(aggiorna_dati == 0 &
   os.path.isfile("dati/date_file.npy") & 
   os.path.isfile("dati/dcp_file.npy") & 
   os.path.isfile("dati/rsi_file.npy") & 
   os.path.isfile("dati/stochrsi_file.npy") & 
   os.path.isfile("dati/willr_file.npy") & 
   os.path.isfile("dati/roc_file.npy")):

    data_date = np.load("dati/date_file.npy")
    dcp = np.load("dati/dcp_file.npy")
    rsi = np.load("dati/rsi_file.npy")
    stochrsi = np.load("dati/stochrsi_file.npy")
    willr = np.load("dati/willr_file.npy")
    roc = np.load("dati/roc_file.npy")  
else:
    print("Scaricando i dati...")
    def download_data(config):
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        ti = TechIndicators(key=config["alpha_vantage"]["mykey"])

        def elaborate_data(name, data):
            data_date = [date for date in data.keys()] # keys() restituisce la lista degli arg dell'ogg stesso
            data_date.reverse() # contiene lista date
            data_= [float(data[date][name]) for date in data.keys()]
            data_.reverse() # inverto l'array
            data_ = np.array(data_) # trasformo in array di numpy
            
            num_data_points = len(data_date)
            display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
            print("Number data points", num_data_points, display_date_range)

            return data_, data_date

        name = config["alpha_vantage"]["key_adjusted_close"]
        data, _ = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])
        dcp, data_date = elaborate_data(name, data)
        
        name = "RSI"
        data, _ = ti.get_rsi(config["alpha_vantage"]["symbol"], 'daily', '20', 'close')
        rsi, _ = elaborate_data(name, data)
        
        name = "FastK"
        data, _ = ti.get_stochrsi(config["alpha_vantage"]["symbol"], 'daily', '20', 'close')
        stochrsi, _ = elaborate_data(name, data)
        
        name = "WILLR"
        data, _ = ti.get_willr(config["alpha_vantage"]["symbol"], 'daily', '20')
        willr, _ = elaborate_data(name, data)
        
        name = "ROC"
        data, _ = ti.get_roc(config["alpha_vantage"]["symbol"], 'daily', '20')
        roc, _ = elaborate_data(name, data)
        
        return data_date, dcp, rsi, stochrsi, willr, roc
    
    
    data_date, dcp, rsi, stochrsi, willr, roc = download_data(config)
    # come e' fatto dcp: Mat(1,len(dcp)) cio? un vettore coi valori dal primo a ieri, in questo ordine per via del reverse

    np.save("dati/date_file", data_date)
    np.save("dati/dcp_file", dcp)
    np.save("dati/rsi_file", rsi)
    np.save("dati/stochrsi_file", stochrsi)
    np.save("dati/willr_file", willr)
    np.save("dati/roc_file", roc)

print("Dati caricati")

# Rendo i vettori della stessa lunghezza, gettando i valori pi? vecchi nel caso differiscano,
# fondamentale perch? DataFrame non accetta vettori con lunghezze differenti fra loro
len_dcp, len_rsi, len_stochrsi, len_willr, len_roc = len(dcp), len(rsi), len(stochrsi), len(willr), len(roc)
min_len = min(len_dcp, len_rsi, len_stochrsi, len_willr, len_roc)

data_date = data_date[len_dcp-min_len:]
dcp = dcp[len_dcp-min_len:]
rsi = rsi[len_rsi-min_len:]
stochrsi = stochrsi[len_stochrsi-min_len:]
willr = willr[len_willr-min_len:]
roc = roc[len_roc-min_len:]

# creo DataFrame
data = {'dcp': dcp,
        'rsi': rsi,
        'stochrsi': stochrsi,
        'willr': willr,
        'roc': roc}
df = pd.DataFrame(data, index=data_date)
print("DataFrame creato")

if salva_su_file == 1: 
    df.to_csv("output/panda.csv")
    print("e salvato in csv")

if(disegna == 1):
    df_to_plot = df.reset_index() # stampare coll'indice e' lento
    plt.plot(df_to_plot.dcp)
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.title("IBM dcp")
    plt.savefig("output/initial_plot.png", dpi=250)
    plt.show();

print("Ora normalizzo")
class Normalizer():
    # una gaussiana ha 2 parametri: mu (indica la media, il centro, x del picco) e la variazione std
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x): 
        self.mu = np.mean(x, axis=(0)) # mean() fa la media dei dati sull'asse specificato. keepdims=True tolto perch? non accettato da DataFrame
        self.sd = np.std(x, axis=(0)) # deviazione std.  
        normalized_x = (x - self.mu)/self.sd
        print(self.mu.shape, self.sd.shape)
        return normalized_x
        
    def inverse_transform(self, x):
        return (x*self.sd) + self.mu
    
    def inverse_transform_lin(self, x):
        return (x*self.sd.iloc[0]) + self.mu.iloc[0]

scaler = Normalizer()
print("df.shape", df.shape)
df = scaler.fit_transform(df)
df = df.astype(float)

X, y = df, df.dcp.values

# if(data_example == 1): print("X.shape X=df: ", X.shape, "y.shape y=df.dcp: ", y.shape)

def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        end_ix = i + n_steps_in # = i + 20
        if end_ix + n_steps_out > len(input_sequences): break # check if we are beyond the dataset

        seq_x = input_sequences[i:end_ix] # seq_x = inp[0,20]; seq_x = inp[1,21]; seq_x = inp[2,22] etc
        seq_y = output_sequence[end_ix:end_ix + n_steps_out] # seq_y = out[21]; seq_y = out[22]; seq_y = out[23] etc
        
        X.append(seq_x), y.append(seq_y)        
    return np.array(X), np.array(y)
# the function boils down to getting 20 samples from X, then looking at the 1 next indices in y, and patching these together.
# Note that because of this we'll throw out the first 1 values of y

X_ss, y_mm = split_sequences(X, y, config["data"]["window_size"], config["data"]["days_predicted"])

train_test_cutoff = round(config["data"]["train_split_size"] * len(X))
X_train = X_ss[:train_test_cutoff]
X_test = X_ss[train_test_cutoff:]
y_train = y_mm[:train_test_cutoff]
y_test = y_mm[train_test_cutoff:]

X_train_tensors = torch.tensor(X_train, requires_grad=True)
X_test_tensors = torch.tensor(X_test, requires_grad=True)
y_train_tensors = torch.tensor(y_train, requires_grad=True)
y_test_tensors = torch.tensor(y_test, requires_grad=True)

class TimeSeriesDataset(Dataset):
 
  def __init__(self, x, y):
 
    self.x = torch.tensor(x, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.float32)
 
  def __len__(self):
    return len(self.y)
   
  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]


dataset_train = TimeSeriesDataset(X_train_tensors, y_train_tensors)
dataset_val = TimeSeriesDataset(X_test_tensors, y_test_tensors)

print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

class LSTM(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size # input size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.hidden_size = hidden_size # neurons in each lstm layer
        self.output_size = output_size # output size

        # LSTM model
        self.fc_1 =  nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=config["model"]["dropout"]) # lstm
        self.dropout = nn.Dropout(config["model"]["dropout"])

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

def run_epoch(dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr

model = LSTM(  config["model"]["output_size"],
              config["model"]["input_size"], 
              config["model"]["lstm_size"], 
              config["model"]["num_lstm_layers"])
model = model.to(config["training"]["device"])
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"]) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

for epoch in range(config["training"]["num_epoch"]):
    loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
    loss_val, lr_val = run_epoch(val_dataloader)
    scheduler.step()

    print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
          .format(epoch + 1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))

# here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

model.eval()

# predict on the training data, to see how well the model managed to learn and memorize
# predicted_train = np.array([])

# for idx, (x, y) in enumerate(train_dataloader):
#     x = x.to(config["training"]["device"])
#     out = model(x)
#     out = out.cpu().detach().numpy()
#     predicted_train = np.concatenate((predicted_train, out[0]))

# predict on the validation data, to see how the model does
predicted_val = np.array([])
for idx, (x, y) in enumerate(val_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    # print(out)
    out = out.cpu().detach().numpy()
    out = out.transpose()
    # print(out)
    predicted_val = np.concatenate((predicted_val, out[0]))
   
predicted_val = scaler.inverse_transform_lin(predicted_val)
y_test = scaler.inverse_transform_lin(y_test)

np.savetxt("output/previsti.csv", np.flip(predicted_val, 0), delimiter='\n', header="Previsti", fmt="%2f")
np.savetxt("output/Reali.csv", np.flip(y_test, 0), delimiter='\n', header="Reali", fmt="%2f")

print("FINE, CIAO")