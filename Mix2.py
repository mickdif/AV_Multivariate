import myModule
#  import mySentiment

import yfinance as yf
import technical_indicators_lib as til
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from sched import scheduler
print("All libraries loaded")

# general options
HLOCV = 1
indicatori = 1
medie = 0
# aggiorna_dati = 0 # 0 = no

do_plot = 1 # 0 no 
print_example = 0 # 1 no
save_output = 1 # 1 no

# LSTM configuration
config = {
    "data": {
        "window_size": 80, # quanti dati usare per predire la prossima chiusura
        "predicted_days": 1, # quanti gg prevedere
        "train_split_size": 0.80,
    },
    "model": {
        "output_size": 1,
        "num_lstm_layers": 2,
        "lstm_size": 100, # MOD erano 32
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

#
# CARICAMENTO DATI
#
TIT = yf.Ticker("TIT.MI")
hist = TIT.history("5y") # get historical market data
data = pd.DataFrame(hist) # data come indice

data = data.rename(columns={"Open": "open", "High": "high", "Low":"low",
                            "Close": "close", "Volume": "volume"}) # technical_indicators_lib vuole i nomi minuscoli
data = data.drop(["Dividends", "Stock Splits"], axis=1)
print("Dati caricati: ", data.shape)

#
# PREPARAZIONE DATI
#

# 0) calcolo medie e indicatori, applico general options
if medie == 1:
    data = til.SMA().get_value_df(data, 5)
    data = data.rename(columns={'SMA':'SMAshort'})
    data = til.SMA().get_value_df(data, 10)
    data = data.rename(columns={'SMA':'SMAlong'})

    data = til.EMA().get_value_df(data, 5)
    data = data.rename(columns={'EMA':'EMAshort'})
    data = til.EMA().get_value_df(data, 10)
    data = data.rename(columns={'EMA':'EMAlong'})

if indicatori == 1:
    data = til.RSI().get_value_df(data, 14)
    data = til.CCI().get_value_df(data, 14)
#  data = til.ADI().get_value_df(data) la maggior parte venirano vuote
    data = til.StochasticKAndD().get_value_df(data, 14)
    data = til.MACD().get_value_df(data)
    data = til.ATR().get_value_df(data, 14)

if HLOCV == 0:
    data = data.drop(["high", "low", "open", "volume"], axis=1)

# 1) Pulire: elimino righe con campi vuoti
print("data row shape: ", data.shape)
data.replace('', np.nan, inplace=True)
data.replace('null', np.nan, inplace=True)
data.dropna(inplace=True)
data.to_csv("output/clean_pd.csv")

input_size = data.shape[1]
print("Elaborazioni dati completa, dimensione matrice dati: ", data.shape)

# GRAFICO dei prezzi di chiusura  
if(do_plot == 1):
    df_to_plot = data.reset_index() # stampare coll'indice e' lento
    plt.plot(df_to_plot.close)
    # plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Adj close price")
    plt.savefig("output/initial_plot.png", dpi=250)
    plt.show();

# 2) normalizzo i dati
scaler = myModule.Normalizer()
data = scaler.fit_transform(data)
data = data.astype(float)
print("Dati normalizzati")

# 3) split: crea una matrice 3D: len(data) x giorni x n di variabili
# da ora in poi: X matrice dei dati, y vettore target
X_split, y_split = myModule.split_sequences(data, data.close.values, config["data"]["window_size"], config["data"]["predicted_days"])
print("Split effettuato; X_split: ", X_split.shape, " y_split: ", y_split.shape)
if print_example == 1:
    print("X_split[-1]: ")
    print(X_split[-1])
    print("y_split [-1]: ", y_split[-1])
    print("y_split [-2]: ", y_split[-2])
    print("y_split [-3]: ", y_split[-3])
    print("y_split [-4]: ", y_split[-4])

# 4) creo i set di training e di testing
train_test_cutoff = round(config["data"]["train_split_size"] * len(data))
X_train = X_split[:train_test_cutoff]
X_test = X_split[train_test_cutoff:]
y_train = y_split[:train_test_cutoff]
y_test = y_split[train_test_cutoff:]

# 5) Trasforma in tensori
X_train_tensors = torch.tensor(X_train, requires_grad=True)
X_test_tensors = torch.tensor(X_test, requires_grad=True)
y_train_tensors = torch.tensor(y_train, requires_grad=True)
y_test_tensors = torch.tensor(y_test, requires_grad=True)

# 6) Inserisce dati in un ogg di tipo Dataset
dataset_train = myModule.TimeSeriesDataset(X_train_tensors, y_train_tensors)
dataset_val = myModule.TimeSeriesDataset(X_test_tensors, y_test_tensors)

print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

# 7) Inserisce dati in un ogg di tipo DataLoader
train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

print("Dati pronti per LSTM")

#
# Creo LSTM
#
model = myModule.LSTM(config["model"]["output_size"],
              input_size, 
              config["model"]["lstm_size"], 
              config["model"]["num_lstm_layers"],
              config["model"]["dropout"])
model = model.to(config["training"]["device"])

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"]) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

#
# Alleno LSTM
#
for epoch in range(config["training"]["num_epoch"]):
    loss_train, lr_train = myModule.run_epoch(model, optimizer, criterion, scheduler, config["training"]["device"],
                                              train_dataloader, is_training=True)
    loss_val, lr_val = myModule.run_epoch(model, optimizer, criterion, scheduler, config["training"]["device"],
                                           val_dataloader)
    scheduler.step()

    print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
          .format(epoch + 1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))

# here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

model.eval()

#
# predict on the training data, to see how well the model managed to learn and memorize
#
predicted_train = np.array([])

for idx, (x, y) in enumerate(train_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    out = out.transpose()
    predicted_train = np.concatenate((predicted_train, out[0]))

predicted_train = scaler.inverse_transform_lin(predicted_train)
y_train = scaler.inverse_transform_lin(y_train)
# predicted_train = scaler.inverse_transform(predicted_train.reshape(-1, 1))
# y_train = scaler.inverse_transform(y_train.reshape(-1, 1))

print("myMSE train: ", myModule.myMSE(y_train, predicted_train))
print("MAPE train: ", np.round(myModule.MAPE(y_train, predicted_train)*100,1),"%")
if save_output == 1:
    np.savetxt("output/reali_train.csv", np.flip(y_train, 0), delimiter='\n', header="Reali train", fmt="%2f")
    np.savetxt("output/previsti_train.csv", np.flip(predicted_train, 0), delimiter='\n', header="Previsti train", fmt="%2f")
    
#
# predict on the validation data, to see how the model does
#
predicted_val = np.array([])
for idx, (x, y) in enumerate(val_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    out = out.transpose()
    predicted_val = np.concatenate((predicted_val, out[0]))
   
predicted_val = scaler.inverse_transform_lin(predicted_val.reshape(-1, 1))
y_test = scaler.inverse_transform_lin(y_test.reshape(-1, 1))

print("myMSE test: ", myModule.myMSE(y_test, predicted_val))
print("MAPE test: ", np.round(myModule.MAPE(y_test, predicted_val)*100,1),"%")
if save_output == 1:
    np.savetxt("output/reali_test.csv", np.flip(y_test, 0), delimiter='\n', header="Reali test", fmt="%2f")
    np.savetxt("output/previsti_test.csv", np.flip(predicted_val, 0), delimiter='\n', header="Previsti test", fmt="%2f")

# GRAFICI FINALI
if(do_plot == 1):
    plt.figure(figsize=(10,6)) #plotting
    plt.plot(y_train, label='Actual Train Data') # actual plot
    plt.plot(predicted_train, label='Predicted Train Data') # predicted plot
    
    plt.title('Time-Series Prediction: training data')
    plt.legend()
    plt.savefig("output/training.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10,6)) #plotting
    plt.plot(y_test, label='Actual Pred Data') # actual plot
    plt.plot(predicted_val, label='Predicted Pred Data') # predicted plot
    
    plt.title('Time-Series Prediction: evaluation data')
    plt.legend()
    plt.savefig("output/evaluation_data.png", dpi=300)
    plt.show()

print("FINE: dati e grafici (se) sono stati salvati nella cartella 'output'")
