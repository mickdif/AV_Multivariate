import myModule # vd file myModule

import technical_indicators_lib as til
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from sched import scheduler


print("All libraries loaded")

# 
# Alcune variabili per modificare il funzionamento del programma (alcuni if vogliono un 2)
# 

indicatori = 1 # 0 = usa solo close
aggiorna_dati = 1 # 0 = no
disegna = 0 # 0 no 
data_example = 1 # 0 no
salva_su_file = 1 # 0 no

config = {
    "data": {
        "window_size": 21, # quanti dati usare per predire la prossima chiusura
        "days_predicted": 1, # quanti gg prevedere
        "train_split_size": 0.8,
    },
    "model": {
        "output_size": 1,
        "num_lstm_layers": 2,
        "lstm_size": 16, # MOD erano 32
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

data = pd.read_csv(r"5yTIT.MI.csv")
data = data.drop(columns=["Close"])
data = data.rename(columns={"Open": "open", "High": "high", "Low":"low", "Adj Close": "close", "Volume": "volume"}) # technical_indicators_lib vuole i nomi minuscoli

data = til.SMA().get_value_df(data, 7)
data = data.rename(columns={'SMA':'SMA7'})
data = til.SMA().get_value_df(data, 14)
data = data.rename(columns={'SMA':'SMA14'})

data = til.EMA().get_value_df(data, 7)
data = data.rename(columns={'EMA':'EMA7'})
data = til.EMA().get_value_df(data, 14)
data = data.rename(columns={'EMA':'EMA14'})

data = til.RSI().get_value_df(data, 14)
data = til.CCI().get_value_df(data, 14)
data = til.ADI().get_value_df(data)
data = til.StochasticKAndD().get_value_df(data, 14)
data = til.MACD().get_value_df(data)

#
# PREPARAZIONE DATI
#

# 1) Pulire: elimino righe con campi vuoti
data.replace('', np.nan, inplace=True)
data.replace('null', np.nan, inplace=True)
data.dropna(inplace=True)

df = data
df.to_csv("pd.csv")
print("Dati caricati")

df = df.drop(columns=["Date"])

if indicatori == 0:
    tmp = {'close': df['close']}
    df = pd.DataFrame(tmp)
    input_size = 1
else:
    input_size = df.shape[1]

# GRAFICO dei prezzi di chiusura  
if(disegna == 1):
    df_to_plot = df.reset_index() # stampare coll'indice e' lento
    plt.plot(df_to_plot.dcp)
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.title("IBM dcp")
    plt.savefig("output/initial_plot.png", dpi=250)
    plt.show();

# 2) normalizzo i dati
scaler = myModule.Normalizer()
df = scaler.fit_transform(df)
df = df.astype(float)
print("Dati normalizzati")
X, y = df, df.close.values

# 3) split: crea una matrice 3D: len(data) x giorni x n di variabili
X_split, y_split = myModule.split_sequences(X, y, config["data"]["window_size"], config["data"]["days_predicted"])

# 4) creo i set di training e di testing
train_test_cutoff = round(config["data"]["train_split_size"] * len(X))
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
if data_example == 1:
    print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
    print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

# 7) Inserisce dati in un ogg di tipo DataLoader
train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

if data_example == 1: print("Dati preelaborati")

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
    loss_train, lr_train = myModule.run_epoch(model, optimizer, criterion, scheduler, config["training"]["device"], train_dataloader, is_training=True)
    loss_val, lr_val = myModule.run_epoch(model, optimizer, criterion, scheduler, config["training"]["device"], val_dataloader)
    scheduler.step()

    print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
          .format(epoch + 1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))

# here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

model.eval()

# predict on the training data, to see how well the model managed to learn and memorize
predicted_train = np.array([])

for idx, (x, y) in enumerate(train_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    out = out.transpose()
    predicted_train = np.concatenate((predicted_train, out[0]))

predicted_train = scaler.inverse_transform_lin(predicted_train)
y_train = scaler.inverse_transform_lin(y_train)

if salva_su_file == 2:
    np.savetxt("output/reali_train.csv", np.flip(y_train, 0), delimiter='\n', header="Reali train", fmt="%2f")
    np.savetxt("output/previsti_train.csv", np.flip(predicted_train, 0), delimiter='\n', header="Previsti train", fmt="%2f")

# predict on the validation data, to see how the model does
predicted_val = np.array([])
for idx, (x, y) in enumerate(val_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    out = out.transpose()
    predicted_val = np.concatenate((predicted_val, out[0]))
   
predicted_val = scaler.inverse_transform_lin(predicted_val)
y_test = scaler.inverse_transform_lin(y_test)

if salva_su_file == 1:
    np.savetxt("output/reali_pred.csv", np.flip(y_test, 0), delimiter='\n', header="Reali pred", fmt="%2f")
    np.savetxt("output/previsti_pred.csv", np.flip(predicted_val, 0), delimiter='\n', header="Previsti pred", fmt="%2f")

if(disegna == 1):
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

print("FINE: dati e grafici sono stati salvanti nella cartella 'output'")
