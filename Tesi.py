import myModule
import mySentiment

import yfinance as yf
import technical_indicators_lib as til
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
print("All libraries loaded")

nome = "output/w.txt"

# input options
HLOCV = 0
indicatori = 0
medie = 0
sentiment = 1
query='Campari'

# general options
do_plot = 0 # 0 no 
print_example = 0 # 0 no
save_output = 0 # 0 no

# configuration
config = {
    "titolo": "CPR.MI", 
    "periodo": "1y",
    
    "window_size": 20, # quanti dati usare per predire la prossima chiusura
    "predicted_days": 1, # quanti gg prevedere
    "ripetizioni": 8,
    "train_split_size": 0.80,

    "hidden_layer": 2,
    "hidden_layer_size": 32, 

    "num_epoch": 10,
    "learning_rate": 0.01,
}
print(config)
with open(nome, 'w') as f: f.write(str(config))

##
# scarico dati
##
TIT = yf.Ticker(config["titolo"])
hist = TIT.history(config["periodo"]) # get historical market data
data = pd.DataFrame(hist) # data come indice
data = data.drop(["Dividends", "Stock Splits"], axis=1)
data = data.rename(columns={"Open": "open", "High": "high", "Low":"low",
                            "Close": "close", "Volume": "volume"}) # technical_indicators_lib vuole i nomi minuscoli

##
# calcolo medie e indicatori, applico input options
##
if medie == 1:
    data = til.SMA().get_value_df(data, 7)
    data = data.rename(columns={'SMA':'SMAshort'})
    data = til.SMA().get_value_df(data, 14)
    data = data.rename(columns={'SMA':'SMAlong'})

    data = til.EMA().get_value_df(data, 7)
    data = data.rename(columns={'EMA':'EMAshort'})
    data = til.EMA().get_value_df(data, 14)
    data = data.rename(columns={'EMA':'EMAlong'})
    
    with open(nome, 'a') as f: f.write("\ncon medie")

if indicatori == 1:
    data = til.RSI().get_value_df(data, 14)
    # data = til.CCI().get_value_df(data, 14)
    # data = til.StochasticKAndD().get_value_df(data, 14)
    # data = til.MACD().get_value_df(data)
    # data = til.ATR().get_value_df(data, 14)
    
    with open(nome, 'a') as f: f.write("\ncon rsi")

if HLOCV == 0:
    data = data.drop(["high", "low", "open", "volume"], axis=1)
    with open(nome, 'a') as f: f.write("\nsolo adj close")
else:
    with open(nome, 'a') as f: f.write("\ncon HLOCV")

input_size = data.shape[1]  
##
# Filtro per data (puo' tornare utile)
##
# data.index = pd.to_datetime(data.index, format='%Y-%m-%d') # Convert the date to datetime64
# data = data.loc[data.index < '2023-10-23']
# data = data.loc[(data.index >= '2013-10-31') & (data.index < '2023-10-23')]

X, y, dd = data, data.close.values, data.index.values # verranno usati come sinonimi alternativamente senza particolare logica

## 
# Pulire: elimino righe con campi vuoti, calcolo input_size
##
data.replace('', np.nan, inplace=True)
data.replace('null', np.nan, inplace=True)
data.dropna(inplace=True)

if print_example == 1: data.to_csv("output/clean_pd.csv")

##
# NORMALIZZO
##
ss = StandardScaler()
mm = MinMaxScaler()

X_trans = ss.fit_transform(data)
y_trans = mm.fit_transform(data.close.values.reshape(-1, 1)) 
if print_example == 1: print("len X_trans", len(X_trans))

##
# SPLIT
##
X_ss, y_mm = myModule.split_sequences(X_trans, y_trans, config["window_size"], config["predicted_days"])
if print_example == 1: print("X_ss e y_mm shape", X_ss.shape, y_mm.shape)

##
# CUTOFF
##
total_samples = len(X_ss)
new_cutoff = config["predicted_days"]*config["ripetizioni"]
train_test_cutoff = round(config["train_split_size"] * (total_samples-new_cutoff))
print("train_test_cutoff: ", train_test_cutoff)

X_train = X_ss[:train_test_cutoff]
X_test = X_ss[train_test_cutoff:-new_cutoff]

y_train = y_mm[:train_test_cutoff]
y_test = y_mm[train_test_cutoff:-new_cutoff] 

X_new = X_ss[-new_cutoff:]
y_new = y_mm[-new_cutoff:]

# data_date = data_date[]
print("Training Shape:", X_train.shape, y_train.shape)
print("Testing Shape:", X_test.shape, y_test.shape) 

# GRAFICO dei prezzi di chiusura 
display_date_range = data.index[0].strftime("%b %Y") + " - " + data.index[-1].strftime("%b %Y")
if(do_plot == 1):
    df_to_plot = data.reset_index() # stampare coll'indice e' lento
    plt.plot(df_to_plot.close)
    # plt.xlabel("Time")
    plt.ylabel("USD $")
    plt.title( config["titolo"] + " - Adj close price - " + display_date_range)
    plt.axvline(label='prediction', x=len(y)-config["predicted_days"], c='g', linestyle='--')
    plt.axvline(label='training-testing', x=train_test_cutoff, c='r', linestyle='--') # size of the training set
    plt.savefig("output/initial_plot.png", dpi=250)
    plt.show();

##
# TENSOR
##
X_train_tensors = torch.tensor(X_train, requires_grad=True, dtype=torch.double)
X_test_tensors = torch.tensor(X_test, requires_grad=True, dtype=torch.double)
X_new_tensors = torch.tensor(X_new, requires_grad=True, dtype=torch.double)
y_train_tensors = torch.tensor(y_train, requires_grad=True, dtype=torch.double)
y_test_tensors = torch.tensor(y_test, requires_grad=True, dtype=torch.double)
y_new_tensors = torch.tensor(y_new, requires_grad=True, dtype=torch.double)

X_train_tensors_final = torch.reshape(X_train_tensors,   
                                      (X_train_tensors.shape[0], config["window_size"], 
                                       X_train_tensors.shape[2]))
X_test_tensors_final = torch.reshape(X_test_tensors,  
                                     (X_test_tensors.shape[0], config["window_size"], 
                                      X_test_tensors.shape[2])) 
X_new_tensors_final = torch.reshape(X_new_tensors,  
                                     (X_new_tensors.shape[0], config["window_size"], 
                                      X_new_tensors.shape[2])) 

print("Training Shape:", X_train_tensors_final.shape, y_train_tensors.shape)
print("Testing Shape:", X_test_tensors_final.shape, y_test_tensors.shape) 

if print_example == 1:
    X_check, y_check = myModule.split_sequences(X, y.reshape(-1, 1), config["window_size"], config["predicted_days"])
    X_check[-1][0:4]
    print("y_check[-1]", y_check[-1])
    print("data.close.values[-50:]", data.close.values[-50:])

import warnings # evita warining su copyconstructor, consiglia di usare clone
warnings.filterwarnings('ignore')

##
# LSTM
##
lstm = myModule.LSTM(config["predicted_days"], 
              input_size, 
              config["hidden_layer_size"], 
              config["hidden_layer"])
lstm.double() # evita err "expected float found double" (o vv) ricorda che double = float64

loss_fn = torch.nn.MSELoss()    # mean-squared error for regression
optimiser = torch.optim.Adam(lstm.parameters(), lr=config["learning_rate"]) 

myModule.training_loop(n_epochs = config["num_epoch"],
              lstm = lstm,
              optimiser = optimiser,
              loss_fn = loss_fn,
              X_train = X_train_tensors_final,
              y_train = y_train_tensors,
              X_test = X_test_tensors_final,
              y_test = y_test_tensors)

##
# PREVISIONI
##
# ricalco e ritrasformo perche' le vecchie variabili le ho usate nel training(?)
    # normalizzo
df_X_ss = ss.transform(data) 
df_y_mm = mm.transform(data.close.values.reshape(-1, 1))
    # split
df_X_ss, df_y_mm = myModule.split_sequences(df_X_ss, df_y_mm, config["window_size"], config["predicted_days"])
    # tensors
df_X_ss = torch.tensor(df_X_ss, requires_grad=True)
df_y_mm = torch.tensor(df_y_mm, requires_grad=True)
    # reshaping
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], config["window_size"], df_X_ss.shape[2]))

# PREVEDO su training e test dataset
    # forward pass
train_predict = lstm(df_X_ss) 
    # numpy conversion
data_predict = train_predict.data.numpy() 
dataY_plot = df_y_mm.data.numpy()
    # reverse transformation
data_predict = mm.inverse_transform(data_predict)
dataY_plot = mm.inverse_transform(dataY_plot)

true, preds = [], []
for i in range(len(dataY_plot)):
    true.append(dataY_plot[i][0])
for i in range(len(data_predict)):
    preds.append(data_predict[i][0])

# GRAFICO
if do_plot == 1:
    plt.figure(figsize=(10,6)) #plotting
    plt.axvline(label='training-testing', x=train_test_cutoff, c='r', linestyle='--') # size of the training set

    plt.plot(true, label='Actual Data') # actual plot
    plt.plot(preds, label='Predicted Data') # predicted plot
    plt.title(config["titolo"] + ' - Time-Series Prediction')
    plt.grid(which='both', axis='x')
    plt.legend()
    plt.savefig("output/train&test.png", dpi=300)
    #plt.show() 

# myModule.metriche(true[train_test_cutoff:], preds[train_test_cutoff:], "Metriche sui soli dati di test")
myModule.metriche(true, preds, "Metriche sui dati di training e di test")

# PREVEDO su nuovi dati
array_pred, array_target = [], []
for i in range(config["ripetizioni"]):
    test_predict = lstm(X_new_tensors_final[i].unsqueeze(0))
    test_predict = test_predict.detach().numpy()
    
    test_target = y_new_tensors[i].detach().numpy().reshape(1, -1)

    test_predict = mm.inverse_transform(test_predict)
    test_target = mm.inverse_transform(test_target)

    array_pred = np.append(array_pred, test_predict[0].tolist())
    array_target = np.append(array_target, test_target[0].tolist())

test_target, test_predict = array_target, array_pred
myModule.metriche(test_target, test_predict, "Metriche sui dati nuovi")

# GRAFICO
if do_plot == 1:
    plt.figure(figsize=(10,6)) #plotting
    a = [x for x in range(int(len(y)-(config["window_size"]+config["predicted_days"]*config["ripetizioni"])), len(y))]
    plt.plot(a, y[-(config["window_size"]+config["predicted_days"]*config["ripetizioni"]):], label='Actual data: ws + pred')
    c = [x for x in range(len(y)-config["predicted_days"]*config["ripetizioni"], len(y))]
    plt.plot(c, test_predict, label='One-shot multi-step prediction', marker=".", markersize=10)
    
    plt.axvline(label='prediction', x=len(y)-config["predicted_days"]*config["ripetizioni"], c='g', linestyle='--')
    plt.title(config["titolo"] + " - Partial date and " + str(config["predicted_days"]*config["ripetizioni"]) + " days prediction")
    plt.grid(which='both', axis='x')
    plt.legend()
    plt.savefig("output/small_plot.png", dpi=300)
    plt.show()

############################################
############## SENTIMENT ###################
############################################
if sentiment == 1:
    data_sa = np.zeros((new_cutoff,2))
    Sent = mySentiment.Sentiment_Analysis()
    for i in range(new_cutoff):
        idx1 = new_cutoff - i
        idx2 = idx1 + 5
        s_a = Sent.do_Analysis(query=query, start=data.index[-idx2], 
                                            end=data.index[-idx1],
                                            debug=True)
        # print("media: ", s_a)
        data_sa[i][0] = test_predict[i]
        data_sa[i][1] = s_a
    print(data_sa)
    sa_pred = np.zeros(new_cutoff)
    for i in range(new_cutoff):
        sa_pred[i] = 3.0/100 * data_sa[i][0] * data_sa[i][1] + data_sa[i][0]
    
myModule.metriche(test_target, sa_pred, "Metriche con SA") 




