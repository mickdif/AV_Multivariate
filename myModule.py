import numpy as np
import torch
import torch.nn as nn

    
def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    # per es n_step_in = 100 e n_step_out = 50
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in # = i + 100
        out_end_ix = end_ix + n_steps_out - 1 # i + 100 + 50 - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x = input_sequences[i:end_ix] # [0:100], [1:101] etc
        seq_y = output_sequence[end_ix-1:out_end_ix, -1] # [99:149], [100, 150] etc
        # il -1 è relativo alla colonna, l'ultima
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)


class LSTM(nn.Module):    
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_classes = num_classes # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2) # lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) # fully connected 
        self.fc_2 = nn.Linear(128, num_classes) # fully connected last layer
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # hidden state
        # h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        h_0 = torch.tensor(torch.zeros(self.num_layers, x.size(0), self.hidden_size), requires_grad=True)
        # cell state
        c_0 = torch.tensor(torch.zeros(self.num_layers, x.size(0), self.hidden_size), requires_grad=True)
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x.double(), (h_0.double(), c_0.double())) # (input, hidden, and internal state)
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) # first dense
        out = self.relu(out) # relu
        out = self.fc_2(out) # final output
        return out

def training_loop(n_epochs, lstm, optimiser, loss_fn, X_train, y_train,
                  X_test, y_test):
    for epoch in range(n_epochs):
        lstm.train()
        outputs = lstm.forward(X_train) # forward pass
        optimiser.zero_grad() # calculate the gradient, manually setting to 0
        # obtain the loss function
        loss = loss_fn(outputs, y_train)
        loss.backward() # calculates the loss of the loss function
        optimiser.step() # improve from loss, i.e backprop
        # test loss
        lstm.eval()
        test_preds = lstm(X_test)
        test_loss = loss_fn(test_preds, y_test)
        if epoch % 24 == 0:
            print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch, 
                                                                      loss.item(), 
                                                                      test_loss.item())) 