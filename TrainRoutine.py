"""
Utiltiy class to train AE model for reconstructing some times series data
given some training, validation and test data
"""

import random

import torch
import pandas as pd 
import numpy as np

from util.CmdDataset import CmdDataset
from util.lstmae import RecurrentAutoencoder

class AutoEncTrainRoutine:
    def __init__(self, seq_len=10, n_feat=1, emb_dim=32):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RecurrentAutoencoder(seq_len, n_feat, self.device, emb_dim)
        
        print("Sequence Length:", seq_len)
        print("Number of features:", n_feat)
        print("------------------------------")
        print("Initialising Autoencoder with:")
        print(self.model)
        print(f"Training on {self.device}")
        print("------------------------------")
        
    def train_model(self, train_x, train_y, val_x, val_y, n_epochs=40, lr=1e-2, batch_size=1):
        train_ds = CmdDataset(train_x, train_y)
        val_ds = CmdDataset(val_x, val_y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.L1Loss(reduction='sum').to(self.device)
        history = dict(train=[], val=[])

        for epoch in range(1, n_epochs + 1):
            model = self.model.train()

            train_losses = []
            val_losses = []

            train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            val_dl = torch.utils.data.DataLoader(val_ds, shuffle=False)

            size = len(train_dl.dataset)
            for batch, (X,y) in enumerate(train_dl):
                # Compute prediction and loss
                X = X.to(self.device)
                pred = model(X)
                loss = criterion(pred, X)
                train_losses.append(loss.item())

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if batch % 10000 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            with torch.no_grad():  # requesting pytorch to record any gradient for this block of code
                for (seq_true, y) in val_dl:
                    seq_true = seq_true.to(self.device)   # putting sequence to gpu
                    seq_pred = model(seq_true)    # prediction

                    loss = criterion(seq_pred, seq_true)  # recording loss

                    val_losses.append(loss.item())    # storing loss into the validation losses

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            history['train'].append(train_loss)
            history['val'].append(val_loss)

            print(f'Epoch {epoch}: train loss = {train_loss}, val loss = {val_loss}')

        return model.eval(), history
    
    def train_model_local(self, model, opt, criterion, train_dl, val_dl, history, n_local_epochs=5):
       
        for epoch in range(1, n_local_epochs + 1):

            model = model.train()
            
            train_losses = []
            val_losses = []
            weights = []

            size = len(train_dl.dataset)
            for batch, (X,y) in enumerate(train_dl):
                # Compute prediction and loss
                X = X.to(self.device)
                pred = model(X)
                loss = criterion(pred, X)
                train_losses.append(loss.item())

                # Backpropagation
                loss.backward()
                opt.step()
                opt.zero_grad()

                if batch % 10000 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            with torch.no_grad():  # requesting pytorch to record any gradient for this block of code
                for (seq_true, y) in val_dl:
                    seq_true = seq_true.to(self.device)   # putting sequence to gpu
                    seq_pred = model(seq_true)    # prediction

                    loss = criterion(seq_pred, seq_true)  # recording loss

                    val_losses.append(loss.item())    # storing loss into the validation losses

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            history['train'].append(train_loss)
            history['val'].append(val_loss)

            params = model.named_parameters()

            for name, param in params:
                weights.append(param.detach().numpy())

            print(f'Epoch {epoch}: train loss = {train_loss}, val loss = {val_loss}')

        return history, weights
    
    
    
    
    def save_model(self, name):
        torch.save(self.model.state_dict(), f"models/{name}")
        print(f"saving AE model from models/{name}")
    
    def load_model(self, name):
        self.model.load_state_dict(torch.load(f"models/{name}", 
                                              map_location=torch.device('cpu')))
        print(f"loading AE model from models/{name}")
  
    def encode_train_data(self, train_x, train_y, fname="data/normal_training_encoded.csv"):
        train_ds = CmdDataset(train_x, train_y)
        encoded = []
        train_dl = torch.utils.data.DataLoader(train_ds, shuffle=False)
        for (X,y) in train_dl:
            self.model.eval()
            with torch.no_grad():
                X = X.to(self.device)
                encoded.append(self.model.encoder(X).squeeze().cpu().numpy())
        enc_df = pd.DataFrame(encoded)
        enc_df.to_csv(fname)
        print(f"saving encoded training data in {fname}")
        return enc_df.astype(np.float32).to_numpy()

    def decode_data(self, encoded: pd.DataFrame):
        encoded_tensor = torch.Tensor(encoded).unsqueeze(0)
        self.model.to(self.device)
        decoded = []
        for (X) in encoded_tensor:
            self.model.eval()
            with torch.no_grad():               
                X = X.to(self.device)
                decoded.append(self.model.decoder(X.to(self.device)).squeeze(dim=1).cpu().numpy())

        return decoded
        


if __name__=="__main__":
    vanilla_ae = AutoEncTrainRoutine()
    vanilla_ae.load_model("lstmae_180_embed32.pth")
    vanilla_ae.train_model()