"""
AE-dpMERF class
"""

from util.DPMERFGenerator import DPMERFGenerator
from util.TrainRoutine import AutoEncTrainRoutine

class AEDPMERF:
    def __init__(self, seq_len=10, n_feat=1, emb_dim=12,
                 is_priv=False, 
                 n_feat_rff=2000, 
                 input_size=10):
        """
        initialises the AE-dpMERF generator with 
            sequence length (seq_len), 
            dimension of input (n_feat),
            embedding dimension for the autoencoder (emb_dim),
            number of random fourier features for DP-MERF (n_feat_rff),
            the dimension of random noise for the generator (input_size)
        """
        self.ae = AutoEncTrainRoutine(seq_len=seq_len, n_feat=n_feat, emb_dim=emb_dim)

        self.dpmerfgen = DPMERFGenerator(is_priv=is_priv, 
                 n_feat=n_feat_rff, 
                 input_size=input_size,
                 seq_len=seq_len)
        
    def save_ae(self, name):
        """
        saves the autoencoder weights under models/{name}
        """
        self.ae.save_model(name)
    
    def load_ae(self, name):
        """
        loads the autoencoder weights from models/{name}
        """
        self.ae.load_model(name)
        return self.ae.model
    
    def train_ae(self, train_ds_path, val_ds_path, n_epochs=20, lr=5e-4, batch_size=1):
        """
        trains the autoencoder 
        input:
            train_ds_path   path to the training data (will be converted to ECGDataset class) #TODO: make generic dataset class
            val_ds path     path to the training data #TODO: same problem as training data
            n_epochs        number of epochs
            lr              learning rate
            batch_size      batch size
        """
        return self.ae.train_model(train_ds_path=train_ds_path, val_ds_path=val_ds_path, n_epochs=n_epochs, lr=lr, batch_size=batch_size)
    
    def encode_train_data(self, train_x, train_y, fname):
        """
        encoded the training times series data to the latent representation
        and save it under {fname}
        """
        return self.ae.encode_train_data(train_x, train_y, fname=fname)
    
    def train_gen(self, data, mini_batch_size=0.1, n_epochs=2000, lr=1e-2, eps=1, delt=1e-5):
        """
        trains the DP-MERF generator and ignores the eps and delt values if is_priv=False
        """
        self.dpmerfgen.train_generator(data=data, mini_batch_size=mini_batch_size, n_epochs=n_epochs, lr=lr, eps=eps, delt=delt)

    def generate(self, n_gen_samples, fname):
        """
        generate some latent data and saves it under data/{fname} and decodes it
        #TODO: save the decoded data as well
        """
        gen_enc = self.dpmerfgen.generate(n_gen_samples, fname=fname)
        gen_dec = self.ae.decode_data(gen_enc)
        return gen_dec