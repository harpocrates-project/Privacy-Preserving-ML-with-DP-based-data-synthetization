"""
Utility class to train DP-MERF generator
"""

import torch
from torch import nn
from torch import optim
import pandas as pd
import numpy as np
from autodp import privacy_calibrator


#taken from DPMERF
class Generative_Model_homogeneous_data(nn.Module):

        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dataset):
            super(Generative_Model_homogeneous_data, self).__init__()

            self.input_size = input_size
            self.hidden_size_1 = hidden_size_1
            self.hidden_size_2 = hidden_size_2
            self.output_size = output_size

            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
            self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
            self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
            self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)

            self.dataset = dataset


        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(self.bn1(hidden))
            output = self.fc2(relu)
            output = self.relu(self.bn2(output))
            output = self.fc3(output)

            # if self.dataset=='credit':
            #     all_pos = self.relu(output[:,-1])
            #     output = torch.cat((output[:,:-1], all_pos[:,None]),1)

            return output

class DPMERFGenerator():
    def __init__(self, is_priv=False, 
                 n_feat=2000, 
                 input_size=10,
                 seq_len=10) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_priv = is_priv
        self.n_feat = n_feat
        self.model = None
        self.input_size = input_size
        self.seq_len = seq_len
    
    # taken from DPMERF
    def dist_matrix(self, X, Y):
        """
        Construct a pairwise Euclidean distance matrix of size X.shape[0] x Y.shape[0]
        """
        sx = np.sum(X**2, 1)
        sy = np.sum(Y**2, 1)
        D2 =  sx[:, np.newaxis] - 2.0*X.dot(Y.T) + sy[np.newaxis, :] 
        # to prevent numerical errors from taking sqrt of negative numbers
        D2[D2 < 0] = 0
        D = np.sqrt(D2)
        return D
    
    # taken from DPMERF
    def meddistance(self, X, subsample=None, mean_on_fail=True):
        """
        Compute the median of pairwise distances (not distance squared) of points
        in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.

        Parameters
        ----------
        X : n x d numpy array
        mean_on_fail: True/False. If True, use the mean when the median distance is 0.
            This can happen especially, when the data are discrete e.g., 0/1, and 
            there are more slightly more 0 than 1. In this case, the m

        Return
        ------
        median distance
        """
        if subsample is None:
            D = self.dist_matrix(X, X)
            Itri = np.tril_indices(D.shape[0], -1)
            Tri = D[Itri]
            med = np.median(Tri)
            if med <= 0:
                # use the mean
                return np.mean(Tri)
            return med

        else:
            assert subsample > 0
            rand_state = np.random.get_state()
            np.random.seed(9827)
            n = X.shape[0]
            ind = np.random.choice(n, min(subsample, n), replace=False)
            np.random.set_state(rand_state)
            # recursion just one
            return self.meddistance(X[ind, :], None, mean_on_fail)
        
    # taken from DPMERF
    def RFF_Gauss(self, n_features, X, W):
        """ this is a Pytorch version of Wittawat's code for RFFKGauss"""

        W = torch.Tensor(W).to(self.device)
        X = X.to(self.device)

        XWT = torch.mm(X, torch.t(W)).to(self.device)
        Z1 = torch.cos(XWT)
        Z2 = torch.sin(XWT)

        Z = torch.cat((Z1, Z2),1) * torch.sqrt(2.0/torch.Tensor([n_features])).to(self.device)
        return Z
    
    # taken from DPMERF with modifications
    def train_generator(self, data, mini_batch_size=0.1, n_epochs=2000, lr=1e-2, eps=1, delt=1e-5):
        n, input_dim = data.shape
        # model specifics
        mini_batch_size = int(np.round(mini_batch_size * n))
        print("minibatch: ", mini_batch_size)
        input_size = self.input_size
        hidden_size_1 = 4 * input_dim
        hidden_size_2 = 2 * input_dim
        output_size = input_dim

        self.model = Generative_Model_homogeneous_data(input_size=input_size, hidden_size_1=hidden_size_1,
                                                          hidden_size_2=hidden_size_2,
                                                          output_size=output_size, dataset=data).to(self.device)


        

        

        # random Fourier features
        n_features = self.n_feat
        draws = n_features // 2

        # random fourier features for numerical inputs only
        num_data_pt_to_discard = 10
        idx_rp = np.random.permutation(n)
        idx_to_discard = idx_rp[0:num_data_pt_to_discard]
        idx_to_keep = idx_rp[num_data_pt_to_discard:]
        med = self.meddistance(data[idx_to_discard, ])
        
        data = data[idx_to_keep,:]
        n = data.shape[0]
        print('total number of datapoints in the training data is', n)
        sigma2 = med ** 2
        W_freq = np.random.randn(draws, input_dim) / np.sqrt(sigma2)
        print(f"random freq {W_freq.shape}")


        ####################################################
        # Privatising quantities if necessary

        """ computing mean embedding of subsampled true data """

        emb1_input_features = self.RFF_Gauss(n_features, torch.Tensor(data), W_freq)
        outer_emb1 = torch.einsum('ki->ki', [emb1_input_features])
        mean_emb1 = torch.mean(outer_emb1, 0)



        """ privatizing each column of mean embedding """
        if self.is_priv:
            print("adding DP noise")
            # desired privacy level
            epsilon = eps
            delta = delt
            privacy_param = privacy_calibrator.gaussian_mech(epsilon, delta)
            sensitivity = 2 / n
            noise_std_for_privacy = privacy_param['sigma'] * sensitivity

            # make sure add noise after rescaling


 
            noise = noise_std_for_privacy * torch.randn(mean_emb1.size())
            noise = noise.to(self.device)

            mean_emb1 = mean_emb1 + noise

        # define details for training
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        how_many_epochs = n_epochs
        how_many_iter = int(n / mini_batch_size)
        training_loss_per_epoch = np.zeros(how_many_epochs)
        print('Starting Training')

        for epoch in range(how_many_epochs):  # loop over the dataset multiple times

            running_loss = 0.0

            for i in range(how_many_iter):

                """ computing mean embedding of generated data """
                # zero the parameter gradients
                optimizer.zero_grad()


                feature_input = torch.randn((mini_batch_size, input_size)).to(self.device)
                input_to_model = feature_input

                #we feed noise + label (like in cond-gan) as input
                outputs = self.model(input_to_model)

                """ computing mean embedding of generated samples """
                emb2_input_features = self.RFF_Gauss(n_features, outputs, W_freq)



                outer_emb2 = torch.einsum('ki->ki', [emb2_input_features])
                mean_emb2 = torch.mean(outer_emb2, 0)


                loss = torch.norm(mean_emb1 - mean_emb2, p=2) ** 2

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if epoch % 100 == 0:
                print('epoch # and running loss are ', [epoch, running_loss])
                training_loss_per_epoch[epoch] = running_loss

    def generate(self, n, fname="enc_generated_private.csv"):
        feature_input = torch.randn((n, self.input_size)).to(self.device)
        input_to_model = feature_input
        outputs = self.model(input_to_model)

        samp_input_features = outputs

        generated_input_features_final = samp_input_features.cpu().detach().numpy()
        pd.DataFrame(generated_input_features_final).to_csv(f"data/generated/"+fname)
        print(f"generating and saving latent data in data/generated/{fname}")
        return generated_input_features_final
        

