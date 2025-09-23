import pandas as pd
import argparse
import os
import time

import plotly.express as px
import plotly.graph_objects as go
from pylab import *
from sklearn.model_selection import train_test_split
from util.aedpmerf import AEDPMERF
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score
from collections import defaultdict




def reading_files(path):

    """
    Read csv files, drop 'Unnamed: 0' column and set 'UtcTime' as index.

    Args:
        path (str): Preprocessed train-test input files.

    Returns:
        df (pandas.DataFrame): Preprocessed DataFrame.
    """

    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    if 'Attack' in df.columns:
        df.Attack = df.Attack.astype(str)
    df=df.reset_index(drop=True)

    return df


def reshaping_data(X, timesteps, test):

    """
    Reshapes the input data into a suitable format for LSTM models.

    Args:
        X (ndarray): Input data array of shape (n_samples, features).
        timesteps (int): Number of time steps or sequence length for the reshaped data.
        test (bool): Specify whether to reshape the tet or training dataset. If test=True it will be reshaped, otherwise train.

    Returns:

        ndarray: Reshaped input data array of shape (n_samples - timesteps + 1, timesteps, features).

    """

    if test:
        Xs = pd.DataFrame()
        for i in X['ProcessGuid'].unique():
            sorteddf=X[X.ProcessGuid == i].sort_values(by="UtcTime")
            temp_col = sorteddf["UtcTime"]
            Attack = sorteddf["Attack"]
            matrix_temporal = sorteddf.drop(["ProcessGuid","UtcTime","Attack","collector_node_id"], axis=1).values # "collector-node-id"

            # Define padding values and amount
            padding_value = -1  # Change this to the value you want for padding
            padding_rows = timesteps - 1  # Number of rows to add as padding

            # Pad the matrix along the rows
            padded_matrix = np.pad(matrix_temporal, ((padding_rows, 0), (0, 0)), mode='constant', constant_values=padding_value)

            for j,z1,z2 in zip(range(len(matrix_temporal) - timesteps + 1),temp_col,Attack): # Ensures that extracted substrings have uniform length of timesteps and do not go outside the original sequence boundary. Avoid extracting incomplete substring
                data = {'Value': [padded_matrix[j:(j + timesteps)]],
                        'ProcessGuid': [i],
                        'UtcTime':z1,
                        "numVentana":j,
                        "Attack":z2}
                df = pd.DataFrame(data)
                Xs=pd.concat([Xs,df])
        return Xs
    else:
        Xs = []
        for i in X.ProcessGuid.unique():
            matrix_temporal = X[X.ProcessGuid == i].sort_values(by="UtcTime").drop(["ProcessGuid","UtcTime"], axis=1).values

            # Define padding values and amount
            padding_value = -1  # Change this to the value you want for padding
            padding_rows = timesteps - 1  # Number of rows to add as padding

            # Pad the matrix along the rows
            padded_matrix = np.pad(matrix_temporal, ((padding_rows, 0), (0, 0)), mode='constant', constant_values=padding_value)
            for j in range(len(matrix_temporal) - timesteps + 1): # Ensures that extracted substrings have uniform length of timesteps and do not go outside the original sequence boundary. Avoid extracting incomplete substring
                Xs.append(padded_matrix[j:(j + timesteps)])


        return np.array(Xs)



def timesteps_calculation(df, timesteps_max):

    timesteps_calc = int(pd.Series(df.ProcessGuid).value_counts().quantile(0.5))
    if timesteps_calc >= timesteps_max:
        timesteps = timesteps_max
    else:
        timesteps = timesteps_calc

    print("timesteps:", timesteps)

    return timesteps

class Client:
    def __init__(self,train_x, train_y, x, val_x, val_y, ts, idx):
        self.train_x = train_x #Training data
        self.train_y = train_y #Labels
        self.orig_x = x
        self.val_x = val_x #validation set x
        self.val_y = val_y #validation set y
        self.timestep = ts
        self.train_x_gen = []
        self.mix_x = []
        self.p = 1
        self.n = train_x.shape[0] #Number of training points provided
        self.id = idx #ID for the user
        self.model = None
        self.opt = None
        self.criterion = []
        self.threshold = 0


def create_client(X, timesteps, id):

    X_train_y = np.zeros(X.shape[0])

    train_x, val_x, train_y, val_y = train_test_split(X, X_train_y, test_size=0.10, shuffle=False)

    client = Client(train_x, train_y, X, val_x, val_y, timesteps, id)

    print(train_x.shape)
    print(val_x.shape)
    print("--------------------------")

    return client



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="sim or diff")
    parser.add_argument('--process', type=str, required=True, help="Process to train")
    parser.add_argument('--attack', type=str, required=True, help="the test attack")
    parser.add_argument('--testorg', type=str, required=True, help="the test organization")
    args = parser.parse_args()


    base_dir = os.path.join(os.getcwd(), "Database")
    # Input data foler
    input_training_data_folder_path_c1= os.path.join(base_dir, f"s1_{args.data}", "training", args.process, "data")
    input_training_data_folder_path_c2= os.path.join(base_dir, f"s2_{args.data}", "training", args.process, "data")
    input_training_data_folder_path_c3= os.path.join(base_dir, f"v1_{args.data}", "training", args.process, "data")
    input_training_data_folder_path_c4= os.path.join(base_dir, f"v2_{args.data}", "training", args.process, "data")
    input_test_data_folder_path = os.path.join(base_dir, f"{args.testorg}_{args.data}_test", args.attack, args.process, "data" )
    experiment_name =args.attack+args.process+args.testorg+args.data

    # Maximum sequence length to be used by the model in training for each batch. The final timesteps parameter will be computed
    # later depending on the input data length
    timesteps_max = 1


    X_train_c1 = reading_files(os.path.join(input_training_data_folder_path_c1,"X_train.csv"))
    X_train_c2 = reading_files(os.path.join(input_training_data_folder_path_c2,"X_train.csv"))
    X_train_c3 = reading_files(os.path.join(input_training_data_folder_path_c3,"X_train.csv"))
    X_train_c4 = reading_files(os.path.join(input_training_data_folder_path_c4,"X_train.csv"))
    X_test = reading_files(os.path.join(input_test_data_folder_path,"X_test.csv"))
    X_test = X_test.drop(["index"], axis=1)


    # Create a new DataFrame with those columns
    X_test_label = X_test[["ProcessGuid","UtcTime","Attack","collector_node_id"]].copy()

    # Remove those columns from the original DataFrame
    X_test = X_test.drop(columns=["ProcessGuid","UtcTime","Attack","collector_node_id"])

     # Get the union of all the columns from all datasets (The master set of all unique column names across all nodes)
    all_columns = set(X_train_c1.columns) | set(X_train_c2.columns) | set(X_train_c3.columns) | set(X_train_c4.columns) | set(X_test.columns)

    #Add any missing columns and fill them with 0
    dfs = [X_train_c1, X_train_c2, X_train_c3, X_train_c4, X_test]
    dfs = [df.reindex(columns=all_columns, fill_value=0) for df in dfs]

    # Node identity Column names to add
    new_cols = ["node1", "node2", "node3", "node4", "test"]

    # Add node identity columns
    for i, df in enumerate(dfs):
        # First fill all new columns with 0
        for col in new_cols:
            df[col] = 0
        # Set the "node" column for this dataframe to 1
        df[new_cols[i]] = 1


    X_train_c1 = dfs[0]
    X_train_c2 = dfs[1]
    X_train_c3 = dfs[2]
    X_train_c4 = dfs[3]
    X_test = dfs[4]

    # The test set will get these two columns from the other training data (all zeros), we remove from here
    X_test = X_test.drop(["ProcessGuid", "UtcTime"], axis=1)

    # Put back the label for the test set
    X_test = pd.concat([X_test, X_test_label],axis=1)


    # Final timesteps depending of the X_train length. This is necessary because the length of the input data sequence is variable,
    # each process has a certain number of rows

    timesteps_c1 = timesteps_calculation(X_train_c1, timesteps_max)
    timesteps_c2 = timesteps_calculation(X_train_c2, timesteps_max)
    timesteps_c3 = timesteps_calculation(X_train_c3, timesteps_max)
    timesteps_c4 = timesteps_calculation(X_train_c4, timesteps_max)

    X_train_c1 = reshaping_data(X_train_c1, timesteps=timesteps_c1,test=False)
    X_train_c2 = reshaping_data(X_train_c2, timesteps=timesteps_c2,test=False)
    X_train_c3 = reshaping_data(X_train_c3, timesteps=timesteps_c3,test=False)
    X_train_c4 = reshaping_data(X_train_c4, timesteps=timesteps_c4,test=False)
    X_test_reshaping = reshaping_data(X_test, timesteps=timesteps_c1,test=True)

    c1 = create_client(X_train_c1, timesteps_c1, 0)
    c2 = create_client(X_train_c2, timesteps_c2, 1)
    c3 = create_client(X_train_c3, timesteps_c3, 2)
    c4 = create_client(X_train_c4, timesteps_c4, 3)

    clients = [c1 , c2, c3, c4]

    emb_dim = 16
    eps = 0.5
    lr = 1e-5
    n_epochs = 2000
    models_dir = os.path.join(os.getcwd(), "models", "FL")

    for c in clients:
        model_name = f"synthetizer_1_embed{emb_dim}_client{c.id}_{experiment_name}_{eps}.pth"
        aedpmerf = AEDPMERF(seq_len=c.timestep, n_feat=c.train_x.shape[2], emb_dim=emb_dim, is_priv=True)
        synethezier, history = aedpmerf.ae.train_model(c.train_x, c.train_y, c.val_x, c.val_y, n_epochs=30, lr=1e-3, batch_size=1)
        enc_df = aedpmerf.encode_train_data(c.train_x, c.train_y, fname=f"data/normal_encoded_embed{emb_dim}_eps{eps}__client{c.id}_{experiment_name}.csv")
        aedpmerf.train_gen(data=enc_df, mini_batch_size=0.1, lr=lr, eps=eps, n_epochs=n_epochs)
        n_gen_samples = enc_df.shape[0]
        gen_data = aedpmerf.generate(n_gen_samples, fname=f"enc_gen_priv_embed{emb_dim}_eps{eps}_client{c.id}_{experiment_name}.csv")
        gen_data = np.array(gen_data)
        gen_data = np.transpose(gen_data, (1, 0, 2))
        c.train_x_gen = gen_data
        c.mix_x = np.concatenate((c.orig_x, c.train_x_gen), axis=0)
