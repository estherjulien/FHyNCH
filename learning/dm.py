import pandas as pd
import numpy as np
import pickle
import glob
import os


def get_data(leaf_data=False, num_inst=-1, scale=False, num_nets=None):
    data_dir = "data"
    # get data files
    if leaf_data:
        d_name = "leaf"
    else:
        d_name = "cherry"

    # get number of networks in data set
    if num_nets is None:
        num_nets = len(glob.glob("data/train/metadata/*"))
        if num_inst > 0:
            num_nets = min(num_nets, num_inst)

    # check if file already exists
    if os.path.exists(f"{data_dir}/train/ML_data_{d_name}_n{num_nets}.pkl"):
        with open(f"{data_dir}/train/ML_data_{d_name}_n{num_nets}.pkl", "rb") as handle:
            X, Y = pickle.load(handle)
        print("Already have data")
    else:
        # otherwise, make data set
        if num_inst > 0:
            files = []
            for i in range(1, num_inst+1):
                files += glob.glob(f"{data_dir}/train/inst_results/ML_data_{i}_r*")
        else:
            files = glob.glob(f"{data_dir}/train/inst_results/*")

        # get data
        output_list = []
        for fil in files:
            with open(fil, "rb") as handle:
                output_list.append(pickle.load(handle))

        # Clean data by deleting bad data points
        X = output_list[0][f"x_{d_name}"]
        Y = output_list[0][f"y_{d_name}"]
        for output in output_list[1:]:
            X = pd.concat([X, output[f"x_{d_name}"]], ignore_index=True)
            Y = pd.concat([Y, output[f"y_{d_name}"]], ignore_index=True)

        if not leaf_data:
            Y = Y[1] + Y[2]

        print(f"{d_name}: before cleaning num. datapoints = {len(X)}, distribution = \n{Y.describe()}")
        # data cleaning
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.dropna(inplace=True)
        Y = Y.loc[X.index]

        # balance
        X["class"] = Y

        # only two percent of cherries are reticulate (or opposite)
        g = X.groupby('class')
        if d_name == "leaf":
            g = pd.DataFrame(
                g.apply(lambda x: x.sample(int(g.size().mean()), replace=True).reset_index(drop=False)))
        else:
            g = pd.DataFrame(
                g.apply(lambda x: x.sample(int(g.size().mean()), replace=True).reset_index(drop=False)))
        X = g
        X.index = g["index"]
        X.drop(["index", "class"], axis=1, inplace=True)
        Y = Y.loc[X.index]
        print(f"{d_name}: after cleaning num. datapoints = {len(X)}, distribution = \n{Y.describe()}")

        # save data
        with open(f"{data_dir}/train/ML_data_{d_name}_n{num_nets}.pkl", "wb") as handle:
            pickle.dump((X, Y), handle)

    if scale:
        X = (X - X.min()) / (X.max() - X.min())
    return X, Y, num_nets
