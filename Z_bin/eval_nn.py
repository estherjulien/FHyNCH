import pandas as pd
import numpy as np
import joblib
import torch
import pickle

from dm import get_data


def get_torch_data(x_pd):
    return torch.Tensor(x_pd.to_numpy()).float()


def get_nn_prediction(net, data):
    with torch.no_grad():
        x = get_torch_data(data)
        y = net(x).detach().cpu().numpy().squeeze()
    return pd.Series(y, index=data.index)


if __name__ == "__main__":
    df_res = pd.DataFrame(columns=["val_acc", "corr_perc", 5, 10, 25, 50, 75, 90, 95])

    # LEAF
    with open("../Snellius/train/ML_data_leaf_n2000.pkl", "rb") as handle:
        x, y = pickle.load(handle)
    x = x.loc[y==1]

    # get model
    leaf_model_name = "../data/rf_models/leaf/rf_leaves_2000n.joblib"
    model_leaf = joblib.load(leaf_model_name)
    preds = model_leaf.predict_proba(x)[:, 1]
    corr_perc = (preds >= .5).sum()/len(preds)

    meta = pd.read_csv(f"../data/rf_models/metadata_rf_2000n.txt")
    meta.index = meta[meta.columns[0]]
    meta_leaf = meta[meta.columns[1]]
    df_res.loc["RF LEAF"] = [meta_leaf.acc_all_leaf, corr_perc, *[np.quantile(preds, q) for q in
                                             [.05, .1, .25, .5, .75, .9, .95]]]

    # test NN results
    leaf_model_name = f"../data/nn_models/leaf/nn_model_leaf_n2000.pt"
    # Loading model
    model_leaf = torch.jit.load(leaf_model_name)
    model_leaf.eval()

    preds = get_nn_prediction(model_leaf, x)
    corr_perc = (preds >= .5).sum()/len(preds)

    meta = pd.read_csv(f"../data/nn_models/leaf/metadata_nn_leaf_n2000.txt")
    meta.index = meta[meta.columns[0]]
    meta = meta[meta.columns[1]]
    df_res.loc["NN LEAF"] = [meta.acc_all, corr_perc, *[np.quantile(preds, q) for q in
                                                               [.05, .1, .25, .5, .75, .9, .95]]]

    # CHERRY
    with open("../Snellius/train/ML_data_cherry_n2000.pkl", "rb") as handle:
        x, y = pickle.load(handle)
    x = x.loc[y==1]

    # get model
    model_name = "../data/rf_models/cherry/rf_cherries_2000n.joblib"
    model = joblib.load(model_name)
    preds = model.predict_proba(x)[:, 1]
    corr_perc = (preds >= .5).sum()/len(preds)
    df_res.loc["RF CHERRY"] = [meta_leaf.acc_all_cherry, corr_perc, *[np.quantile(preds, q) for q in
                                             [.05, .1, .25, .5, .75, .9, .95]]]

    # test NN results
    model_name = f"../data/nn_models/cherry/nn_model_cherry_n2000.pt"
    # Loading model
    model = torch.jit.load(model_name)
    model.eval()

    preds = get_nn_prediction(model, x)
    corr_perc = (preds >= .5).sum()/len(preds)

    meta = pd.read_csv(f"../data/nn_models/cherry/metadata_nn_cherry_n2000.txt")
    meta.index = meta[meta.columns[0]]
    meta = meta[meta.columns[1]]
    df_res.loc["NN CHERRY"] = [meta.acc_all, corr_perc, *[np.quantile(preds, q) for q in
                                                               [.05, .1, .25, .5, .75, .9, .95]]]