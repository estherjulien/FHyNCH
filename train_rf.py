from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import argparse
import joblib
import time
import os

from learning.dm import get_data


def train_cherry(X, Y, name=None, multiclass=False):
    os.makedirs("data/rf_info", exist_ok=True)
    if name is None:
        model_name = f"data/rf_cherries.joblib"
    else:
        model_name = f"data/rf_cherries_{name}.joblib"

    # split data in train and validation
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)

    # train model
    st_train = time.time()
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    train_dur = time.time() - st_train

    # evaluation
    if multiclass:
        score_rf = rf.score(X_val, Y_val)
        score_rf_no_cher = rf.score(X_val[Y_val[0] == 1], Y_val[Y_val[0] == 1])
        score_rf_cher = rf.score(X_val[Y_val[1] == 1], Y_val[Y_val[1] == 1])
        score_rf_ret_cher = rf.score(X_val[Y_val[2] == 1], Y_val[Y_val[2] == 1])
        score_rf_no_ret_cher = rf.score(X_val[Y_val[3] == 1], Y_val[Y_val[3] == 1])

        print(f"\nRF CHERRIES overall validation accuracy = {score_rf}")
        print(f"RF CHERRIES no cherry validation accuracy = {score_rf_no_cher}")
        print(f"RF CHERRIES cherry validation accuracy = {score_rf_cher}")
        print(f"RF CHERRIES ret cherry validation accuracy = {score_rf_ret_cher}")
        print(f"RF CHERRIES no ret cherry validation accuracy = {score_rf_no_ret_cher}")
        scores = [score_rf, score_rf_no_cher, score_rf_cher, score_rf_ret_cher, score_rf_no_ret_cher]
    else:
        score_rf = rf.score(X_val, Y_val)
        score_rf_not_pick = rf.score(X_val[Y_val == 0], Y_val[Y_val == 0])
        score_rf_pick = rf.score(X_val[Y_val == 1], Y_val[Y_val == 1])
        print(f"\nRF CHERRIES overall validation accuracy = {score_rf}")
        print(f"RF CHERRIES no cherry validation accuracy = {score_rf_not_pick}")
        print(f"RF CHERRIES cherry validation accuracy = {score_rf_pick}")
        scores = [score_rf, score_rf_not_pick, score_rf_pick]
    # feature importance
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Feature importance:\n")
    print(feature_importance)
    feature_importance.to_csv(f"data/rf_info/feat_imp_cherries_{name}.txt")

    # save
    joblib.dump(rf, model_name)

    return scores, train_dur


def train_leaf(X, Y, name=None):
    os.makedirs("data", exist_ok=True)
    if name is None:
        model_name = f"data/rf_leaves.joblib"
    else:
        model_name = f"data/rf_leaves_{name}.joblib"

    # split data in train and validation
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)

    # train model
    st_train = time.time()
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    train_dur = time.time() - st_train

    # evaluation
    score_rf = rf.score(X_val, Y_val)
    score_rf_not_pick = rf.score(X_val[Y_val == 0], Y_val[Y_val == 0])
    score_rf_pick = rf.score(X_val[Y_val == 1], Y_val[Y_val == 1])
    print(f"\nRF LEAVES overall validation accuracy = {score_rf}")
    print(f"RF LEAVES no cherry validation accuracy = {score_rf_not_pick}")
    print(f"RF LEAVES cherry validation accuracy = {score_rf_pick}")
    scores = [score_rf, score_rf_not_pick, score_rf_pick]

    # feature importance
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Feature importance:\n")
    print(feature_importance)
    feature_importance.to_csv(f"data/rf_info/feat_imp_leaves_{name}.txt")

    # save
    joblib.dump(rf, model_name)
    return scores, train_dur


def main(args):
    X_cherry, Y_cherry, num_nets = get_data(leaf_data=False, num_inst=args.n_tr_nets)
    name = f"{num_nets}n"
    acc_cherry, train_dur_cherry = train_cherry(X_cherry, Y_cherry, name=name)
    X_leaf, Y_leaf, _ = get_data(leaf_data=True)
    if "Leaf missing" in X_leaf:
        X_leaf.drop("Leaf missing", axis=1, inplace=True)
    acc_leaf, train_dur_leaf = train_leaf(X_leaf, Y_leaf, name=name)

    index = ["N", "num_data_cherry", "acc_all_cherry", "acc_no_pick", "acc_pick", "train_dur_cherry",
                 "num_data_leaf", "acc_all_leaf", "acc_no_pick", "acc_pick", "train_dur_leaf"]
    metadata = pd.Series([num_nets, len(X_cherry), *acc_cherry, train_dur_cherry,
                          len(X_leaf), *acc_leaf, train_dur_leaf],
                         index=index, dtype=float)
    metadata.to_csv(f"data/rf_info/metadata_rf_{name}.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tr_nets', type=int, default=-1)
    args = parser.parse_args()
    main(args)
