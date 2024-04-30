import pandas as pd


def generate_train_test_df(full_df):
    # Get train and test sets
    train = full_df[full_df["YEAR"] < 2023]
    X_train = train.drop(columns=["N_product_purchased", "P_date"])
    y_train = train["N_product_purchased"]

    test = full_df[full_df["YEAR"] == 2023]
    X_test = test.drop(columns=["N_product_purchased", "P_date"])
    y_test = test["N_product_purchased"]

    return train, X_train, y_train, test, X_test, y_test
