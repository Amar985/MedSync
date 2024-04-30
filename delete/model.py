from xgboost import XGBRegressor
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot_model_predictions(full_df, product, hospital, plot_train=False):
    columns = ["P_date", "N_product_purchased"]
    partial_df = full_df.loc[
        (full_df["PURCHASING_HOSPITAL"] == hospital)
        & (full_df["P_code_NUM"] == product)
    ][columns]
    partial_df = generate_date_features(partial_df)
    partial_df = add_timeseries_features(partial_df)
    train, X_train, y_train, test, X_test, y_test = generate_train_test_df(partial_df)

    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)

    X_train["PREDICTION"] = model.predict(X_train)
    X_train["REAL"] = y_train
    X_train["P_date"] = train["P_date"]

    X_test["PREDICTION"] = model.predict(X_test)
    X_test["REAL"] = y_test
    X_test["P_date"] = test["P_date"]

    plt.figure(figsize=(30, 10))
    if plot_train:
        sns.lineplot(
            x="P_date",
            y="PREDICTION",
            data=X_train,
            marker="o",
            label="train preds",
        )
        sns.lineplot(
            x="P_date", y="REAL", data=X_train, marker="o", label="train real"
        )

    sns.lineplot(
        x="P_date", y="PREDICTION", data=X_test, marker="o", label="test preds"
    )
    sns.lineplot(x="P_date", y="REAL", data=X_test, marker="o", label="test real")

    plt.legend()
    plt.show()
    pass
def generate_date_features(df):
    df["YEAR"] = df["P_date"].dt.year
    df["MONTH"] = np.sin(2 * np.pi * df["P_date"].dt.month / 12)
    df["DAYOFMONTH"] = np.sin(2 * np.pi * df["P_date"].dt.day / 31)
    df["DAYOFYEAR"] = np.sin(2 * np.pi * df["P_date"].dt.dayofyear / 365)
    return df
def add_timeseries_features(df):
    df["ROLLING_MEAN_3M"] = df["N_product_purchased"].rolling(90).mean()
    df["WEIGHTED_MEAN_3M"] = (
        df["N_product_purchased"]
        .rolling(90)
        .apply(lambda x: np.average(x, weights=range(1, len(x) + 1)))
    )
    df["EWMA_3M"] = df["N_product_purchased"].ewm(span=90).mean()
    df["ROLLING_MEAN_1Y"] = df["N_product_purchased"].rolling(365).mean()
    df["WEIGHTED_MEAN_1Y"] = (
        df["N_product_purchased"]
        .rolling(365)
        .apply(lambda x: np.average(x, weights=range(1, len(x) + 1)))
    )
    df["EWMA_1Y"] = df["N_product_purchased"].ewm(span=365).mean()
    # average N_product_purchased over year
    df["AVG_1Y"] = df.groupby(["YEAR"])["N_product_purchased"].transform("mean")
    df["AVG_1Y"] = df["AVG_1Y"].fillna(df["AVG_1Y"].mean())
    return df
def generate_train_test_df(full_df):
    # Get train and test sets
    train = full_df[full_df["YEAR"] < 2023]
    X_train = train.drop(columns=["N_product_purchased", "P_date"])
    y_train = train["N_product_purchased"]

    test = full_df[full_df["YEAR"] == 2023]
    X_test = test.drop(columns=["N_product_purchased", "P_date"])
    y_test = test["N_product_purchased"]

    return train, X_train, y_train, test, X_test, y_test
