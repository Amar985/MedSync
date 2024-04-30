import numpy as np


# basic date features
def generate_date_features(df):
    df["YEAR"] = df["P_date"].dt.year
    df["MONTH"] = np.sin(2 * np.pi * df["P_date"].dt.month / 12)
    df["DAYOFMONTH"] = np.sin(2 * np.pi * df["P_date"].dt.day / 31)
    df["DAYOFYEAR"] = np.sin(2 * np.pi * df["P_date"].dt.dayofyear / 365)
    return df


def add_timeseries_features(df):
    # MEANS
    df["ROLLING_MEAN_3M"] = df["N_product_purchased"].rolling(90).mean()
    df["ROLLING_MEAN_1Y"] = df["N_product_purchased"].rolling(365).mean()

    # WEIGHTED MEANS
    df["WEIGHTED_MEAN_3M"] = (
        df["N_product_purchased"]
        .rolling(90)
        .apply(lambda x: np.average(x, weights=range(1, len(x) + 1)))
    )
    df["WEIGHTED_MEAN_1Y"] = (
        df["N_product_purchased"]
        .rolling(365)
        .apply(lambda x: np.average(x, weights=range(1, len(x) + 1)))
    )

    # EXPONENTIAL WEIGHTED MEANS
    df["EWMA_1W"] = df["N_product_purchased"].ewm(span=7).mean()
    df["EWMA_1M"] = df["N_product_purchased"].ewm(span=30).mean()
    df["EWMA_3M"] = df["N_product_purchased"].ewm(span=90).mean()
    df["EWMA_1Y"] = df["N_product_purchased"].ewm(span=365).mean()

    # LAGS
    df["SHIFT_1W"] = df["N_product_purchased"].shift(7)
    df["SHIFT_1M"] = df["N_product_purchased"].shift(30)
    df["SHIFT_3M"] = df["N_product_purchased"].shift(90)
    df["SHIFT_1Y"] = df["N_product_purchased"].shift(365)

    # DIFFS
    df["DIFF_1W"] = df["N_product_purchased"].diff(7)
    df["DIFF_1M"] = df["N_product_purchased"].diff(30)
    df["DIFF_3M"] = df["N_product_purchased"].diff(90)
    df["DIFF_1Y"] = df["N_product_purchased"].diff(365)

    return df
