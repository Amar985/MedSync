import numpy as np
import pandas as pd


# dictionary P_code_NUM to Cost
def get_product_price_dict(df):
    return (
        df.groupby("P_code_NUM")["Cost"].max()
        / df.groupby("P_code_NUM")["N_units"].max()
    ).to_dict()


def smape_score(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
