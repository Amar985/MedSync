import pandas as pd


def preprocessing(df):
    # Separate code into two columns
    new_columns = df["P_code"].str.extract(r"([a-zA-Z]+)([0-9]+)", expand=False)
    df["P_code_CLASS"] = new_columns[0]
    df["P_code_NUM"] = new_columns[1]
    df.drop(columns=["P_code"], inplace=True)

    # P_date to datetime in day/month/year format
    df["P_date"] = pd.to_datetime(df["P_date"], dayfirst=True)
    df.sort_values(by=["P_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # separate H_code in three columns by '-'
    origin_separated_columns = df["H_code"].str.split("-", expand=True)
    df["PURCHASING_HOSPITAL"] = origin_separated_columns[1]
    df["PURCHASING_DEPARTMENT"] = origin_separated_columns[2]
    df.drop(columns=["H_code"], inplace=True)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df
