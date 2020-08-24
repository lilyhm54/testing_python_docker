import pandas as pd


def remove_columns(df, columns):
    """
    remove unwanted columns
    """
    cols = [x for x in list(df.columns) if x not in columns]
    return df[cols]


def check_for_nan(df):
    """
    Checking df for any rows with NaN
    """
    return df.dropna()


def age_buckets(df, age_col):
    """
    Sort age column into age buckets
    """
    df['AgeBucket'] = pd.qcut(df[age_col], q=8)
    return df


def fare_buckets(df, fare_col):
    """
    Sort fares into buckets
    """
    df['FareBucket'] = pd.qcut(df[fare_col], q=4)
    return df


def hot_encoding(df, hot_enc_columns):
    """
    Hot encoding specfic columns
    """
    cols = [x for x in list(df.columns) if x in hot_enc_columns]

    for x in cols:
        dummies = pd.get_dummies(df[x], x)
        df = pd.concat([df.drop(x, axis=1), dummies], axis=1)
    return df
