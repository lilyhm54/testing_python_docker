# main file for running test data analysis 
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf

def remove_columns(df, columns = ['Cabin', 'Name', 'Ticket']):
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

def age_buckets(df):
    """
    Sort age column into age buckets
    """
    age_col = 'Age'
    df.loc[df[age_col] <= 17, 'AgeBuckets'] = '0_17'
    df.loc[(df[age_col] > 17) & (df[age_col] <= 24), 'AgeBuckets'] = '18_24'
    df.loc[(df[age_col] >= 25) & (df[age_col] <= 34), 'AgeBuckets'] = '25_34'
    df.loc[(df[age_col] >= 35) & (df[age_col] <= 44), 'AgeBuckets'] = '35_44'
    df.loc[(df[age_col] >= 45) & (df[age_col] <= 54), 'AgeBuckets'] = '45_54'
    df.loc[(df[age_col] >= 55) & (df[age_col] <= 64), 'AgeBuckets'] = '55_64'
    df.loc[(df[age_col] >= 65) & (df[age_col] <= 74), 'AgeBuckets'] = '65_74'
    df.loc[(df[age_col] >= 75), 'AgeBuckets'] = '75_plus'
    return df 

def fare_buckets(df):
    """
    Sort fares into buckets
    """
    fare_col = 'Fare'
    df.loc[df[fare_col] <= 9, 'FareBucket'] = '0_9'
    df.loc[(df[fare_col] > 10) & (df[fare_col] <= 16), 'FareBucket'] = '10_16'
    df.loc[(df[fare_col] >= 17) & (df[fare_col] <= 33), 'FareBucket'] = '17_33'
    df.loc[(df[fare_col] >= 34), 'FareBucket'] = '34_plus'
    return df

def hot_encoding(df):
    """
    Hot encoding spefic columns 
    """
    cols = [x for x in list(df.columns) if x in ['Sex', 'Embarked', 'AgeBuckets', 'FareBucket']]
    
    for x in cols: 
        dummies = pd.get_dummies(df[x], x)
        df = pd.concat([df.drop(x, axis=1), dummies], axis=1)
    return df

def main():

    # load data 
    df = pd.read_csv('/Users/lilymartin/DEVOPS/testing_python_docker/app/data/test.csv')

    # remove uneeded columns: Name, tikcet, Cabin
    df = remove_columns(df, ['Cabin', 'Name', 'Ticket', 'PassengerId'])

    # remove rows with NaN 
    df = check_for_nan(df)

    # Bucket groups: Age, Fare 
    df = age_buckets(df)
    df = fare_buckets(df)

    # encode categorical variable: Sex, Embarked 
    df = hot_encoding(df)

    # remove bucketed columns
    df = remove_columns(df, ['Age', 'Fare'])

    # df to numpy array
    X = np.array(df)
    
    # Run through model 
    reconstructed_model = tf.keras.models.load_model('/Users/lilymartin/DEVOPS/testing_python_docker/app/model/20200823_titantic_m1.h5')
    predicted = reconstructed_model.predict_classes(X)

    # Return df with predicted class 
    df['predicted'] = predicted
    print(df.head())

    return df

if __name__ == '__main__':
    main()
