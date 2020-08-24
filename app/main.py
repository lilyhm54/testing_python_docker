# main file for running test data analysis
import app.src.cleaning.cleaning as cl
import app.config as conf

import pandas as pd
import numpy as np
import tensorflow as tf


def main():

    # load data 
    df = pd.read_csv(conf.FILE_PATHS['TESTING_DATA_PATH'])

    # remove unneeded columns: Name, ticket, Cabin
    df = cl.remove_columns(df, conf.CLEANING_CONFIG['REMOVE_COLUMNS_FIRST_PASS'])

    # remove rows with NaN 
    df = cl.check_for_nan(df)

    # Bucket groups: Age, Fare 
    df = cl.age_buckets(df, conf.CLEANING_CONFIG['AGE_COLUMN'])
    df = cl.fare_buckets(df, conf.CLEANING_CONFIG['FARE_COLUMN'])

    # encode categorical variable: Sex, Embarked 
    df = cl.hot_encoding(df, conf.CLEANING_CONFIG['COLUMNS_TO_HOT_ENCODE'])

    # remove bucketed columns
    df = cl.remove_columns(df, conf.CLEANING_CONFIG['REMOVE_COLUMNS_SECOND_PASS'])

    # df to numpy array
    X = np.array(df)
    
    # Run through model
    reconstructed_model = tf.keras.models.load_model(conf.FILE_PATHS['MODEL_PATH'])
    predicted = reconstructed_model.predict_classes(X)

    # Return df with predicted class 
    df['predicted'] = predicted
    print(df.head())

    return df


if __name__ == '__main__':
    main()

