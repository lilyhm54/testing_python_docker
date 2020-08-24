FILE_PATHS = {
    'TESTING_DATA_PATH': '/Users/lilymartin/DEVOPS/testing_python_docker/app/data/test.csv',
    'MODEL_PATH': '/Users/lilymartin/DEVOPS/testing_python_docker/app/model/20200823_titanic_m1.h5'
}

CLEANING_CONFIG = {
    'REMOVE_COLUMNS_FIRST_PASS': ['Cabin', 'Name', 'Ticket', 'PassengerId'],
    'REMOVE_COLUMNS_SECOND_PASS': ['Age', 'Fare'],
    'COLUMNS_TO_HOT_ENCODE': ['Sex', 'Embarked', 'AgeBucket', 'FareBucket'],
    'AGE_COLUMN': 'Age',
    'FARE_COLUMN': 'Fare'
}
