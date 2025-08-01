import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

def train_model():
    df = pd.read_csv('creditcard_2023.csv')
    print(df.head())
    print(df.isnull().sum())
    df.fillna(df.select_dtypes(include=np.number).mean(), inplace=True)
    X = df.drop(['id', 'Class'], axis = 1, errors='ignore')
    y = df['Class']

    #This basically puts the model in 80 percent train and 20 percent test
    # Always remove the useless features 
    #Then Split the features
    #After that Remove Scale the values the standard way 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    scaler = StandardScaler()
    X_test.shape


    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(
    n_estimators = 100,
    max_depth = 10, 
    min_samples_split = 5, 
    random_state = 42
)
    rf_model.fit(X_train_scaled, y_train)
    y_pred = rf_model.predict(X_test_scaled)

    print(classification_report(y_test, y_pred))
    return rf_model, scaler


if __name__ == "__main__":
    rf_model = train_model()
