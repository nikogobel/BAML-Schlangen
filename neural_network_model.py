import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.layers import BatchNormalization

def load_datasets():
    # Load datasets
    df_train = pd.read_csv('cleaned_training_dataset.csv')
    df_test = pd.read_pickle('cleaned_test_dataset.pkl')  # Assuming this exists
    df_index = pd.read_csv('test_set_id.csv')
    df_submission = pd.read_csv('pub_YwCznU3.csv')

    # Drop the 'Unnamed: 0' column
    df_train.drop(columns=['Unnamed: 0'], inplace=True)

    return df_train, df_test, df_index, df_submission

def preprocess_data(df_train, df_test):
    # Identify non-numeric columns
    non_numeric_cols_train = df_train.select_dtypes(include=['object', 'category']).columns
    non_numeric_cols_test = df_test.select_dtypes(include=['object', 'category']).columns

    # Apply one-hot encoding to non-numeric columns
    df_train = pd.get_dummies(df_train, columns=non_numeric_cols_train, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=non_numeric_cols_test, drop_first=True)

    # Splitting the data
    X = df_train.drop(columns=['reviews_Like'])
    y = df_train['reviews_Like'].astype(int)  # Ensure correct encoding

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Normalizing the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def build_model(input_shape):
    # Neural Network Model
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, X_train, y_train, X_test, y_test):
    # Training the model
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

    return history

def evaluate_model(model, X_test, y_test):
    # Evaluating the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

def main():
    df_train, df_test, _, _ = load_datasets()
    X_train, X_test, y_train, y_test = preprocess_data(df_train, df_test)
    model = build_model(input_shape=(X_train.shape[1],))
    history = train_model(model, X_train, y_train, X_test, y_test)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
