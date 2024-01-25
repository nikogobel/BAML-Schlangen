import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.layers import BatchNormalization

seed = 2024
np.random.seed(seed)

def load_datasets():
    # Load datasets
    df_train = pd.read_csv('cleaned_training_dataset.csv')
    df_test = pd.read_pickle('cleaned_test_dataset.pkl')
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
    
    df_train = df_train.astype('float32')
    df_test = df_test.astype('float32')
    
    # Splitting the data
    X = df_train.drop(columns=['reviews_Like'])
    y = df_train['reviews_Like'].astype(int)  # Ensure correct encoding
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    

    # Normalizing the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    y_val.to_csv('y_val_NN.csv')
    return X_train, y_train, X_val, y_val, df_test

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

def train_model(model, X_train, y_train, X_val, y_val):
    # Training the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

    return history

def evaluate_model(model, X_val, y_val):
    # Evaluating the model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
    
    
    pred = model.predict(X_val)
    pred = np.round(pred)
    y_val = y_val.to_numpy()
    
    # calculate accuracy on training set
    error_rate = np.mean(y_val != pred)
    print("Error rate:", error_rate)
    print("Balanced Validation Accuracy:", balanced_accuracy_score(y_val, pred))

def main():
    df_train, df_test, df_index, df_submission= load_datasets()
    X_train, y_train, X_val, y_val, df_test = preprocess_data(df_train, df_test)
    model = build_model(input_shape=(X_train.shape[1],))
    
    # train model
    history = train_model(model, X_train, y_train, X_val, y_val)

    evaluate_model(model, X_val, y_val)
    
    # predict on test set
    test_pred = model.predict(df_test)
    
    # match predictions with index
    df_index['test_pred'] = test_pred
    df_submission['prediction'] = df_index.set_index('reviews_TestSetId')['test_pred'].reindex(
        df_submission['id']).values
    df_submission['prediction'] = df_submission['prediction'].fillna(0.0)
    df_submission['prediction'] = df_submission['prediction'].apply(lambda x: 0 if x < 0.5 else 1).astype(int)
    
    # save submission
    df_submission.to_csv('predictions_BAML_Schlangen_2.csv', index=False)

if __name__ == "__main__":
    main()
