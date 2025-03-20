import pandas as pd

data=pd.read_csv("C:\\Users\\deric\\Desktop\\mlops\\kaggle_rainfall\\train.csv")
y=data["rainfall"]

X=data.drop(columns=['id','rainfall'])

#striping extra spaces
X.columns = X.columns.str.strip()

# droping rows with NaN
X_cleaned = X.dropna()
y_cleaned=y.dropna()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri('http://127.0.0.1:5000')
with mlflow.start_run():
    X_train,X_test,y_train,y_test=train_test_split(X_cleaned,y_cleaned,test_size=0.2,random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build the Neural Network model
    model = Sequential()

    # Input layer (input shape should match the number of features)
    model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))

    # Hidden layer
    model.add(Dense(32, activation='relu'))

    # Output layer (single neuron for regression task)
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    # Train the model
    history = model.fit(X_train_scaled, y_train, epochs=150, batch_size=32, validation_data=(X_test_scaled, y_test))

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)

    mlflow.log_param("model","ANN2")
    mlflow.log_metric("mean square error",mse)
    mlflow.log_metric("r2",r2)

    mlflow.sklearn.log_model(model,"ANN2")

    print(f"sme :{mse}")
    print(f"r2 : {r2}")