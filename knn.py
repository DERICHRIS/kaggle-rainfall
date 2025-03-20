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
from sklearn.neighbors import KNeighborsRegressor
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error,r2_score

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

with mlflow.start_run():
    model=KNeighborsRegressor(n_neighbors=1)
    X_train,X_test,y_train,y_test=train_test_split(X_cleaned,y_cleaned,test_size=0.2,random_state=42)

    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)

    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)

    mlflow.log_param("model","KNN1")
    mlflow.log_metric("mean square error",mse)
    mlflow.log_metric("r2",r2)

    mlflow.sklearn.log_model(model,"KNN1")

    print(f"sme :{mse}")
    print(f"r2 : {r2}")