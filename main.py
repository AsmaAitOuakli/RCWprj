from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC, SVR
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import base64

app = FastAPI()

# Autoriser toutes les origines
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def handle_missing_values(df, method):
    if method == 'FFill':
        df.fillna(method='ffill', inplace=True)
    elif method == 'BFill':
        df.fillna(method='bfill', inplace=True)
    elif method == 'FFill&BFfill':
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

def handle_categorical_values(df, method):
    categorical_columns = df.select_dtypes(include=['object']).columns
    if method == 'One-Hot Encoding':
        df_encoded = pd.get_dummies(df, columns=categorical_columns)
    elif method == 'Ordinal Encoding':
        encoder = OrdinalEncoder()
        df_encoded = df.copy()
        df_encoded[categorical_columns] = encoder.fit_transform(df[categorical_columns])
    return df_encoded

def check_column_type(column_values):
    unique_values = column_values.unique()
    num_unique_values = len(unique_values)
    if num_unique_values <= 10:
        return 'Discrete'
    else:
        return 'Continuous'
    
@app.get("/")
async def read_root():
    return {"message": "Hello World"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    return {"columns": df.columns.tolist(), "head": df.head().to_dict(), "tail": df.tail().to_dict()}

@app.post("/eda")
async def eda(file: UploadFile = File(...), encoding_method: str = 'One-Hot Encoding'):
    df = pd.read_csv(file.file)
    df_encoded = handle_categorical_values(df, encoding_method)
    corr_matrix = df_encoded.corr().to_dict()
    return {"correlation_matrix": corr_matrix, "head": df_encoded.head().to_dict(), "tail": df_encoded.tail().to_dict()}

@app.post("/predictive_model")
async def predictive_model(output_column: str, file: UploadFile = File(...), encoding_method: str = 'One-Hot Encoding', missing_method: str = 'None'):
    df = pd.read_csv(file.file)
    if encoding_method != 'None':
        df = handle_categorical_values(df, encoding_method)
    if missing_method != 'None':
        handle_missing_values(df, missing_method)
    X = df.drop(columns=[output_column])
    Y = df[output_column]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=9)
    column_type = check_column_type(Y)
    if column_type == 'Discrete':
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        matrix = confusion_matrix(Y_test, predictions).tolist()
        return {"confusion_matrix": matrix}
    elif column_type == 'Continuous':
        model = LinearRegression()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(Y_test, predictions)
        mse = mean_squared_error(Y_test, predictions)
        r2 = r2_score(Y_test, predictions)
        return {"mae": mae, "mse": mse, "r2": r2}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
