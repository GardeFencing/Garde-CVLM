from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn import linear_model
from pathlib import Path
import pandas as pd
import numpy as np


def predict(data, test_data, k):

    df = pd.read_csv(data)
    df = df.drop(['screen_height', 'screen_width'], axis=1)

    df_test = pd.read_csv(test_data)
    df_test = df_test.drop(['screen_height', 'screen_width'], axis=1)

    # Calculate direction vectors and midpoints for both eyes
    df['eye_center_x'] = (df['left_iris_x'] + df['right_iris_x']) / 2
    df['eye_center_y'] = (df['left_iris_y'] + df['right_iris_y']) / 2
    
    df_test['eye_center_x'] = (df_test['left_iris_x'] + df_test['right_iris_x']) / 2
    df_test['eye_center_y'] = (df_test['left_iris_y'] + df_test['right_iris_y']) / 2

    # Calculate eye movement vectors
    df['left_direction_x'] = df['left_iris_x'].diff()
    df['left_direction_y'] = df['left_iris_y'].diff()
    df['right_direction_x'] = df['right_iris_x'].diff()
    df['right_direction_y'] = df['right_iris_y'].diff()
    
    df_test['left_direction_x'] = df_test['left_iris_x'].diff()
    df_test['left_direction_y'] = df_test['left_iris_y'].diff()
    df_test['right_direction_x'] = df_test['right_iris_x'].diff()
    df_test['right_direction_y'] = df_test['right_iris_y'].diff()
    
    # Calculate inter-eye distance for scale normalization
    df['eye_distance'] = np.sqrt((df['right_iris_x'] - df['left_iris_x'])**2 + 
                                (df['right_iris_y'] - df['left_iris_y'])**2)
    df_test['eye_distance'] = np.sqrt((df_test['right_iris_x'] - df_test['left_iris_x'])**2 + 
                                     (df_test['right_iris_y'] - df_test['left_iris_y'])**2)
    
    # Fill NaN values with 0 for the first row
    df.fillna(0, inplace=True)
    df_test.fillna(0, inplace=True)

    # Enhanced feature set for X coordinate prediction
    X_train_x = df[['eye_center_x', 'left_direction_x', 'right_direction_x', 'eye_distance']]
    y_train_x = df['point_x']

    sc = StandardScaler()
    X_train_x = sc.fit_transform(X_train_x)

    X_test_x = df_test[['eye_center_x', 'left_direction_x', 'right_direction_x', 'eye_distance']]
    y_test_x = df_test['point_x']

    sc = StandardScaler()
    X_test_x = sc.fit_transform(X_test_x)

    # Increased polynomial degree and adjusted regularization for better sensitivity
    model = make_pipeline(PolynomialFeatures(
        3), linear_model.Ridge(alpha=0.3))
    model.fit(X_train_x, y_train_x)
    y_pred_x = model.predict(X_test_x)

    # Enhanced feature set for Y coordinate prediction
    X_train_y = df[['eye_center_y', 'left_direction_y', 'right_direction_y', 'eye_distance']]
    y_train_y = df['point_y']

    sc = StandardScaler()
    X_train_y = sc.fit_transform(X_train_y)

    X_test_y = df_test[['eye_center_y', 'left_direction_y', 'right_direction_y', 'eye_distance']]
    y_test_y = df_test['point_y']

    sc = StandardScaler()
    X_test_y = sc.fit_transform(X_test_y)

    # Increased polynomial degree and adjusted regularization for better sensitivity
    model = make_pipeline(PolynomialFeatures(
        3), linear_model.Ridge(alpha=0.3))
    model.fit(X_train_y, y_train_y)
    y_pred_y = model.predict(X_test_y)

    # Apply coordinate mapping correction
    VIDEO_WIDTH = 640  # Standard webcam width
    VIDEO_HEIGHT = 480  # Standard webcam height
    
    # Map predictions from webcam space to screen space
    y_pred_x = y_pred_x * (VIDEO_WIDTH / np.max(y_pred_x))
    y_pred_y = y_pred_y * (VIDEO_HEIGHT / np.max(y_pred_y))

    data = np.array([y_pred_x, y_pred_y]).T
    model = KMeans(n_clusters=k, n_init='auto', init='k-means++')
    y_kmeans = model.fit_predict(data)

    data = {'True X': y_test_x, 'Predicted X': y_pred_x,
            'True Y': y_test_y, 'Predicted Y': y_pred_y}

    df_data = pd.DataFrame(data)
    df_data['True XY'] = list(zip(df_data['True X'], df_data['True Y']))

    # remove unwanted data
    df_data = df_data[(df_data['Predicted X'] >= 0) &
                      (df_data['Predicted Y'] >= 0)]


    def func_precision_x(group): return np.sqrt(
        np.sum(np.square([group['Predicted X'], group['True X']])))

    def func_presicion_y(group): return np.sqrt(
        np.sum(np.square([group['Predicted Y'], group['True Y']])))

    precision_x = df_data.groupby('True XY').apply(func_precision_x)
    precision_y = df_data.groupby('True XY').apply(func_presicion_y)

    precision_xy = (precision_x + precision_y) / 2
    precision_xy = precision_xy / np.mean(precision_xy)

    def func_accuracy_x(group): return np.sqrt(
        np.sum(np.square([group['True X'] - group['Predicted X']])))

    def func_accuracy_y(group): return np.sqrt(
        np.sum(np.square([group['True Y'] - group['Predicted Y']])))

    accuracy_x = df_data.groupby('True XY').apply(func_accuracy_x)
    accuracy_y = df_data.groupby('True XY').apply(func_accuracy_y)

    accuracy_xy = (accuracy_x + accuracy_y) / 2
    accuracy_xy = accuracy_xy / np.mean(accuracy_xy)

    data = {}

    for index, row in df_data.iterrows():

        outer_key = str(row['True X']).split('.')[0]
        inner_key = str(row['True Y']).split('.')[0]

        if outer_key not in data:
            data[outer_key] = {}

        data[outer_key][inner_key] = {
            'predicted_x': df_data[(df_data['True X'] == row['True X']) & (df_data['True Y'] == row['True Y'])]['Predicted X'].values.tolist(),
            'predicted_y': df_data[(df_data['True X'] == row['True X']) & (df_data['True Y'] == row['True Y'])]['Predicted Y'].values.tolist(),
            'PrecisionSD': precision_xy[(row['True X'], row['True Y'])],
            'Accuracy': accuracy_xy[(row['True X'], row['True Y'])]
        }

    data['centroids'] = model.cluster_centers_.tolist()

    return data


def train_to_validate_calib(calib_csv_file, predict_csv_file):
    dataset_train_path = calib_csv_file
    dataset_predict_path = predict_csv_file

    # Carregue os dados de treinamento a partir do CSV
    data = pd.read_csv(dataset_train_path)

    # Para evitar que retorne valores negativos: Aplicar uma transformação logarítmica aos rótulos (point_x e point_y)
    # data['point_x'] = np.log(data['point_x'])
    # data['point_y'] = np.log(data['point_y'])

    # Separe os recursos (X) e os rótulos (y)
    X = data[['left_iris_x', 'left_iris_y', 'right_iris_x', 'right_iris_y']]
    y = data[['point_x', 'point_y']]

    # Crie e ajuste um modelo de regressão linear
    model = linear_model.LinearRegression()
    model.fit(X, y)

    # Carregue os dados de teste a partir de um novo arquivo CSV
    dados_teste = pd.read_csv(dataset_predict_path)

    # Faça previsões
    previsoes = model.predict(dados_teste)

    # Para evitar que retorne valores negativos: Inverter a transformação logarítmica nas previsões
    # previsoes = np.exp(previsoes)

    # Exiba as previsões
    print("Previsões de point_x e point_y:")
    print(previsoes)
    return previsoes.tolist()


def train_model(session_id):
    # Download dataset
    dataset_train_path = f'{Path().absolute()}/public/training/{session_id}/train_data.csv'
    dataset_session_path = f'{Path().absolute()}/public/sessions/{session_id}/session_data.csv'

    # Importing data from csv
    raw_dataset = pd.read_csv(dataset_train_path)
    session_dataset = pd.read_csv(dataset_session_path)

    # Calculate eye centers and movement vectors
    raw_dataset['eye_center_x'] = (raw_dataset['left_iris_x'] + raw_dataset['right_iris_x']) / 2
    raw_dataset['eye_center_y'] = (raw_dataset['left_iris_y'] + raw_dataset['right_iris_y']) / 2
    
    session_dataset['eye_center_x'] = (session_dataset['left_iris_x'] + session_dataset['right_iris_x']) / 2
    session_dataset['eye_center_y'] = (session_dataset['left_iris_y'] + session_dataset['right_iris_y']) / 2

    # Calculate direction vectors
    raw_dataset['left_direction_x'] = raw_dataset['left_iris_x'].diff()
    raw_dataset['left_direction_y'] = raw_dataset['left_iris_y'].diff()
    raw_dataset['right_direction_x'] = raw_dataset['right_iris_x'].diff()
    raw_dataset['right_direction_y'] = raw_dataset['right_iris_y'].diff()
    
    session_dataset['left_direction_x'] = session_dataset['left_iris_x'].diff()
    session_dataset['left_direction_y'] = session_dataset['left_iris_y'].diff()
    session_dataset['right_direction_x'] = session_dataset['right_iris_x'].diff()
    session_dataset['right_direction_y'] = session_dataset['right_iris_y'].diff()
    
    # Calculate inter-eye distance for scale normalization
    raw_dataset['eye_distance'] = np.sqrt((raw_dataset['right_iris_x'] - raw_dataset['left_iris_x'])**2 + 
                                        (raw_dataset['right_iris_y'] - raw_dataset['left_iris_y'])**2)
    session_dataset['eye_distance'] = np.sqrt((session_dataset['right_iris_x'] - session_dataset['left_iris_x'])**2 + 
                                            (session_dataset['right_iris_y'] - session_dataset['left_iris_y'])**2)
    
    # Fill NaN values with 0 for the first row
    raw_dataset.fillna(0, inplace=True)
    session_dataset.fillna(0, inplace=True)

    train_stats = raw_dataset.describe()
    train_stats = train_stats.transpose()

    dataset_t = raw_dataset
    dataset_s = session_dataset.drop(['timestamp'], axis=1)

    # Drop the columns that will be predicted
    X = dataset_t.drop(['timestamp', 'mouse_x', 'mouse_y'], axis=1)

    Y1 = dataset_t.mouse_x
    Y2 = dataset_t.mouse_y

    # Create polynomial features for better direction sensitivity
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Use Ridge regression with adjusted alpha for better sensitivity
    MODEL_X = linear_model.Ridge(alpha=0.3)
    MODEL_Y = linear_model.Ridge(alpha=0.3)
    
    MODEL_X.fit(X_poly, Y1)
    MODEL_Y.fit(X_poly, Y2)

    # Transform session data
    X_session_poly = poly.transform(dataset_s)
    
    GAZE_X = MODEL_X.predict(X_session_poly)
    GAZE_Y = MODEL_Y.predict(X_session_poly)

    # Apply coordinate mapping correction
    VIDEO_WIDTH = 640  # Standard webcam width
    VIDEO_HEIGHT = 480  # Standard webcam height
    
    # Map predictions from webcam space to screen space
    GAZE_X = GAZE_X * (VIDEO_WIDTH / np.max(GAZE_X))
    GAZE_Y = GAZE_Y * (VIDEO_HEIGHT / np.max(GAZE_Y))

    # Ensure predictions are within screen bounds
    GAZE_X = np.clip(GAZE_X, 0, VIDEO_WIDTH)
    GAZE_Y = np.clip(GAZE_Y, 0, VIDEO_HEIGHT)
    
    # Apply exponential moving average for smoother predictions with shorter window
    window = 2  # Reduced from 3 for faster response
    GAZE_X = pd.Series(GAZE_X).ewm(span=window).mean().values
    GAZE_Y = pd.Series(GAZE_Y).ewm(span=window).mean().values

    return {"x": GAZE_X, "y": GAZE_Y}


def model_for_mouse_x(X, Y1):
    print('-----------------MODEL FOR X------------------')
    # split dataset into train and test sets (80/20 where 20 is for test)
    X_train, X_test, Y1_train, Y1_test = train_test_split(X, Y1, test_size=0.2)

    model = linear_model.LinearRegression()
    model.fit(X_train, Y1_train)

    Y1_pred_train = model.predict(X_train)
    Y1_pred_test = model.predict(X_test)

    Y1_test = normalizeData(Y1_test)
    Y1_pred_test = normalizeData(Y1_pred_test)

    print(
        f'Mean absolute error MAE = {mean_absolute_error(Y1_test, Y1_pred_test)}')
    print(
        f'Mean squared error MSE = {mean_squared_error(Y1_test, Y1_pred_test)}')
    print(
        f'Mean squared log error MSLE = {mean_squared_log_error(Y1_test, Y1_pred_test)}')
    print(f'MODEL X SCORE R2 = {model.score(X, Y1)}')

    # print(f'TRAIN{Y1_pred_train}')
    # print(f'TEST{Y1_pred_test}')
    return model


def model_for_mouse_y(X, Y2):
    print('-----------------MODEL FOR Y------------------')
    # split dataset into train and test sets (80/20 where 20 is for test)
    X_train, X_test, Y2_train, Y2_test = train_test_split(X, Y2, test_size=0.2)

    model = linear_model.LinearRegression()
    model.fit(X_train, Y2_train)

    Y2_pred_train = model.predict(X_train)
    Y2_pred_test = model.predict(X_test)

    Y2_test = normalizeData(Y2_test)
    Y2_pred_test = normalizeData(Y2_pred_test)

    print(
        f'Mean absolute error MAE = {mean_absolute_error(Y2_test, Y2_pred_test)}')
    print(
        f'Mean squared error MSE = {mean_squared_error(Y2_test, Y2_pred_test)}')
    print(
        f'Mean squared log error MSLE = {mean_squared_log_error(Y2_test, Y2_pred_test)}')
    print(f'MODEL X SCORE R2 = {model.score(X, Y2)}')

    # print(f'TRAIN{Y2_pred_train}')
    print(f'TEST{Y2_pred_test}')
    return model


def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
