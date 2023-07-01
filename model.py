import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from pickle import dump, load

def open_data(path="clients.csv"):
    df = pd.read_csv(path)
    df.columns = [i.replace(' ', '_').lower() for i in df.columns]

    return df


def split_data(df: pd.DataFrame):
    #get target
    df['target'] = 0
    df.loc[df.satisfaction == 'satisfied', 'target'] = 1
    df = df.drop('satisfaction', axis=1)
    y = df['target']
    X = df.drop('target', axis = 1)

    return X, y


def preprocess_data(df: pd.DataFrame, test=True):
    #delete na
    # df = df.drop('id', axes=1)
    df = df.dropna()

    #encode gender
    df.loc[df['gender'] == 'Male', 'gender'] = 1
    df.loc[df['gender'] == 'Female', 'gender'] = 0

    #replace zero and >100 age with median
    mediana = df.loc[(18 < df['age']) & (df['age']< 100), 'age'].median()
    df.loc[df['age'] == 0, 'age'] = mediana
    df.loc[df['age'] > 100, 'age'] = mediana

    #encode customer type
    df.loc[df['customer_type'] == 'Loyal Customer', 'customer_type'] = 1
    df.loc[df['customer_type'] == 'disloyal Customer', 'customer_type'] = 0

    #encode type of travel
    df.loc[df['type_of_travel'] == 'Business travel', 'type_of_travel'] = 1
    df.loc[df['type_of_travel'] == 'Personal Travel', 'type_of_travel'] = 0

    #delete idx with flight_distance > 9320, replace zeros to median
    HIGH_BORDER = 9320
    df = df.loc[df['flight_distance'] <= HIGH_BORDER]
    df.loc[df['flight_distance'] == 0,  'flight_distance'] = df.flight_distance.median()

    #create is_delayed, drop departure_delay_in_minutes
    df['is_delayed'] = np.where(df['departure_delay_in_minutes'] == 0, 0, 1)
    df = df.drop('departure_delay_in_minutes', axis=1)

    #create is_arrival_delayed, drop arrival_delay_in_minutes
    df['is_arrival_delayed'] = np.where(df['arrival_delay_in_minutes'] == 0, 0, 1)
    df = df.drop('arrival_delay_in_minutes', axis=1)

    # remove all except (0, 1, 2, 3, 4, 5), zeros replace to moda
    cols = ['inflight_wifi_service', 'departure/arrival_time_convenient', 'ease_of_online_booking',
       'gate_location', 'food_and_drink', 'online_boarding', 'seat_comfort',
       'inflight_entertainment', 'on-board_service', 'leg_room_service',
       'baggage_handling', 'checkin_service', 'inflight_service',
       'cleanliness']
    for col in cols:
        df  = df[(df[col] == 0.0) | (df[col] == 1.0) | (df[col] == 2.0)
      | (df[col] == 3.0) | (df[col] == 4.0) | (df[col] == 5.0)]
        df.loc[df[col] == 0.0, col] = df[col].mode()
    
    df = df.astype({
    'gender': int,
    'customer_type': int,
    'type_of_travel' : int,
    'class': 'object'
    })
    df["inflight_wifi_service"] = df["inflight_wifi_service"].astype("object")
    for col in cols:
        df[col] = df[col].astype('object')

    

    if test:
        X_df, y_df = split_data(df)
    else:
    
        
        X_df = df

    if test:
        return X_df, y_df
    else:
        return X_df
    

def fit_and_save_model(X_df, y_df, path="model.pickle"):
    model = CatBoostClassifier(cat_features=['class'],
                           loss_function="Logloss",
                           custom_metric='AUC',
                           random_seed=42,
                           early_stopping_rounds=20,
                           auto_class_weights='Balanced',
                           learning_rate=.02)
    model.fit(X_df, y_df)

    test_prediction = model.predict_proba(X_df)[:, 1]
    score = roc_auc_score( y_df, test_prediction)
    print(f"Model roc-auc score is {score}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")


def load_model_and_predict(df, path="model.pickle"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)
    # prediction = np.squeeze(prediction)

    prediction_proba = model.predict_proba(df)[0][1]
    # prediction_proba = np.squeeze(prediction_proba)

    # encode_prediction_proba = {
    #     0: "Клиент не доволен с вероятностью",
    #     1: "Клиент доволен с вероятностью"
    # }

    encode_prediction = {
        0: "Жаль, что клиент остался не доволен сервисом, будем работать!",
        1: "Ура! клиент ушел довольным, так держать!"
    }

    # prediction_data = {}
    # for key, value in encode_prediction_proba.items():
    #     prediction_data.update({value: prediction_proba[key]})

    # prediction_df = pd.DataFrame(prediction_data, index=[0])
    # prediction = encode_prediction[prediction]

    return encode_prediction[prediction], prediction_proba

if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    fit_and_save_model(X_df, y_df)
