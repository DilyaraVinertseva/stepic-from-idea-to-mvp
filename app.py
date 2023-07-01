import pandas as pd
import streamlit as st
from PIL import Image
from model import open_data, preprocess_data, split_data, load_model_and_predict


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('smile_1.jfif')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Aircompany customer's satisfaction predictor",
        page_icon=image,

    )

    st.write(
        """
        # Классификация пассажиров авиакомпании
        Определяем, кто из пассажиров остался доволен полетом, а кто – нет.
        """
    )

    st.image(image)


def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)


def write_prediction(prediction):
    st.write("## Предсказание")
    st.write(prediction)

    # st.write("## Вероятность удовлетворенности")
    # st.write(prediction_proba)


def process_side_bar_inputs():
    st.sidebar.header('Заданные пользователем параметры')
    user_input_df = sidebar_input_features()

    train_df = open_data()
    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]
    write_user_data(user_X_df)

    prediction = load_model_and_predict(user_X_df)
    write_prediction(prediction)


def sidebar_input_features():
    '''gender', 'age', 'customer_type', 'type_of_travel', 'class',
       'flight_distance', 'departure_delay_in_minutes',
       'arrival_delay_in_minutes', 'inflight_wifi_service',
       'departure/arrival_time_convenient', 'ease_of_online_booking',
       'gate_location', 'food_and_drink', 'online_boarding', 'seat_comfort',
       'inflight_entertainment', 'on-board_service', 'leg_room_service',
       'baggage_handling', 'checkin_service', 'inflight_service',
       'cleanliness', 'satisfaction'],
'''
    gender = st.sidebar.selectbox("Пол", ("Мужской", "Женский"))
    age = st.sidebar.slider("Возраст", min_value=0, max_value=100, value=20,
                            step=1)
    customer_type = st.sidebar.selectbox('Тип_лояльности', ('Лояльный', "Нелояльный"))
    type_of_travel = st.sidebar.selectbox("Тип поездки", ("Рабочая", "Частная"))
    servclass = st.sidebar.selectbox("Класс обслуживания", ("Business", "Eco", "Eco Plus"))
    flight_distance = st.sidebar.slider("Дальность перелета", 
                        min_value=0, max_value=9320, value=500, step=1)
    departure_delay_in_minutes = st.sidebar.slider("Задержка отправления", 
                        min_value=0, max_value=25000, value=0, step=1)
    arrival_delay_in_minutes = st.sidebar.slider("Задержка прибытия", 
                        min_value=0, max_value=15000, value=0, step=1)
    inflight_wifi_service = st.sidebar.slider("Оценка клиентом интернета на борту",
                        min_value=1, max_value=5, value=1, step=1)
    dep_arvl_time_convenient = st.sidebar.slider("Оценка клиентом удобства времени прилета и вылета", 
                        min_value=1, max_value=5, value=1, step=1)
    ease_of_online_booking = st.sidebar.slider("Оценка клиентом удобства онлайн-бронирования",
                         min_value=1, max_value=5, value=1, step=1)
    gate_location = st.sidebar.slider("Оценка клиентом расположения выхода на посадку в аэропорту", 
                        min_value=1, max_value=5, value=1, step=1)
    food_and_drink = st.sidebar.slider("Оценка клиентом еды и напитков на борту",
                        min_value=1, max_value=5, value=1, step=1)
    online_boarding = st.sidebar.slider("Оценка клиентом выбора места в самолете", 
                        min_value=1, max_value=5, value=1, step=1)
    seat_comfort = st.sidebar.slider("Оценка клиентом удобства сиденья", 
                        min_value=1, max_value=5, value=1, step=1)
    inflight_entertainment = st.sidebar.slider("Оценка клиентом развлечений на борту", 
                        min_value=1, max_value=5, value=1, step=1)
    on_board_service = st.sidebar.slider("Оценка клиентом посадки на борт", 
                        min_value=1, max_value=5, value=1, step=1)
    leg_room_service = st.sidebar.slider("Оценка клиентом места в ногах на борту", 
                        min_value=1, max_value=5, value=1, step=1)
    baggage_handling = st.sidebar.slider("Оценка клиентом обращения с багажом", 
                        min_value=1, max_value=5, value=1, step=1)
    checkin_service = st.sidebar.slider("Оценка клиентом регистрации на рейс", 
                        min_value=1, max_value=5, value=1, step=1)
    inflight_service = st.sidebar.slider("Оценка клиентом обслуживания на борту", 
                        min_value=1, max_value=5, value=1, step=1)
    cleanliness = st.sidebar.slider("Оценка клиентом чистоты на борту", 
                        min_value=1, max_value=5, value=1, step=1)
    

    translatetion = {
        "Мужской": "male",
        "Женский": "female",
        "Лояльный": "Loyal Customer",
        "Нелояльный": "disloyal Customer",
        "Рабочая": "Business travel",
        "Частная": "Personal Travel",
    }

    data = {
        "gender": translatetion[gender],
        "age": age,
        'customer_type': translatetion[customer_type],
        'type_of_travel': translatetion[type_of_travel], 
        'class': servclass,
       'flight_distance': flight_distance,
        'departure_delay_in_minutes': departure_delay_in_minutes,
       'arrival_delay_in_minutes': arrival_delay_in_minutes, 
       'inflight_wifi_service': inflight_wifi_service,
       'departure/arrival_time_convenient': dep_arvl_time_convenient, 
       'ease_of_online_booking':ease_of_online_booking,
       'gate_location': gate_location, 
       'food_and_drink': food_and_drink, 
       'online_boarding': online_boarding, 
       'seat_comfort': seat_comfort,
       'inflight_entertainment': inflight_entertainment, 
       'on-board_service': on_board_service, 
       'leg_room_service': leg_room_service,
       'baggage_handling': baggage_handling, 
       'checkin_service' : checkin_service, 
       'inflight_service': inflight_service,
       'cleanliness': cleanliness
        
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()
