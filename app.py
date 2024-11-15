import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import util
import data_handler
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

# para rodar esse arquivo
# streamlit run app.py

print("Abriu a pagina")

# verifica se a senha de acesso está correta
#if not util.check_password():
#    # se a senha estiver errada, para o processamento do app
#    print("Usuario nao logado")
#    st.stop()

print("Carregou a pagina")

# Aqui começa a estrutura do App que vai ser executado em produção (nuvem AWS)

# primeiro de tudo, carrega os dados para um dataframe
dados = data_handler.load_data()

# carrega o modelo de predição já treinado e validado
model = pickle.load(open('./models/final_classification_model_games.pkl', 'rb'))   

# começa a estrutura da interface do sistema
st.title('Predict Online Gaming Behavior')

data_analyses_on = st.toggle('Show dataset')

if data_analyses_on:
    # essa parte é só um exemplo de que é possível realizar diversas visualizações e plotagens com o streamlit
    st.header('Gaming Behavior - Dataframe')
    
    # exibe todo o dataframe dos dados
    st.dataframe(dados)

    # plota um gráfico de barras com a contagem dos dados
    st.header('EngagementLevel')
    st.bar_chart(dados.EngagementLevel.value_counts())
    
# daqui em diante vamos montar a inteface para capturar os dados de input do usuário para realizar a predição
st.header('Engagement Level Predict')

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input('Age in years', step=1, value=43)
with col2:
    gender = st.selectbox("Gender", ("Male", "Female"))
with col3:
    location = st.selectbox("Location", ("Asia", "Europe", "USA", "Other"))

col1, col2, col3, col4 = st.columns(4)
with col1:
    game_genre = st.selectbox("Game Genre", ("Action", "RPG", "Simulation", "Sports", "Strategy"))
with col2:
    game_difficulty = st.selectbox("Game Difficulty", ("Easy", "Medium", "Hard"))
with col3:
    in_game_purchases = st.number_input('In Game Purchases', step=1, value=0)
with col4:
    achievements_unlocked = st.number_input('Achievements Unlocked', step=1, value=25)

col1, col2, col3, col4 = st.columns(4)
with col1:
    play_time_hours = st.number_input('Play Time Hours', step=0.1, value=16.3)
with col2:
    sessions_per_week = st.number_input('Sessions Per Week', step=1, value=6)
with col3:
    avg_session_duration_minutes = st.number_input('Avg Session in Minutes', step=1, value=108)
with col4:
    player_level = st.number_input('Player Level', step=1, value=79)

submit = st.button('Predict Engagement')

# data mapping
standard_scaler_age = pickle.load(open('./models/standard_scaler_age.pkl', 'rb'))
one_hot_encoder_gender = pickle.load(open('./models/one_hot_encoder_gender.pkl', 'rb'))
one_hot_encoder_location = pickle.load(open('./models/one_hot_encoder_location.pkl', 'rb'))
one_hot_encoder_game_genre = pickle.load(open('./models/one_hot_encoder_game_genre.pkl', 'rb'))
standard_scaler_play_time_hours = pickle.load(open('./models/standard_scaler_play_time_hours.pkl', 'rb'))
standard_scaler_in_game_purchases = pickle.load(open('./models/standard_scaler_in_game_purchases.pkl', 'rb'))
label_encoder_game_difficulty = pickle.load(open('./models/label_encoder_game_difficulty.pkl', 'rb'))
standard_scaler_sessions_per_week = pickle.load(open('./models/standard_scaler_sessions_per_week.pkl', 'rb'))
standard_scaler_avg_session_duration_minutes = pickle.load(open('./models/standard_scaler_avg_session_duration_minutes.pkl', 'rb'))
standard_scaler_player_level = pickle.load(open('./models/standard_scaler_player_level.pkl', 'rb'))
standard_scaler_achievements_unlocked = pickle.load(open('./models/standard_scaler_achievements_unlocked.pkl', 'rb'))
 
# verifica se o botão submit foi pressionado
if submit:
    x_age = standard_scaler_age.transform(np.array(age).reshape(1, -1))
    x_gender = one_hot_encoder_gender.transform(np.array(gender).reshape(1, -1))
    x_location = one_hot_encoder_location.transform(np.array(location).reshape(1, -1))
    x_game_genre = one_hot_encoder_game_genre.transform(np.array(game_genre).reshape(1, -1))
    x_play_time_hours = standard_scaler_play_time_hours.transform(np.array(play_time_hours).reshape(1, -1))
    x_in_game_purchases = standard_scaler_in_game_purchases.transform(np.array(in_game_purchases).reshape(1, -1))
    x_game_difficulty = label_encoder_game_difficulty.transform(np.array(game_difficulty).reshape(1, -1)).reshape(-1, 1)
    x_sessions_per_week = standard_scaler_sessions_per_week.transform(np.array(sessions_per_week).reshape(1, -1))
    x_avg_session_duration_minutes = standard_scaler_avg_session_duration_minutes.transform(np.array(avg_session_duration_minutes).reshape(1, -1))
    x_player_level = standard_scaler_player_level.transform(np.array(player_level).reshape(1, -1))
    x_achievements_unlocked = standard_scaler_achievements_unlocked.transform(np.array(achievements_unlocked).reshape(1, -1))
    
    x_data = np.concatenate((x_age, x_gender, x_location, x_game_genre, x_play_time_hours, x_in_game_purchases, x_game_difficulty, x_sessions_per_week, x_avg_session_duration_minutes, x_player_level, x_achievements_unlocked), axis=1)
    
    print(x_data.shape)
    print(x_data)
    
    # Realiza a predição
    results = model.predict(x_data)
    print(results)

    # Exibe a classe predita (nivel de engajamento)
    if len(results) == 1:
        st.subheader(results[0])
    
    # Realiza a predição com a probabilidade de cada classe
    results_proba = model.predict_proba(x_data)
    print(results_proba)

    # Exibe a probabilidade de cada classe    
    if len(results_proba) == 1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.text('Low: ' + str(results_proba[0][1]))
        with col2:
            st.text('Medium: ' + str(results_proba[0][2]))
        with col3:
            st.text('High: ' + str(results_proba[0][0]))
    

    
