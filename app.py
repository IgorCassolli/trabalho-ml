import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import util
import data_handler
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

# para rodar esse arquivo
# streamlit run app.py

st.set_page_config(layout="wide")

st.title("Trabalho do Grau B - Ciência de Dados e Big Data")
st.header("Autores: ")
st.subheader("Aluno 1: Jones Marlos Pinheiro da Rosa")
st.subheader("Aluno 2: Igor da Silva Cassolli", divider="gray")

st.markdown("<a href='https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset'>"+
            "Dataframe: Predict Online Gaming Behavior Dataset</a>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: left;'>Modelagem preditiva com <span style='color: red;'>objetivo"+
            "</span> de predizer <span style='color: red;'>a retenção de um jogador.</span></h5>", unsafe_allow_html=True)

st.header('Dados do Dataset')

#Carregar CSV
dados = data_handler.load_data()

#Carregar modelo treinado
model = pickle.load(open('./models/final_classification_model_games.pkl', 'rb'))   

on = st.toggle('Exibir análise dos dataset')

if on:
    st.dataframe(dados)

    st.subheader('Contagem por Engajamento')
    st.bar_chart(dados.EngagementLevel.value_counts())

    st.subheader('Contagem por gênero')
    st.bar_chart(dados.Gender.value_counts())

    st.subheader("Horas Jogadas por Gênero e Tipo de Jogo")
    avg_playtime_by_genre_gender = dados.groupby(["GameGenre", "Gender"])["PlayTimeHours"].mean().reset_index()

    chart_data = avg_playtime_by_genre_gender.pivot(index="GameGenre", columns="Gender", values="PlayTimeHours")

    fig, ax = plt.subplots(figsize=(16, 6))
    for gender in chart_data.columns:
        ax.plot(chart_data.index, chart_data[gender], label=gender, marker='o')
    ax.set_xlabel("Gênero do Jogo", fontsize=10)
    ax.set_ylabel("Horas Jogadas (Média)", fontsize=10)
    ax.legend(title="Gênero", fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Nível Médio dos Jogadores por Engajamento")
    avg_level_by_engagement = dados.groupby("EngagementLevel")["PlayerLevel"].mean()
    st.bar_chart(avg_level_by_engagement)

    st.subheader("Distribuição de Jogadores por Dificuldade do Jogo")
    difficulty_counts = dados['GameDifficulty'].value_counts()
    st.bar_chart(difficulty_counts)
    
st.header('Preditor de retenção do jogador')

inGamePurchasesMap = {
    "Não": 0,
    "Sim": 1,
}

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input('Idade:', step=1, value=0)
with col2:
    gender = st.selectbox("Gênero:", ("Male", "Female"))
with col3:
    location = st.selectbox("Localização:", ("Asia", "Europe", "USA", "Other"))

col1, col2, col3, col4 = st.columns(4)
with col1:
    game_genre = st.selectbox("Gênero do jogo:", ("Action", "RPG", "Simulation", "Sports", "Strategy"))
with col2:
    game_difficulty = st.selectbox("Dificuldade do jogo:", ("Easy", "Medium", "Hard"))
with col3:
    in_game_purchases = inGamePurchasesMap[st.selectbox("Tem compras no jogo?", ("Não", "Sim"))]
with col4:
    achievements_unlocked = st.number_input('Número de conquistas desbloqueadas:', step=1, value=0)

col1, col2, col3, col4 = st.columns(4)
with col1:
    play_time_hours = st.number_input('Média de horas jogadas por sessões:', step=0.1, value=0.0)
with col2:
    sessions_per_week = st.number_input('Número de sessões de jogo por semana:', step=1, value=0)
with col3:
    avg_session_duration_minutes = st.number_input('Duração média de cada sessão:', step=1, value=0)
with col4:
    player_level = st.number_input('Nível atual do jogador:', step=1, value=0)

submit = st.button('Predizer')

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
    
    x_data = np.concatenate((x_age, x_gender, x_location, x_game_genre, 
                             x_play_time_hours, x_in_game_purchases, x_game_difficulty, 
                             x_sessions_per_week, x_avg_session_duration_minutes, x_player_level,
                             x_achievements_unlocked), axis=1)
    
    # Realiza a predição
    results = model.predict(x_data)

    if len(results) == 1:
        st.subheader(results[0])
    
    results_proba = model.predict_proba(x_data)

    # Exibe a probabilidade de cada classe    
    if len(results_proba) == 1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.text('Low: ' + str(results_proba[0][1]))
        with col2:
            st.text('Medium: ' + str(results_proba[0][2]))
        with col3:
            st.text('High: ' + str(results_proba[0][0]))
    

    
