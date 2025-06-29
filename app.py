import joblib
import pandas as pd
import numpy as np
import random
import streamlit as st
from datetime import time, date
from geopy.geocoders import Nominatim

def escale(x_input, min_scaled, max_scaled, min_original, max_original):
    x_final = ((x_input - min_original) / (max_original - min_original)) * (max_scaled - min_scaled) + min_scaled
    return x_final


# Inicializa o geocodificador
geolocator = Nominatim(user_agent="geoapi")

def treat_columns(df_train):
  df = df_train.copy()

  y = df['Class']
  X = df.drop(columns=['Class','Time'])

  ranges = [(-3.08, 3.29), (-1.49, 2.06), (-0.88, 2.78), (-2.42, 2.94), (-1.81, 1.43), (-1.80, 1.64), (-1.65, 1.49), (-0.62, 0.72), (-1.69, 3.06), (-1.61, 1.11), (-1.69, 3.11), (-4.73, 2.52), (-1.82, 3.42), (-1.43, 2.90), (-1.90, 1.83), (-1.54, 1.59), (-1.16, 1.79), (-1.44, 1.41), (-1.55, 1.43), (-0.47, 0.46), (-0.57, 0.35), (-1.36, 1.06), (-0.42, 0.32), (-1.06, 1.13), (-0.68, 0.94), (-1.03, 1.03), (-0.24, 0.27), (-0.10, 0.17), (-45.69, 107.85)]

  for col, (lower, upper) in zip(X.columns, ranges):
    X[col] = X[col].clip(lower=lower, upper=upper)

  numeric = X.columns
  scaler = StandardScaler()
  X[numeric] = scaler.fit_transform(X[numeric])

  features = X.copy()
  labels = y.copy()

  return features, labels

chanel_options = ['Aplicativo', 'Site', 'Loja Física', 'Loja de Aplicativos', 'Carteira Digital', 'Outros']
estab_options = ['Supermercado', 'Padaria', 'Restaurante', 'Lanchonete', 'Farmácia', 'Posto de gasolina', 'Salão de cabeleireiro', 'Clínica médica', 'Loja de roupas', 'Loja de eletrônicos', 'Academia', 'Barbearia', 'Pizzaria', 'Cafeteria', 'Loja de cosméticos', 'Outros']

dict_chanels = {
    'Aplicativo': '-1',
    'Site': '0',
    'Loja Física': '1',
    'Loja de Aplicativos': '2',
    'Carteira Digital': '3',
    'Outros': '4'
}

dict_estab = {
    'Supermercado': '-50',
    'Padaria': '-44',
    'Restaurante': '-38',
    'Lanchonete': '-32',
    'Farmácia': '-26',
    'Posto de gasolina': '-20',
    'Salão de cabeleireiro': '-14',
    'Clínica médica': '-8',
    'Loja de roupas': '-2',
    'Loja de eletrônicos': '4',
    'Academia': '10',
    'Barbearia': '16',
    'Pizzaria': '22',
    'Cafeteria': '28',
    'Loja de cosméticos': '34',
    'Outros': '39'
}


# Carregar modelo e dados
pipeline = joblib.load('final_model.pkl')
df_original = pd.read_parquet('Classes.parquet')


# Início do design do site
st.set_page_config(layout="wide")
st.title("Identificação de Fraudes com Cartão de Crédito")

menu = st.sidebar.selectbox("Escolha uma opção", [
    "Entenda os dados",
    "Preveja se uma transação é fraudulenta",
    "Busque transações por ID",
    "Entenda a escolha do modelo"
])

if menu == "Entenda os dados":

elif menu == "Busque transações por ID":

elif menu == "Entenda a escolha do modelo":
