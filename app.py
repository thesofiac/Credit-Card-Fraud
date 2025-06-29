import joblib
import re
import pandas as pd
import numpy as np
import random
import streamlit as st
from datetime import time, date
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
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
    st.subheader("Entenda os dados")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div style='text-align: justify'><h5>Neste projeto, temos por objetivo determinar transações fraudulentas feitas com cartão de crédito, a partir de um banco de dados de quase 300 mil transações, descritas por meio de 28 variáveis transformadas e pelo seu valor. <br><br> Por se tratarem de dados confidenciais, não trabalhamos com as features originais e sim transformadas. <br><br> A partir delas, foi possível determinar o modelo de Classificação Binária, que possibilitou o cálculo da probabilidade de uma transação ser ou não fraudulenta. <br><br> Em casos como esse, é de interesse principal a detecção de fraude, mesmo que em detrimento de um maior número de falsos positivos, pois entende-se que, sinalizada a transação como uma possível fraude, outros mecanismos de segurança podem ser utilizados em paralelo para confirmar não prejudicar o cliente. <br><br> O modelo escolhido foi o XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=200, random_state=42, subsample=0.8), que mostrou recall de 94% em dados de teste, valor considerado alto, já que os dados são altamente desbalanceados - com apenas 0.17% de positivos.</h5></div>", unsafe_allow_html=True)

elif menu == "Preveja se uma transação é fraudulenta":
    st.subheader("Preveja se uma transação é fraudulenta")

    # Transação
    f1 = st.number_input("Valor da Transação", value=0.0)
    f2 = st.date_input("Data da Transação", value=date.today())
    dt_dias = int((date.today() - f2).days)

    f3 = st.time_input("Horário da Transação", value=time(8, 30))

    cep1 = st.text_area("CEP da transação (somente números)", value="00000000")
    cep1_limpo = re.sub(r"\D", "", cep1)

    if len(cep1_limpo) == 8:
        cep1_formatado = f"{cep1_limpo[:5]}-{cep1_limpo[5:]}"

    location1 = geolocator.geocode({"postalcode": cep1_formatado, "country": "Brazil"})

    if location1:
        f4 = location1.latitude
        f5 = location1.longitude

    f6 = st.number_input("IP do dispositivo (somente números)", value=0.0)
    f7 = st.selectbox("Canal de compra", chanel_options)

    # Transação Anterior
    f8 = st.number_input("Valor da Última Transação", value=0.0)
    f9 = st.date_input("Data da última Transação", value=date.today())
    dt_dias1 = int((date.today() - f9).days)

    f10 = st.time_input("Horário da última Transação", value=time(8, 30))

    cep2 = st.text_area("CEP da última transação (somente números)", value="00000000")
    cep2_limpo = re.sub(r"\D", "", cep2)

    if len(cep2_limpo) == 8:
        cep2_formatado = f"{cep2_limpo[:5]}-{cep2_limpo[5:]}"

    location2 = geolocator.geocode({"postalcode": cep2_formatado, "country": "Brazil"})

    if location1:
        f11 = location2.latitude
        f12 = location2.longitude

    f13 = st.number_input("IP da última transação", value=0.0)
    f14 = st.selectbox("Canal da última compra", chanel_options)

    # Cliente
    f15 = st.number_input("Frequência de Transação (média de transações no dia da semana)", value=0.0)
    f16 = st.number_input("Média de Gasto do Usuário (por transação)", value=0.0)

    cep = st.text_area("CEP do cliente (somente números)", value="00000000")
    cep_limpo = re.sub(r"\D", "", cep)

    if len(cep_limpo) == 8:
        cep_formatado = f"{cep_limpo[:5]}-{cep_limpo[5:]}"

    location = geolocator.geocode({"postalcode": cep_formatado, "country": "Brazil"})

    if location:
        f17 = location.latitude
        f18 = location.longitude

    f19 = st.selectbox("Canal mais frequente de compra", chanel_options)
    f20 = st.number_input("IP mais frequente de compra", value=0.0)

    # Estabelecimento
    f21 = st.selectbox("Tipo de Estabelecimento", estab_options)

    st.write(f"A probabilidade da transação ser fraudulenta é: {dt_dias:.0f}%")
    
    # Variáveis finais
    final0 = 0
    final1 = f1
    final2 = escale(dt_dias, -56.4, 2.5, 0, 30)
    final3 = escale(f3, -72.7, 22.1, 0, 86400)
    final4 = escale(f4, -48, 9.4, -90, 90)
    final5 = escale(f5, -5.7, 16.9, -180, 180)
    final6 = escale(f6, -113.7, 34.8, 0, 999999999999)
    final7 = int(dict_chanels[f7])
    final8 = escale(f8, -43.6, 120.6, 0, 10000)
    final9 = escale(dt_dias1, -73.2, 20, 0, 30)
    final10 = escale(f10, -13.4, 15.6, 0, 86400)
    final11 = escale(f11, -24.6, 23.7, -90, 90)
    final12 = escale(f12, -4.8, 12, -180, 180)
    final13 = escale(f13, -18.7, 7.8, 0, 999999999999)
    final14 = int(dict_chanels[f14])
    final15 = escale(f15, -19.2, 10.5, 0, 12)
    final16 = escale(f16, -4.5, 8.9, 0, 10000)
    final17 = escale(f17, -14.1, 17.3, -90, 90)
    final18 = escale(f18, -25.2, 9.3, -180, 180)
    final19 = int(dict_chanels[f19])
    final20 = escale(f20, -7.2, 5.6, 0, 999999999999)
    final21 = int(dict_chanels[f21])
    final22 = random.uniform(-1.2, 3.4)
    final23 = random.uniform(-1.2, 3.4)
    final24 = random.uniform(-1.2, 3.4)
    final25 = random.uniform(-1.2, 3.4)
    final26 = random.uniform(-1.2, 3.4)
    final27 = random.uniform(-1.2, 3.4)
    final28 = random.uniform(-1.2, 3.4)
    final29 = random.uniform(-1.2, 3.4)
    final30 = 0

    if st.button("Prever"):
        input_df = pd.DataFrame([[final0, final2, final3, final4, final5, final6, final7, final8, final9, final10, final11, final12, final13, final14, final15, final16, final17, final18, final19, final20, final21, final22, final23, final24, final25, final26, final27, final28, final29, final1, final30]], columns=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class'])
        features, labels = treat_columns(input_df)
        prob1 = pipeline.predict_proba(features)[0, 1]
        st.write(f"A probabilidade da transação ser fraudulenta é: {prob1*100:.0f}%")

elif menu == "Busque transações por ID":
    st.subheader("Busque transações por ID")
    id_input = st.text_input("Digite o ID da transação")

    if st.button("Buscar"):
        filt = df_original[df_original['id'] == int(id_input)]
        if not filt.empty:
            valor = filt.iloc[0]['Fraude']
            st.write(f"A transação de ID {id_input} é: {valor}")
        else:
            st.error("ID não encontrado nos resultados.")

elif menu == "Entenda a escolha do modelo":
    st.subheader("Entenda a escolha do modelo")
    st.markdown("""<div style='text-align: justify;'><h5>Neste projeto, temos por objetivo identificar transações fraudulentas feitas com cartões de crédito e, para selecionar o melhor modelo e a metodologia a ser aplicada, inicialmente foram tratados os valores nulos, outliers e erros de grafia.<br><br>
        Como transações fraudulentas costumam ser eventos raros, foi identificado forte desbalanceamento das classes - cerca de 0.17% de positivos no conjunto geral de dados - quatro técnicas de balanceamento foram testadas: SMOTE, ADASYN, Random Over-Sampling e Random Under-Sampling.<br><br>
        Os modelos de classificação avaliados inicialmente foram: <i>LogisticRegression</i>, <i>RandomForestClassifier</i> e <i>XGBClassifier</i>, testados em uma gama de hiperparâmetros definida por meio do <i>GridSearchCV</i>.<br><br>
        Variáveis constantes foram removidas com base em sua variância, utilizando o <i>VarianceThreshold</i>, e as 10 variáveis mais relevantes foram selecionadas por meio do <i>SelectKBest</i>.<br><br>
        O <b>recall</b> foi adotado como métrica principal de avaliação dos modelos, pois, de acordo com os objetivos do problema, é mais importante identificar os casos verdadeiros positivos do que os verdadeiros negativos.<br><br>
        O modelo com melhor desempenho foi o <i>XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=200, random_state=42, subsample=0.8)</i>, com balanceamento RandomOverSampler(random_state=42), atingindo recall de 94% no conjunto de teste. Não houve overfitting.<br><br>
        Considerando o desbalanceamento original dos dados, esse desempenho foi considerado satisfatório, já que, mesmo com menos de 1% de casos positivos, o modelo conseguiu identificar corretamente 94% deles em dados nunca vistos.</h5></div>""", unsafe_allow_html=True)
