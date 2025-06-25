import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler

def treat_columns(df_train):
  df = df_train.copy()

  for col in df.select_dtypes(include='float64').columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1 * IQR
    upper = Q3 + 1 * IQR
    df[col] = df[col].clip(lower, upper)  # Limita valores fora do intervalo

  numeric = df.select_dtypes(include='float64').columns
  scaler = StandardScaler()
  df[numeric] = scaler.fit_transform(df[numeric])

  train_features = df.drop(columns=['Class'])
  train_labels = df['Class']
  
  df['ID'] = range(1, len(df) + 1)
  enrollee_ids = df['ID']

  return train_features, train_labels, enrollee_ids

# Carregar modelo e dados
pipeline = joblib.load('final_model.pkl')
df_train = pd.read_parquet('creditcard.parquet')

# Processar
features, labels, enrollee_ids = treat_columns(df_train)
probs = pipeline.predict_proba(features)[:, 1]
features_cols = features.columns

# Juntar resultado
resultados = pd.DataFrame({
    'ID transação': enrollee_ids.values,
    'Probabilidade': probs
})

st.set_page_config(layout="wide")
st.title("Identificação de Fraudes com Cartão de Crédito")

menu = st.sidebar.selectbox("Escolha uma opção", [
    "Entenda os dados",
    "Busque transações por ID",
    "Entenda a escolha do modelo"
])

if menu == "Entenda os dados":
    st.subheader("Entenda os dados")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div style='text-align: justify'><h5>Neste projeto, temos por objetivo determinar transações fraudulentas feitas com cartão de crédito, a partir de um banco de dados de quase 300 mil transações, descritas por meio de 28 variáveis transformadas e pelo seu valor. <br><br> Por se tratarem de dados confidenciais, não trabalhamos com as features originais e sim transformadas. <br><br> A partir delas, foi possível determinar o modelo de Classificação Binária, que possibilitou o cálculo da probabilidade de uma transação ser ou não fraudulenta. <br><br> Em casos como esse, é de interesse principal a detecção de fraude, mesmo que em detrimento de um maior número de falsos positivos, pois entende-se que, sinalizada a transação como uma possível fraude, outros mecanismos de segurança podem ser utilizados em paralelo para confirmar não prejudicar o cliente. <br><br> O modelo escolhido foi o XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=200, random_state=42, subsample=0.8), que mostrou recall de 94% em dados de teste, valor considerado alto, já que os dados são altamente desbalanceados - com apenas 0.17% de positivos.</h5></div>", unsafe_allow_html=True)

elif menu == "Busque transações por ID":
    st.subheader("Busque transações por ID")
    id_input = st.text_input("Digite o ID da transação")

    if st.button("Buscar"):
        filt = resultados[resultados['ID transação'] == int(id_input)]
        if not filt.empty:
            prob2 = filt.iloc[0]['Probabilidade']
            st.write(f"A probabilidade da transação de ID {id_input} ser fraudulenta, é: {prob2*100:.0f}%")
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
