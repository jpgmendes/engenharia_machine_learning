import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fname = './modelo_kobe.pkl'
operation_file = './Data/processed/data_operation.parquet'
train_file = './Data/operalization/base_train.parquet'
test_file = './Data/operalization/base_test.parquet'

############################################ SIDE BAR TITLE
st.sidebar.title('Control Panel')
st.sidebar.markdown(f"""
Control of Kobe Bryant converted shots and variables input to new data evaluation.
""")

############################################ LEITURA DOS DADOS
@st.cache(allow_output_mutation=True)
def load_model(model):
    return joblib.load(model)

@st.cache(allow_output_mutation=True)
def load_data(data):
    return pd.read_parquet(data)


model = load_model(fname)
top_data = load_data(operation_file)
train_data = load_data(train_file)
test_data = load_data(test_file)
features = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']

############################################ TITULO
st.title(f"""
Online system of Kobe Bryant shots.
""")

st.markdown(f"""
This interface can be used to explanation of results classified by
the model considering the variables used to characterize the shots. 
The selected model consumed a total of {train_data.shape[0]} shots.
The variables used in the model input are {features}
""")

############################################ ENTRADA DE VARIAVEIS
st.sidebar.header('Variables input')
form = st.sidebar.form("input_form")
input_variables = {}
for cname in features:
    input_variables[cname] = form.slider(cname.capitalize(),
                                   train_data[cname].min(),
                                   train_data[cname].max())
form.form_submit_button("Evaluate")



############################################ PREVISAO DO MODELO 
@st.cache
def predict_user(input_variables):
    X = pd.DataFrame([input_variables])
    return {
        'Probability': model.predict_proba(X)[0, 1],
        'Classification': model.predict(X)[0]
    }

user_shot = predict_user(input_variables)

if user_shot['Classification'] == 0:
    st.sidebar.markdown("""Classification:
    <span style="color:red">*Non converted* shot</span>.
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""Classification:
    <span style="color:green">*Converted* shot</span>.
    """, unsafe_allow_html=True)

############################################ PAINEL COM AS PREVISOES HISTORICAS
train_data['prob'] = model.predict_proba(train_data[features])[:,0]
train_data['operation_label'] = model.predict(train_data[features])
test_data['prob'] = model.predict_proba(test_data[features])[:,0]
test_data['operation_label'] = model.predict(test_data[features])


fignum = plt.figure(figsize=(10,6))
sns.distplot(train_data[train_data['shot_made_flag'] == 0].prob, ax = plt.gca(), label=['Miss'])
sns.distplot(train_data[train_data['shot_made_flag'] == 1].prob, ax = plt.gca(), label=['Goal'])

# User wine
plt.axvline(user_shot['Probability'], color='k', linewidth=2, linestyle=':', label='Selected shot')

plt.title('Historical shots')
plt.ylabel('Estimated density')
plt.xlabel('Probability of goal')
plt.xlim((0,1))
plt.grid(True)
plt.legend(loc='best')
st.pyplot(fignum)

