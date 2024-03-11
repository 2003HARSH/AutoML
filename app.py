import streamlit as st
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from models.Linear_Regression import Linear_Regression
from models.Logistic_Regression import Logistic_Regression
from models.Decision_Tree_Classifier import Decision_Tree_Classifier
from models.Desicion_Tree_Regressor import Decision_Tree_Regressor 
import graphs

st.set_page_config(
        page_title="AutoML",
)

if 'fit_clicked' not in st.session_state:
    st.session_state['fit_clicked']=False

st.header('Welcome to Automated ML')

with st.sidebar.header('MLDLC'):
    file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
with st.sidebar:
    model=st.selectbox('2. Choose your model',['Select','Linear Regression','Logistic Regression','Decision Tree Classifier','Decision Tree Regressor'])
    
if file is not None:
    df=pd.read_csv(file)
    if st.button('Show EDA'):
        profile = ProfileReport(df,explorative=True)
        st.title("EDA")
        st_profile_report(profile)
    if st.button('Hide EDA'):
        st.write(unsafe_allow_html=True)
    st.header('**DataFrame**')
    st.dataframe(df.head())

def callback():
    st.session_state['fit_clicked']=True

def show_output_regression(model):
    if st.button('Fit the model',on_click=callback) or st.session_state['fit_clicked']:
        model.fit()
        st.header('Predict')
        st.text('The predicted value is '+ str(model.predict(model.input_col)))
        if type(model).__name__=='Decision_Tree_Regressor':
            graphs.tree_plot(model)

def show_output_classification(model):
    if st.button('Fit the model',on_click=callback) or st.session_state['fit_clicked']:
        model.fit()
        st.header('Predict')
        st.text('The predicted value is '+ str(model.predict(model.input_col)))
        if type(model).__name__=='Decision_Tree_Classifier':
            graphs.tree_plot(model)
        graphs.decision_boundry(model)

if model !='Select':
    if model == 'Linear Regression':
        model=Linear_Regression(df)
        model.set_params()
        show_output_regression(model)

    elif model=='Logistic Regression':
        model=Logistic_Regression(df)
        model.set_params()
        show_output_classification(model)

    elif model=='Decision Tree Classifier':
        model=Decision_Tree_Classifier(df)
        model.set_params()
        show_output_classification(model)

    elif model=='Decision Tree Regressor':
        model=Decision_Tree_Regressor(df)
        model.set_params()
        show_output_regression(model)
     
        
        




    

    



