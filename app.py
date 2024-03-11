import streamlit as st
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from ml_models import Linear_Regression,Logistic_Regression

if 'fit_clicked' not in st.session_state:
    st.session_state['fit_clicked']=False


st.header('Welcome to Automated ML')



with st.sidebar.header('MLDLC'):
    file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
with st.sidebar:
    model=st.selectbox('2. Choose your model',['Select','Linear Regression','Logistic Regression'])
    


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

if model !='Select':
    if model == 'Linear Regression':
        input_col=st.multiselect('**Select input columns**',df.columns)
        target_col=st.selectbox('**Choose output Column**',df.columns)
        total_col=input_col+[target_col]
        st.dataframe(df[total_col].head())
        test_size = st.number_input("**Choose testing data size**",value=0.2,min_value=0.1,max_value=0.6)
        random_state = st.number_input("**Choose random state**",value=2,min_value=1,max_value=4294967295)
        l_r=Linear_Regression(input_col,target_col,test_size,df,random_state)
        if st.button('Fit the model',on_click=callback) or st.session_state['fit_clicked']:
            l_r.fit()
            st.header('Predict')
            st.text('The predicted value is '+ str(l_r.predict(input_col)))
    

    elif model=='Logistic Regression':
        input_col=st.multiselect('**Select input columns**',df.columns)
        target_col=st.selectbox('**Choose output Column**',df.columns)
        total_col=input_col+[target_col]
        st.dataframe(df[total_col].head())
        test_size = st.number_input("**Choose testing data size**",value=0.2,min_value=0.1,max_value=0.6)
        random_state = st.number_input("**Choose random state**",value=2,min_value=1,max_value=4294967295)
        l_r=Logistic_Regression(input_col,target_col,test_size,df,random_state)
        if st.button('Fit the model',on_click=callback) or st.session_state['fit_clicked']:
            l_r.fit()
            st.header('Predict')
            st.text('The predicted value is '+ str(l_r.predict(input_col)))
        




    

    



