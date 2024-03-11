import streamlit as st
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

class Linear_Regression:
    def __init__(self,df) -> None:
        self.input_col=None
        self.target_col=None
        self.test_size=None
        self.df=df
        self.random_state=None
        self.lr=None
    
    def set_params(self):
        self.input_col=st.multiselect('**Select input columns**',self.df.columns)
        self.target_col=st.selectbox('**Choose output Column**',self.df.columns)
        total_col=self.input_col+[self.target_col]
        try:
            st.dataframe(self.df[total_col].sample(5))
            self.test_size = st.number_input("**Choose testing data size**",value=0.2,min_value=0.1,max_value=0.6)
            self.random_state = st.number_input("**Choose random state**",value=2,min_value=1,max_value=4294967295)
        except:
            pass

    def fit(self):
        X_train,X_test,y_train,y_test = train_test_split(self.df[self.input_col],self.df[self.target_col],test_size=self.test_size,random_state=self.random_state)
        lr=LinearRegression()
        lr.fit(X_train,y_train)
        st.header("lr.coef_")
        st.text(lr.coef_)
        st.header("lr.intercept_")
        st.text(lr.intercept_ )
        y_pred = lr.predict(X_test)
        st.header("Mean Absolute Error (MAE)")
        st.text(mean_absolute_error(y_test,y_pred))
        st.header("Mean Squared Error (MAE)")
        st.text(mean_squared_error(y_test,y_pred))
        st.header("R2 Score")
        st.text(r2_score(y_test,y_pred))
        self.lr=lr

    def predict(self,input_col):
        pred_input=[]
        for i in range(0,len(input_col)):
            pred_input.append(st.text_input(input_col[i]))
        try:
            new_input=[float(i) for i in pred_input]
            new_input=np.array([pred_input],dtype=np.float64)
            return self.lr.predict(new_input)[0]   
        except:
            return None
        

