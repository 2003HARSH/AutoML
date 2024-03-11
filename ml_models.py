import streamlit as st
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class Linear_Regression:
    def __init__(self,input_col,target,test_size,df,random_state) -> None:
        self.input_col=input_col
        self.target=target
        self.test_size=test_size
        self.df=df
        self.random_state=random_state
        self.lr=None
    
    def fit(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
        X_train,X_test,y_train,y_test = train_test_split(self.df[self.input_col],self.df[self.target],test_size=self.test_size,random_state=self.random_state)
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
        

class Logistic_Regression:
    def __init__(self,input_col,target,test_size,df,random_state) -> None:
        self.input_col=input_col
        self.target=target
        self.test_size=test_size
        self.df=df
        self.random_state=random_state
        self.clf=None
    
    def fit(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score,confusion_matrix
        X_train,X_test,y_train,y_test = train_test_split(self.df[self.input_col],self.df[self.target],test_size=self.test_size,random_state=self.random_state)
        clf = LogisticRegression()
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        st.header('Accuracy Score')
        st.text(accuracy_score(y_test,y_pred))
        st.header('Confusion Matrix')
        st.dataframe(pd.DataFrame(confusion_matrix(y_test,y_pred)))
        self.clf=clf
    
    def predict(self,input_col):
        pred_input=[]
        for i in range(0,len(input_col)):
            pred_input.append(st.text_input(input_col[i]))
        try:
            new_input=[float(i) for i in pred_input]
            new_input=np.array([pred_input],dtype=np.float64)
            return self.clf.predict(new_input)[0]   
        except:
            return None
