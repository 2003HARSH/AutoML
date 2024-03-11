import streamlit as st
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

class Logistic_Regression:
    def __init__(self,df) -> None:
        self.input_col=None
        self.target_col=None
        self.test_size=None
        self.df=df
        self.random_state=None
        self.clf=None
        self.solver=None
        self.penalty=None
        self.max_iter=None

    def set_params(self):
        self.input_col=st.multiselect('**Select input columns**',self.df.columns)
        self.target_col=st.selectbox('**Choose output Column**',self.df.columns)
        total_col=self.input_col+[self.target_col]
        try:
            st.dataframe(self.df[total_col].sample(5))
            self.solver = st.selectbox("solver", options=["lbfgs", "newton-cg", "liblinear", "sag", "saga"])
            if self.solver in ["newton-cg", "lbfgs", "sag"]:
                penalties = ["l2", "none"]
            elif self.solver == "saga":
                penalties = ["l1", "l2", "none", "elasticnet"]
            elif self.solver == "liblinear":
                penalties = ["l1"]
            self.penalty = st.selectbox("penalty", options=penalties)
            self.max_iter = st.number_input("max_iter", 100, 2000, step=50, value=100)

            self.test_size = st.number_input("**Choose testing data size**",value=0.2,min_value=0.1,max_value=0.6)
            self.random_state = st.number_input("**Choose random state**",value=2,min_value=1,max_value=4294967295)
        except: 
            pass

    def fit(self):
        X_train,X_test,y_train,y_test = train_test_split(self.df[self.input_col],self.df[self.target_col],test_size=self.test_size,random_state=self.random_state)
        clf = LogisticRegression(solver=self.solver,penalty=self.penalty,max_iter=self.max_iter)
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

