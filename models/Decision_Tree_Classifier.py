import streamlit as st
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix



class Decision_Tree_Classifier:
    def __init__(self,df) -> None:
        self.input_col=None
        self.target_col=None
        self.test_size=None
        self.df=df
        self.random_state=None
        self.clf=None
        self.criterian=None
        self.splitter=None
        self.min_samples_leaf=None
        self.min_samples_split=None
        self.min_weight_fraction_leaf=None
        self.max_leaf_nodes=None
        self.max_features=None
        self.min_impurity_decrease=None
        self.ccp_alpha=None
        self.max_depth=None
        

    def set_params(self):
        self.input_col=st.multiselect('**Select input columns**',self.df.columns)
        self.target_col=st.selectbox('**Choose output Column**',self.df.columns)
        total_col=self.input_col+[self.target_col]
        try:
            st.dataframe(self.df[total_col].sample(5))
            self.criterian=st.selectbox('criterian',options=['gini','entropy','log_loss'])
            self.splitter=st.selectbox('splitter',options=['best','random'])
            var=st.number_input("max_depth (0 Signifies None)", 0, 500)
            if var==0:
                self.max_depth=None
            else:
                self.max_depth=var

            self.min_samples_split = st.number_input("min_samples_split", 1, 100, value=2)
            self.min_samples_leaf = st.number_input("min_samples_leaf", 1, 100, value=1)
            self.min_weight_fraction_leaf = st.number_input("min_weight_fraction_leaf",0.0,0.5,step=0.,format="%0.2f")
            self.max_features=st.selectbox('max_features',options=[None,'sqrt','log2'])
            self.min_impurity_decrease = st.number_input("min_impurity_decrease",step=1.,format="%.2f")
            self.ccp_alpha = st.number_input("ccp_alpha",step=1.,format="%.2f")
            
            self.test_size = st.number_input("**Choose testing data size**",value=0.2,min_value=0.1,max_value=0.6)
            self.random_state = st.number_input("**Choose random state**",value=2,min_value=1,max_value=4294967295)
        except: 
            pass

    def fit(self):
        X_train,X_test,y_train,y_test = train_test_split(self.df[self.input_col],self.df[self.target_col],test_size=self.test_size,random_state=self.random_state)
        clf = DecisionTreeClassifier(criterion=self.criterian,splitter=self.splitter,max_depth=self.max_depth,min_samples_split=self.min_samples_split,min_samples_leaf=self.min_samples_leaf,min_weight_fraction_leaf=self.min_weight_fraction_leaf,max_features=self.max_features,min_impurity_decrease=self.min_impurity_decrease,ccp_alpha=self.ccp_alpha)
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

