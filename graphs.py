import streamlit as st
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import plot_tree


def decision_boundry(model):
    try:
        columns=st.multiselect('**Select any 2 columns in the order of previous input columns** (preferred)',model.input_col)    
        plot_decision_regions(model.df[columns].values,model.df[model.target_col].values,model.clf, legend=2)
        #add filler values for multifearture data
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    except:
        pass

def tree_plot(model):
    try:
        plot_tree(model.clf,filled=True,class_names=model.target_col,feature_names=model.input_col)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    except:
        pass




    
