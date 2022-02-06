import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("max_columns", 100)
from imblearn import over_sampling
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

rforest_grid = data['model']



def show_predict_page():
    st.title("Resume Callback Prediction")
    st.subheader("Data is taken from Philip Oreopoulos")
    st.write("### Please try to extract/interpret information on your resume \
    to predict whether your resume will receive a callback")

    female = st.selectbox("Sex",options=['Male' , 'Female'])
    race = st.selectbox("Name Ethnicity/Background",options=['White' , 'East Asian', 'South Asian'])
    city = st.selectbox("Application City",options=['Vancouver' , 'Toronto', 'Montreal'])
    skillspeaking = st.slider("Speaking Ability", 1, 100,1)
    skillsocialper = st.slider("Social Ability", 1, 100,1)

    female = 0 if female == 'Male' else 1
    east_asian , south_asian , anglo = 0,0,0
    if race == 'White':
	    anglo = 1
    elif race == 'East Asian':
	    east_asian = 1
    else:
        south_asian = 1
    vancouver, toronto, montreal = 0,0,0
    if city == 'Vancouver':
	    vancouver = 1
    elif city == 'Toronto':
	    toronto = 1
    else:
	    montreal = 1
    
    ok = st.button('Calculate Callback Probability')
    if ok:
        X_check = pd.DataFrame(np.array([[female, skillspeaking, skillsocialper, east_asian, south_asian, anglo, vancouver, toronto, montreal]]),
                   columns=['female', 'skillspeaking', 'skillsocialper', 
                   'east_asian', 'south_asian','anglo', 'vancouver',
                   'toronto', 'montreal'])

        X_check = X_check.astype(float)
        y_check = rforest_grid.predict(X_check)
        y_pred_prob = rforest_grid.predict_proba(X_check)
        if y_check == 1:
	        st.subheader('Resume would have received a callback with a probability of {}%'.format(round(y_pred_prob[0][1]*100 , 3)))
        else:
	        st.subheader('Resume would not have received a callback with a probability of {}%'.format(round(y_pred_prob[0][0]*100 , 3)))
