import streamlit as st 
import pandas as pd
import joblib 

# import the model with joblib 
model = joblib.load("Notebook/model.joblib") 

# species prediction function 
def species_predict (sep_len, sep_wid, pet_len, pet_wid): 
    d = [
        {
            "sepal_length": sep_len,
            "sepal_width": sep_wid,
            "petal_length": pet_len, 
            "petal_width": pet_wid
        }
    ] 
    Xnew = pd.DataFrame(d) 
    pred = model.predict(Xnew) 
    probs = model.predict_proba(Xnew) 
    probs_dct = {}
    species = model.classes_
    for s, p in zip(species, probs.flatten()): 
        probs_dct[s] = float(p) 

        
    return pred, probs_dct 


# start creating the streamlit app 
st.set_page_config(page_title ="Iris Project") 

# Adding the title to the webpage 

st.title ("Iris End to End project") 
st.subheader("By Harshada Erande") 

# Take user input
sep_len = st.number_input("sepal length", min_value = 0.00, step= 0.01)
sep_wed = st.number_input("sepal width", min_value = 0.00, step= 0.01)
pet_len = st.number_input("petal length", min_value = 0.00, step= 0.01)
pet_wed = st.number_input("petal width", min_value = 0.00, step= 0.01) 

# Create a button to predict
button = st.button ("predict", type="primary") 

# If button is click 
if button: 
    pred, probs = species_predict(sep_len, sep_wed, pet_len, pet_wed,)
    st.subheader(f"prediction : {pred}") 
    for  s, p  in probs.items():
         st.subheader(f"{s} : probability {p}") 
         st.progress (p)
 