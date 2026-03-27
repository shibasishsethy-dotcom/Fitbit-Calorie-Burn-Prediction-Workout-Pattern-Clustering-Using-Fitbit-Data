



import streamlit as st
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Calorie Burn ML App", layout="wide")

# -----------------------------
# BACKGROUND STYLE (LIGHT + RELAXED)
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #f5f7fa, #e4ecf7);
}
h1, h2, h3 {
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", [
    "Project Introduction",
    "Calorie Prediction",
    "Workout Clustering",
    "Creator Info"
])

# PAGE 1: INTRODUCTION

if page == "Project Introduction":
    st.title("🔥 Calorie Burn Prediction & Workout Analysis")
    
    st.write("""
    This project combines **Supervised Learning** and **Unsupervised Learning** 
    to analyze workout data and predict calorie burn.

    ### ✅ What this Model does:
    - Predict calories burned using ML models
    - Identify workout patterns using clustering
    - Visualize fitness behavior

    ### 🧠 Models Used:
    - Gradient Boosting
    - Random Forest
    - SVR

    ### 📊 Unsupervised:
    - PCA (Dimensionality Reduction)
    - KMeans Clustering
    -DBSCAN

    This helps in understanding:
    👉 Workout intensity  
    👉 Fitness efficiency  
    👉 User behavior patterns  
    """)


# PAGE 2: SUPERVISED LEARNING

elif page == "Calorie Prediction":

    st.title("🔥 Calorie Burn Prediction")

    import streamlit as st
    import pandas as pd
    import pickle

    model = pickle.load(open("Model used\lasso2.pkl","rb"))
    scaler = pickle.load(open("Model used\lasso2_scaler.pkl","rb"))
    # expected_columns= pickle.load(r"C:\CALORY_BURN_MACHINE_LEARNING_PROJECT\columns.pkl")
    with open(r"C:\CALORY_BURN_MACHINE_LEARNING_PROJECT\columns.pkl", "rb") as f:
        expected_columns = pickle.load(f)







    import streamlit as st

   
    st.write("Enter your workout details below:")

    # -----------------------------
    # USER INPUTS
    # -----------------------------

    col1, col2 = st.columns(2)

    with col1:
        age=st.slider("Age",0,100,20)
        avg_bpm = st.number_input("Average BPM", min_value=20, max_value=220, value=120)
        resting_bpm = st.number_input("Resting BPM", min_value=20, max_value=120, value=70)
        duration = st.number_input("Session Duration (hours)", min_value=0.1, max_value=10.0, value=1.0)

        workout_type = st.selectbox(
            "Workout Type",
            ["Cardio", "Strength", "HIIT", "Yoga","Mixx"]
        )

    with col2:
        fat = st.number_input("Fat Percentage", min_value=5.0, max_value=50.0, value=20.0)
        freq = st.number_input("Workout Frequency (days/week)", min_value=1, max_value=10, value=3)
        bmi = st.number_input("BMI", min_value=1.0, max_value=100.0, value=22.0)
        met = st.number_input("Base MET", min_value=1.0, max_value=100.0, value=5.0)
        hr_intensity = st.number_input("HR Intensity", 0.0, 10.0, 0.1)


    # ENCODING WORKOUT TYPE


    workout_mapping = {
        "Cardio": 0,
        "HIIT": 1,
        "Mixx":2,
        "Strength":3,
        "Yoga":4
    }
    #encoding the workout type so when user choose type it will convert to number for prediction
    workout_encoded = workout_mapping[workout_type]



    if st.button("Predict Calories 🔥"):
        st.subheader("Your input Details")

        input_data = {
            "Age":age,
            "Avg_BPM": avg_bpm,
            "Resting_BPM": resting_bpm,
            "Session_Duration (hours)":duration,
            "Workout_Type":workout_encoded,
            "Fat_Percentage": fat,
            "Workout_Frequency (days/week)":freq,
            "BMI": bmi,
            "Base_MET": met,
            "HR_Intensity": hr_intensity
        }

      

        input_df=pd.DataFrame([input_data])
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col]=0
        input_df=input_df[expected_columns]

        st.write(input_df)

        scaled_input=scaler.transform(input_df)
        
        prediction=model.predict(scaled_input)[0]

        st.success(f"Estimated Calories Burned: {prediction:.2f}")


# PAGE 3: UNSUPERVISED LEARNING

elif page == "Workout Clustering":

    st.title("📊 Workout Pattern Clustering")

    df = pd.read_csv("C:\CALORY_BURN_MACHINE_LEARNING_PROJECT\df1.csv")

    # Preprocessing (same as your notebook)
    # df = df.drop(columns=['Gender','Weight (kg)','Height (m)',
    #                       'Water_Intake (liters)','Effective_MET',
    #                       'Experience_Level','Max_BPM'])

    X = df.drop(columns=['Workout_Type'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_pca)

    score = silhouette_score(X_pca, df['Cluster'])

    st.write(f"### Silhouette Score: {score:.3f}")

    # Plot
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='viridis', ax=ax)

    ax.set_title("Workout Clusters")
    ax.set_xlabel("PCA1: Intensity & Calories")
    ax.set_ylabel("PCA2:Frequency/Duration & BPM")

    st.pyplot(fig)

    # Interpretation
    st.subheader("📌 Cluster Insights")

    st.write("""
    🟣 **Cluster 0**  
    - Low intensity workouts  
    - Moderate heart response  
    - Example: Yoga, light activity 
    - Medium to high Duration 

    🟢 **Cluster 1**  
    - Moderate calorie burn  
    - Controlled heart rate  
    - Example: Strength training  
    - low to Medium Duration
             
    🟡 **Cluster 2**  
    - High intensity  
    - High BPM & calories  
    - Example: HIIT, Cardio 
    -low to high duration
    """)

    st.subheader("Findings")

    st.text("""
            
    🟣Cluster 0 Purple **Left side Located on the left side (low PCA 1) Medium to high PCA 2 means Lower intensity workouts But some moderate heart response Likely low-intensity workouts (e.g., yoga, light activity)

    🟢 Cluster 1 (Green-Bottom side) Located lower side (low PCA 2) Medium PCA 1 Meaning:Moderate calorie burn Low heart rate response

    📌 We can infer that their is Possibly of steady workouts (e.g., strength training, controlled exercise)

    🟡Cluster 2 **(Yellow – Right side) Located right side (high PCA 1) Medium to high PCA 2

    We can say that High intensity High calorie burn + high BPM

    **📌 Interpretation:**Likely high-intensity workouts (HIIT, cardio)
    """)

# =============================
# PAGE 4: CREATOR INFO
# =============================
elif page == "Creator Info":





    st.title("👨‍💻 Project Creator")

    st.markdown("---")

    st.subheader("Calorie Burn Prediction & Workout Analysis")

    st.write("""
    We developed a **machine learning prediction model** that predicts *calorie burn* based on user input.
     K-Means clustering effectively identified 3 distinct user segments based on clustering
             


    """)

    st.markdown("---")

    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👨‍💻 Developer")
        st.write("**Shibasish Sethy**")

        

    with col2:
        st.subheader("🧠 Skills Used")
        st.write("""
        - Python  
        - Pandas, NumPy
        - Scikit-learn
        - Matplotlib, Seaborn
        - Streamlit
 
        """)

    st.markdown("---")

    st.subheader("🛠️ Technologies Used")

    st.write("""
    - **Python** – Data processing & analysis  
    - **Streamlit** – Prediction and Analysis  
    - **Pandas** – Data cleaning  
    - **Matplotlib & Seaborn** – Data visualization  
    """)

    st.markdown("---")

    st.success("✨ Thank you for exploring this project!")





































