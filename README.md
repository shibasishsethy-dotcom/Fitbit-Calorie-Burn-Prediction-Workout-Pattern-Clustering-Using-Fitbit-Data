# Fitbit-Calorie-Burn-Prediction-Workout-Pattern-Clustering-Using-Fitbit-Data

# 🔥 Calorie Burn Prediction & Workout Analysis

This project combines **Supervised Learning** and **Unsupervised Learning** to analyze workout data and predict calorie burn using Machine Learning.

-------------------------------------------------------------------------------------

## 🚀 Project Overview

This application helps to:

✅ Predict calories burned based on workout data  
✅ Identify workout behavior patterns using clustering  
✅ Understand user fitness levels and workout intensity  

-----------------------------------------------------------------------------------

## 🧠 Machine Learning Techniques Used

### 🔹 Supervised Learning
- Linear Regression
- Ridge & Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost
- SVR

📌 **Best Performance:**
- Lasso →R2 Score: 0.93

-------------------------------------------------------------------

### 🔹 Unsupervised Learning
- PCA (Dimensionality Reduction)
- K-Means Clustering
- DBSCAN (for outlier detection)

---

## 📊 Data Preprocessing

- Removed irrelevant features (e.g., Weight, Height, Water Intake)
- Handled multicollinearity (removed highly correlated features)
- Encoded categorical variables (Workout Type, Gender)
- Applied scaling using StandardScaler
- Applied Box-Cox transformation for normalization

---------------------------------------------------------------------------------------------

## 📉 PCA Interpretation

- **PCA 1 → Workout Intensity**
  - Influenced by Calories Burned, HR Intensity, MET

- **PCA 2 → Workout Consistency / Fitness Efficiency**
  - Influenced by Duration, Frequency, BPM

--------------------------------------------------------------------------------------------------

## 📊 Clustering Insights (K-Means)

### 🔥 3-Cluster Model (Best Model — Silhouette Score: 0.388)

🟣 **Cluster 0 (Low Intensity)**
- Low calorie burn  
- Light workouts (Yoga, casual activity)  
- Beginner-level users  

🟢 **Cluster 1 (Moderate / Controlled)**
- Moderate calorie burn  
- Stable heart rate  
- Strength or controlled workouts  

🟡 **Cluster 2 (High Intensity)**
- High calorie burn  
- High BPM  
- HIIT / Cardio users  

---

### 🔥 4-Cluster Model (More Detailed Segmentation)

🟣 Low-intensity users  
🔵 High-intensity users 
🟢 Consistent users  
🟡 Irregular users 

📌 However, this model had a slightly lower silhouette score (0.372), meaning less clear separation.

--------------------------------------------------------------------------------------------------

## 📌 Key Insight (VERY IMPORTANT)

- Users are mainly differentiated by:
  - Workout Intensity  
  - Workout Consistency  

- Behavior is **continuous**, not sharply separated.

--------------------------------------------------------------------------------------------------------------

## 🔍 DBSCAN Insight

- DBSCAN produced:
  - **1 major cluster**
  - Few outliers (-1)

📌 Interpretation:
- Data does not have strong density-based clusters  
- Workout behavior changes gradually  
- No clear “dense groups” exist  

✅ **Usefulness of DBSCAN:**
- Identifies **outliers**
- Detects **abnormal workout patterns**

------------------------------------------------------------------------------------

## 🎯 Final Conclusion

- **K-Means is more suitable** for this dataset  
  → Because it separates users based on distance  

- **DBSCAN is useful for anomaly detection**  
  → Helps find unusual or extreme workout behavior  

------------------------------------------------------------------------------------

## 💻 Streamlit App Features

- 🔥 Calorie Prediction (User Input Based)
- 📊 Workout Clustering Visualization
- 📈 PCA-based analysis
- 🎯 User-friendly interface

----------------------------------------------------------------------------------------------

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit

----------------------------------------------------------------------------------------

## 👨‍💻 Creator

**Shibasish Sethy**

-

------------------------------------------------------------------------------------------

## 🎯 Goal

To build intelligent systems that can:
- Analyze user behavior  
- Predict outcomes  
- Provide meaningful insights  

---

## ⭐ Key Learning

- Difference between **distance-based vs density-based clustering**
- Importance of **feature engineering**
- Real-world interpretation of ML models

---
