import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Rock vs Mine Detection",
    page_icon="🎯",
    layout="centered"
)

# ========== LOAD DATA & MODEL ==========
# Load dataset without header
df = pd.read_csv("sonar_data.csv", header=None)
df.columns = [f"Feature_{i}" for i in range(df.shape[1] - 1)] + ["Class"]

# Load model
with open("rock_mine_model.pkl", "rb") as file:
    model = pickle.load(file)

accuracy = 0.84  # Example accuracy

# ========== HEADER ==========
st.title("🎯 Rock vs Mine Detection App")
st.markdown("""
This app uses **Logistic Regression** to classify whether a sonar signal is from a **Rock** or a **Mine**.  
Dataset: [Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+sonar)

**Applications:**
- Naval mine detection
- Geological surveys
- Underwater object classification
""")

# ========== DATA VISUALIZATION ==========
st.subheader("📊 Dataset Overview")
fig, ax = plt.subplots(figsize=(5, 3))
sns.countplot(x="Class", data=df, palette="Set2", ax=ax)
ax.set_title("Class Distribution: Rock vs Mine")
st.pyplot(fig)

st.markdown("""
**Why Logistic Regression?**
- Simple & interpretable
- Works well for binary classification
- Handles high-dimensional data effectively
""")

# ========== PREDICTION SECTION ==========
st.subheader("🔍 Test the Model")

# Input
if "input_str" not in st.session_state:
    st.session_state["input_str"] = ""

input_str = st.text_area("Enter **60 comma-separated values**:", st.session_state["input_str"])

# Fill random values
if st.button("🎲 Fill with Random Values"):
    random_values = np.random.uniform(0, 1, 60)
    input_str = ",".join([f"{v:.3f}" for v in random_values])
    st.session_state["input_str"] = input_str
    st.rerun()

# Show current input
if input_str.strip():
    st.code(input_str, language="text")

# Prediction
if st.button("🚀 Predict"):
    try:
        values = [float(x) for x in input_str.split(",")]
        if len(values) != 60:
            st.error("⚠ Please enter exactly 60 numeric values.")
        else:
            prediction = model.predict(np.array(values).reshape(1, -1))[0]
            label = "Rock 🪨" if prediction == "R" else "Mine 💣"
            st.success(f"**Prediction:** {label}")
            st.info(f"Model Accuracy: **{accuracy*100:.2f}%**")
    except ValueError:
        st.error("❌ Invalid input. Ensure all values are numbers.")

# ========== FUTURE SCOPE ==========
st.subheader("🚀 Future Scope & Benefits")
st.markdown("""
- **Real-Time Naval Defense**: Detect and classify mines instantly to prevent naval disasters.
- **Marine Research**: Aid geological surveys by identifying underwater rock formations.
- **Autonomous Submarines**: Integrate with AI-powered subs for underwater exploration.

**Credits**: Developed by Kanav Chauhan  
""")
