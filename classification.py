import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ------------------------------------------------------
# Load Dataset
# ------------------------------------------------------
df = pd.read_csv("drug200.csv")

st.title("💊 Drug Classification ML App")
st.write("Exploring drug200.csv and building ML models.")

# ------------------------------------------------------
# Dataset Preview
# ------------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Convert categorical columns
cat_cols = ["Sex", "BP", "Cholesterol", "Drug"]
df[cat_cols] = df[cat_cols].astype("category")

# ------------------------------------------------------
# Plots
# ------------------------------------------------------
st.subheader("📊 Scatterplot")
fig1, ax1 = plt.subplots()
sns.scatterplot(data=df, x="Na_to_K", y="BP", hue="Drug", marker="*")
st.pyplot(fig1)

st.subheader("📦 Boxplot")
fig2, ax2 = plt.subplots()
sns.boxplot(data=df, x="Drug", y="Na_to_K")
st.pyplot(fig2)

st.subheader("🔥 Correlation Heatmap")
fig3, ax3 = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True)
st.pyplot(fig3)

# ------------------------------------------------------
# ML Preparation
# ------------------------------------------------------
st.subheader("⚙️ ML Model Training")

y = df["Drug"]
X = df.drop(columns=["Drug"])

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=15
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ML Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
}

scores = {}

# Train Models
if st.button("Train Models"):
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        scores[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted"),
            "Recall": recall_score(y_test, pred, average="weighted"),
            "F1-score": f1_score(y_test, pred, average="weighted"),
        }

    st.subheader("📈 Model Performance")
    st.write(pd.DataFrame(scores).T.style.highlight_max(axis=0))


# ------------------------------------------------------
# User Input Prediction (Dynamic)
# ------------------------------------------------------
st.subheader("🔮 Predict Drug for User Input")

# Let user choose the trained model
model_choice = st.selectbox("Choose Model for Prediction", list(models.keys()))

model = models[model_choice]
model.fit(X_train_scaled, y_train)

# Dynamic Input Form
st.write("### Enter Patient Details")

input_data = {}

for col in df.columns:
    if col == "Drug":
        continue
    
    if df[col].dtype.name == "category":  # categorical columns
        input_data[col] = st.selectbox(f"{col}", df[col].cat.categories)
    else:  # numeric columns
        input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))

# Convert input into DataFrame
input_df = pd.DataFrame([input_data])

# One-hot encode input
input_df = pd.get_dummies(input_df)

# Align with training columns
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Scale
input_scaled = scaler.transform(input_df)

# Predict button
if st.button("Predict Drug"):
    prediction = model.predict(input_scaled)
    st.success(f"### Predicted Drug: **{prediction[0]}**")
