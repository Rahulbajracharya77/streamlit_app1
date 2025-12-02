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
# Streamlit Page Setup
# ------------------------------------------------------
st.set_page_config(page_title="Drug Classification App", layout="wide")
st.title("💊 Drug Classification ML App")
st.write("Upload drug200.csv to explore data & build ML models.")

# ------------------------------------------------------
# File Upload
# ------------------------------------------------------
uploaded_file = st.file_uploader("Upload drug200.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------------
    # Convert categorical columns
    # -------------------------------------
    cat_cols = ['Sex', 'BP', 'Cholesterol', 'Drug']
    df[cat_cols] = df[cat_cols].astype("category")

    # -------------------------------------
    # Plots
    # -------------------------------------
    st.subheader("📊 Scatterplot")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=df, x='Na_to_K', y='BP', hue='Drug', marker='*')
    st.pyplot(fig1)

    st.subheader("📦 Boxplot")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x='Drug', y='Na_to_K')
    st.pyplot(fig2)

    st.subheader("🔥 Correlation Heatmap")
    fig3, ax3 = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True)
    st.pyplot(fig3)

    # -------------------------------------
    # ML PREPARATION
    # -------------------------------------
    st.subheader("⚙️ ML Model Training")

    y = df["Drug"]
    X = df.drop(columns=["Drug"])

    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=15
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC()
    }

    scores = {}

    # Train button
    if st.button("Train Models"):
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            scores[name] = {
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, average='weighted'),
                "Recall": recall_score(y_test, pred, average='weighted'),
                "F1-score": f1_score(y_test, pred, average='weighted'),
            }

        st.subheader("📈 Model Performance")
        st.write(pd.DataFrame(scores).T.style.highlight_max(axis=0))
else:
    st.info("Please upload the drug200.csv file to continue.")
