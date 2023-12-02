import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from summarytools import dfSummary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



path = "data.csv"
df = pd.read_csv(path)

st.title("Machine Learning Breast Cancer Analysis")

df['diagnosis'] = df['diagnosis'].apply(lambda val: 1 if val == 'M' else 0)

st.write("### Displaying the first 5 rows of the DataFrame:")
st.dataframe(df.head(5))

st.write("### Dropping columns 'id' and 'Unnamed: 32':")
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
st.dataframe(df.head(15))

st.write("### Describing the dataset:")
st.dataframe(df.describe())


st.write("### Checking for null values")
st.dataframe(df.isna().sum())


st.write("### Boxplot of Radius mean and Diagnosis:")
boxplot_fig, ax = plt.subplots()
sns.boxplot(x='diagnosis', y='radius_mean', hue='diagnosis', data=df, ax=ax)
st.pyplot(boxplot_fig)



st.write("### Scatter Plot of Area mean and Smoothness mean by diagnosis:")
scatter_fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x='area_mean', y='smoothness_mean', hue='diagnosis', data=df, ax=ax)
st.pyplot(scatter_fig)


st.write("### Correlation Values:")
correlation = df.corr()['diagnosis'].apply(abs).sort_values(ascending=False)
st.write(correlation)



st.write("### Machine Learning Results:")


# Classification Task
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    'Logistic Regression': LogisticRegression(),
    'k-Nearest Neighbors': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Results
results = {}

for model_name, model in models.items():
    # Fit and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Save results
    results[model_name] = {'Accuracy': accuracy, 'Classification Report': report}

    # Display results
    st.write(f"#### {model_name}")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write("Classification Report:\n", report)

# Display results in a table
st.write("### Results Table:")
results_table = pd.DataFrame(results).transpose()
st.table(results_table)