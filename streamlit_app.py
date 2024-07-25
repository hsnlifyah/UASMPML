import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Setting up Streamlit
st.title('Online Foods Dataset Analysis and Model Training')

# Load dataset
@st.cache
def load_data():
    return pd.read_csv('onlinefoods.csv')

df = load_data()

# Display dataset info
if st.sidebar.checkbox('Show dataset info'):
    st.write(df.info())
    st.write(df.describe())

# Check for missing values
if st.sidebar.checkbox('Show missing values'):
    st.write(df.isnull().sum())

# Visualize some key features
if st.sidebar.checkbox('Visualize Marital Status'):
    fig, ax = plt.subplots()
    sns.countplot(x='Marital Status', data=df, ax=ax)
    ax.set_title('Count of Marital Status')
    st.pyplot(fig)

if st.sidebar.checkbox('Visualize Monthly Income by Age'):
    fig, ax = plt.subplots()
    sns.barplot(x='Monthly Income', y='Age', data=df, ax=ax)
    ax.set_title('Monthly Income by Age')
    st.pyplot(fig)

# Select only numeric columns for correlation matrix
if st.sidebar.checkbox('Show Correlation Matrix'):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    fig, ax = plt.subplots()
    sns.heatmap(df[numeric_columns].corr(), annot=True, ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)

# Handling missing values
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])

# Encoding categorical variables
categorical_features = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Feedback']
numerical_features = ['Age', 'Family size', 'latitude', 'longitude']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Splitting the dataset into training and testing sets
X = df.drop('Output', axis=1)
y = df['Output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Train and evaluate models
models = {
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'{name} Accuracy: {accuracy:.2f}')
    st.text(classification_report(y_test, y_pred))

# Hyperparameter Tuning for Random Forest
if st.sidebar.checkbox('Perform Hyperparameter Tuning for Random Forest'):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    st.write(f'Best Parameters: {grid_search.best_params_}')
    best_rf_model = grid_search.best_estimator_

    # Re-evaluate the tuned model
    y_pred_tuned = best_rf_model.predict(X_test)
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    st.write(f'Tuned Random Forest Accuracy: {accuracy_tuned:.2f}')
    st.text(classification_report(y_test, y_pred_tuned))
