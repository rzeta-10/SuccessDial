# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.compose import make_column_selector as selector

# 1. Data Loading
# Load the dataset
data = pd.read_csv('bank_data.csv')  # Replace with the actual file path

# Display first few rows of the dataset
print(data.head())

# Checking for null values and data types
print(data.info())
print(data.isnull().sum())

# 2. Exploratory Data Analysis (EDA)
# Basic Descriptive Statistics
print(data.describe())

# Checking the class distribution of the target variable
print(data['target'].value_counts())

# Correlation Analysis
corr_matrix = data.corr()
print(corr_matrix)

# 3. Data Visualization
# Univariate Analysis: Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# Univariate Analysis: Job distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='job', data=data)
plt.title('Job Distribution')
plt.xticks(rotation=45)
plt.show()

# Bivariate Analysis: Duration vs. Balance
plt.figure(figsize=(10, 6))
sns.scatterplot(x='balance', y='duration', hue='target', data=data)
plt.title('Balance vs Duration')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 4. Statistical Analysis
# Chi-square test for categorical variables
from scipy.stats import chi2_contingency

categorical_features = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'poutcome']
for feature in categorical_features:
    contingency_table = pd.crosstab(data[feature], data['target'])
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    print(f'Chi-square test for {feature} vs Target: p-value={p}')

# 5. Train-Validation Split
# Splitting the data
X = data.drop('target', axis=1)
y = data['target'].apply(lambda x: 1 if x == 'yes' else 0)  # Encode target as 0 or 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Data Cleaning/Preprocessing
# Column transformer for preprocessing
numeric_features = selector(dtype_include=np.number)
categorical_features = selector(dtype_include=object)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 7. Building a Baseline Model
# Logistic Regression
logreg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression())])

# Train the model
logreg_pipeline.fit(X_train, y_train)

# Predict on the test data
y_pred_logreg = logreg_pipeline.predict(X_test)

# Evaluate the model
print("Logistic Regression Model Performance:")
print(classification_report(y_test, y_pred_logreg))
print(f"Accuracy: {accuracy_score(y_test, y_pred_logreg)}")

# Decision Tree Classifier
tree_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', DecisionTreeClassifier(random_state=42))])

# Train the model
tree_pipeline.fit(X_train, y_train)

# Predict on the test data
y_pred_tree = tree_pipeline.predict(X_test)

# Evaluate the model
print("\nDecision Tree Model Performance:")
print(classification_report(y_test, y_pred_tree))
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree)}")

# Confusion Matrix
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_logreg), annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_tree), annot=True, fmt='d', cmap='Blues')
plt.title('Decision Tree Confusion Matrix')

plt.show()
