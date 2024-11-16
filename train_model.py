import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from keras.wrappers.scikit_learn import K
# Load the dataset
file_path = './data/dataset1.csv'  # Update to the correct file path
df = pd.read_csv(file_path)

# Set index and drop unnecessary columns
df.set_index('CENSEOID', inplace=True)
df.drop(['CLIENTID', 'CLIENT'], axis=1, inplace=True)

# Check distribution of target variable
print(df['V28HCCCODED'].value_counts())

# Ensure there are at least two classes in the target
if df['V28HCCCODED'].nunique() <= 1:
    raise ValueError(
        "The dataset contains only one class. Add more diverse samples.")

# Separate features and target variable
X = df.drop('V28HCCCODED', axis=1)
y = df['V28HCCCODED']

# Focus on 'MEMBERAGEGROUP' feature
age_group_col = 'MEMBERAGEGROUP'

# Ensure 'MEMBERAGEGROUP' is treated as categorical
X[age_group_col] = X[age_group_col].astype(str)

# Step 1: Visualize Age Group Distribution
# plt.figure(figsize=(10, 6))
# sns.countplot(data=df, x='MEMBERAGEGROUP', palette='coolwarm')
# plt.title('Distribution of Age Groups')
# plt.xlabel('Age Group')
# plt.ylabel('Frequency')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# Step 2: Visualize Target Distribution (V28HCCCODED)
# plt.figure(figsize=(8, 6))
# sns.countplot(data=df, x='V28HCCCODED', palette='viridis')
# plt.title('Distribution of Target Variable (V28HCCCODED)')
# plt.xlabel('V28HCCCODED')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.show()
numerical_cols = [
    col for col in X.columns if X[col].dtype in ['int64', 'float64']]
# Step 3: Correlation Matrix for Numerical Features

# plt.figure(figsize=(12, 10))
# sns.heatmap(df[numerical_cols].corr(), annot=True,
#             cmap='coolwarm', vmin=-1, vmax=1)
# plt.title('Correlation Matrix of Numerical Features')
# plt.tight_layout()
# plt.show()

# Identify categorical and numerical features
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']

# Data Imputation and Feature Scaling
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ])

# Define models to evaluate
models = {
    # 'Logistic Regression': LogisticRegression(max_iter=1000),
    # 'Random Forest': RandomForestClassifier(),
    # 'Decision Tree': DecisionTreeClassifier(),
    'Neural Network':  MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=1000, random_state=42)
}

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Evaluate models using cross-validation
best_model = None
best_accuracy = 0
model_performances = {}

for model_name, model in models.items():
    # Create a pipeline with preprocessing and model
    clf = Pipeline(
        steps=[('preprocessor', preprocessor), ('classifier', model)])

    # Perform cross-validation
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    accuracy = scores.mean()
    model_performances[model_name] = accuracy

    print(f'{model_name} Accuracy: {accuracy:.4f}')

    # Update the best model if current model is better
    if accuracy > best_accuracy:
        best_model = clf
        best_accuracy = accuracy

# Step 4: Visualize Model Performance
# plt.figure(figsize=(10, 6))
# sns.barplot(x=list(model_performances.keys()), y=list(
#     model_performances.values()), palette='magma')
# plt.title('Model Performance Comparison')
# plt.xlabel('Model')
# plt.ylabel('Accuracy')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# Hyperparameter tuning on the best model
if isinstance(best_model.named_steps['classifier'], RandomForestClassifier):
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }
elif isinstance(best_model.named_steps['classifier'], LogisticRegression):
    param_grid = {
        'classifier__C': [0.1, 1.0, 10],
        'classifier__penalty': ['l2']
    }
elif isinstance(best_model.named_steps['classifier'], SVC):
    param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf']
    }
elif isinstance(best_model.named_steps['classifier'], MLPClassifier):
    param_grid = {
        'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'classifier__activation': ['relu', 'tanh'],
        'classifier__alpha': [0.0001, 0.001, 0.01]
    }
else:
    param_grid = {}

if param_grid:
    grid_search = GridSearchCV(
        best_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

# Fit the best model on the training set
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate model performance
test_accuracy = accuracy_score(y_test, y_pred)
print(
    f'\nBest Model: {best_model.named_steps["classifier"].__class__.__name__}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Step 5: Confusion Matrix Visualization
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test, y_pred),
#             annot=True, cmap='coolwarm', fmt='d')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.tight_layout()
# plt.show()
