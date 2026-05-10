import os
import pandas as pd
import numpy as pd_np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from ucimlrepo import fetch_ucirepo

def main():
    print("Starting Cardiovascular Disease Prediction Analysis...")
    
    # Create directory for plots
    os.makedirs("plots", exist_ok=True)
    
    # 1. Data Loading
    print("Fetching dataset from UCI Machine Learning Repository...")
    heart_disease = fetch_ucirepo(id=45)
    
    # Data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    # The target in the Cleveland dataset is an integer from 0 to 4. 
    # 0 means no heart disease, 1-4 mean presence of heart disease.
    # We convert it to a binary classification problem: 0 (No) and 1 (Yes).
    y = y.copy()
    y['num'] = (y['num'] > 0).astype(int)
    
    # Combine for EDA
    df = pd.concat([X, y], axis=1)
    
    print(f"Dataset loaded with {df.shape[0]} samples and {df.shape[1]} features (including target).")
    
    # 2. Data Pre-processing
    print("Performing Data Pre-processing...")
    
    # Check for missing values
    print("Missing values per feature:")
    print(df.isnull().sum())
    
    # Fill missing values with median for simplicity
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # Update X and y after imputation
    X = df.drop('num', axis=1)
    y = df['num']
    
    # 3. Exploratory Data Analysis (EDA)
    print("Generating visualizations...")
    
    # 3a. Target variable distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='num', data=df, hue='num', palette='Set2', legend=False)
    plt.title('Distribution of Heart Disease (0 = No, 1 = Yes)')
    plt.xlabel('Heart Disease Presence')
    plt.ylabel('Count')
    plt.savefig('plots/target_distribution.png')
    plt.close()
    
    # 3b. Feature distributions (histograms)
    df.hist(figsize=(14, 12), bins=20, color='skyblue', edgecolor='black')
    plt.suptitle('Histograms of Features', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('plots/feature_distributions.png')
    plt.close()
    
    # 3c. Correlation matrix heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    # 3d. Box plots for key features vs target
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='num', y='age', data=df, hue='num', palette='Set1', legend=False)
    plt.title('Age Distribution by Heart Disease Presence')
    plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
    plt.ylabel('Age')
    plt.savefig('plots/age_vs_target.png')
    plt.close()
    
    print("Plots saved in the 'plots' directory.")
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # 4. Model Training & Evaluation
    print("Training Machine Learning Models...")
    
    models = {
        'Support Vector Machines (SVM)': SVC(kernel='rbf', probability=True, random_state=42),
        'K-Nearest Neighbor (KNN)': KNeighborsClassifier(n_neighbors=5),
        'Decision Trees (DT)': DecisionTreeClassifier(random_state=42),
        'Logistic Regression (LR)': LogisticRegression(random_state=42),
        'Random Forest (RF)': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model_name = ""
    best_accuracy = 0
    best_model = None
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"{name} Accuracy: {accuracy*100:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model
            
    print("\n--- Summary of Model Accuracies ---")
    for name, acc in results.items():
        print(f"{name}: {acc*100:.2f}%")
        
    print(f"\nBest Model: {best_model_name} with {best_accuracy*100:.2f}% accuracy.")
    
    # 5. Save final model
    model_filename = 'best_heart_disease_model.pkl'
    # Also save the scaler so we can use it during inference
    joblib.dump({'model': best_model, 'scaler': scaler}, model_filename)
    print(f"Best model and scaler saved as '{model_filename}'")
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()
