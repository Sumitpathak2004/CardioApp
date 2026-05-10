# CardioApp ❤️

CardioApp is a Machine Learning project and interactive web application that predicts the risk of cardiovascular disease based on clinical metrics. 

## Features
- **Data Analysis & Visualization:** Python script `analysis.py` automatically fetches the UCI Heart Disease dataset, performs EDA, and outputs correlation and distribution plots.
- **Machine Learning Model:** Evaluated multiple algorithms (SVM, KNN, Decision Trees, Logistic Regression) and selected **Random Forest**, which achieved the highest accuracy (90.16%).
- **Interactive Web App:** A clean, user-friendly frontend built with **Streamlit** to instantly predict a user's cardiovascular risk.

## Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sumitpathak2004/CardioApp.git
   cd CardioApp
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

## Deploying to Streamlit Community Cloud
Since this repository is already hosted on GitHub, you can easily deploy this app for free:
1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
2. Click **New app**.
3. Select your `Sumitpathak2004/CardioApp` repository.
4. Set the main file path to `app.py`.
5. Click **Deploy!** 

## Files Included
- `app.py`: The Streamlit web application.
- `analysis.py`: Script to download data, perform EDA, and train the model.
- `best_heart_disease_model.pkl`: The pre-trained Random Forest model and `StandardScaler` used by `app.py`.
- `plots/`: Visualizations generated during data analysis.
