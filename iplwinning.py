import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
team_wise_df = pd.read_csv('teamwise_home_and_away.csv')
matches_df = pd.read_csv('matches.csv')

# Prepare features and target variable
def prepare_features(df):
    features = df[['home_win_percentage', 'away_win_percentage']].values
    return features

# Train the model
def train_model():
    # Create a sample target variable (0 or 1) based on a simple rule
    team_wise_df['target'] = np.where(team_wise_df['home_win_percentage'] > team_wise_df['away_win_percentage'], 1, 0)

    # Features and target variable
    X = prepare_features(team_wise_df)
    y = team_wise_df['target']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    return model

# Prediction function
def predict_winner(model, team1, team2):
    home_win_percentage = team_wise_df.set_index('team')['home_win_percentage'].to_dict()
    away_win_percentage = team_wise_df.set_index('team')['away_win_percentage'].to_dict()

    # Get win percentages for the teams
    team1_home_win_percentage = home_win_percentage.get(team1, 0)
    team1_away_win_percentage = away_win_percentage.get(team1, 0)
    team2_home_win_percentage = home_win_percentage.get(team2, 0)
    team2_away_win_percentage = away_win_percentage.get(team2, 0)

    # Prepare features for prediction
    # Using team1 features only for prediction
    features = np.array([[team1_home_win_percentage, team1_away_win_percentage]])

    # Predict the winner
    prediction = model.predict(features)
    return "Team 1 wins" if prediction[0] == 1 else "Team 2 wins"

# Streamlit UI
st.title('IPL Winning Team Prediction')

# Train the model
model = train_model()

# Create dropdowns for team selection
team1 = st.selectbox('Select Team 1', team_wise_df['team'])
team2 = st.selectbox('Select Team 2', team_wise_df['team'])

# Predict button
if st.button('Predict'):
    if team1 == team2:
        st.write("Please select two different teams.")
    else:
        result = predict_winner(model, team1, team2)
        st.write(f"The predicted winner is: {result}")
