import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class F1PredictionModel:
    def __init__(self, start_year=2018, end_year=2023):
        self.start_year = start_year
        self.end_year = end_year
        self.model = None
        self.scaler = StandardScaler()
        
    def collect_historical_data(self):
        """
        Collect comprehensive F1 race data across multiple seasons
        """
        all_race_data = []
        
        for year in range(self.start_year, self.end_year + 1):
            try:
                # Collect data for each Grand Prix
                for event in fastf1.get_event_schedule(year):
                    try:
                        # Load race session
                        race_session = fastf1.get_session(year, event.EventName, 'R')
                        race_session.load()
                        
                        # Extract key features for each driver
                        for driver in race_session.results['Driver']:
                            driver_laps = race_session.laps.pick_driver(driver)
                            
                            # Feature extraction
                            race_features = {
                                'Year': year,
                                'Grand Prix': event.EventName,
                                'Driver': driver,
                                'Team': race_session.results[race_session.results['Driver'] == driver]['Team'].values[0],
                                'Average_Lap_Time': driver_laps['LapTime'].mean().total_seconds(),
                                'Fastest_Lap': driver_laps.pick_fastest()['LapTime'].total_seconds(),
                                'Number_of_Laps': len(driver_laps),
                                'Qualifying_Position': race_session.results[race_session.results['Driver'] == driver]['GridPosition'].values[0],
                                'Final_Position': race_session.results[race_session.results['Driver'] == driver]['Position'].values[0],
                                'Finished_Race': 1 if race_session.results[race_session.results['Driver'] == driver]['Status'].values[0] == 'Finished' else 0
                            }
                            
                            all_race_data.append(race_features)
                    
                    except Exception as session_error:
                        print(f"Error processing {event.EventName} in {year}: {session_error}")
            
            except Exception as year_error:
                print(f"Error processing year {year}: {year_error}")
        
        # Convert to DataFrame
        self.race_data = pd.DataFrame(all_race_data)
        return self.race_data
    
    def prepare_data(self):
        """
        Prepare data for machine learning model
        """
        # One-hot encode categorical features
        race_data_encoded = pd.get_dummies(self.race_data, columns=['Driver', 'Team', 'Grand Prix'])
        
        # Select features and target
        features = race_data_encoded.drop(['Final_Position', 'Year'], axis=1)
        target = race_data_encoded['Final_Position']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Scale numerical features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self):
        """
        Train a Random Forest Classifier for race outcome prediction
        """
        # Collect and prepare data
        self.collect_historical_data()
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Initialize and train the model
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        
        # Print performance metrics
        print("Model Performance Metrics:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix Visualization
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix of F1 Race Outcome Prediction')
        plt.xlabel('Predicted Position')
        plt.ylabel('Actual Position')
        plt.show()
        
        return self.model
    
    def predict_race_outcome(self, driver_features):
        """
        Predict race outcome for given driver features
        """
        if self.model is None:
            raise ValueError("Model must be trained first. Call train_model() first.")
        
        # Scale input features
        scaled_features = self.scaler.transform([driver_features])
        
        # Predict probability of different finishing positions
        position_probabilities = self.model.predict_proba(scaled_features)
        
        return position_probabilities

# Example usage
if __name__ == "__main__":
    # Initialize and train the model
    f1_predictor = F1PredictionModel(start_year=2018, end_year=2023)
    f1_predictor.train_model()
    
    # Example prediction (you would need to prepare actual driver features)
    example_driver_features = [
        # Prepare your driver's specific features here
        # This is a placeholder and would need actual data
        0.5,  # Average Lap Time
        45.2,  # Fastest Lap
        50,    # Number of Laps
        5,     # Qualifying Position
        1      # Finished Previous Race
    ]
    
    prediction = f1_predictor.predict_race_outcome(example_driver_features)
    print("Prediction Probabilities for Different Finishing Positions:")
    print(prediction)