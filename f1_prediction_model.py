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
        self.race_data = None
        
    def collect_historical_data(self):
        """
        Collect comprehensive F1 race data across multiple seasons
        """
        all_race_data = []
        
        for year in range(self.start_year, self.end_year + 1):
            try:
                # Get event schedule using correct method
                event_schedule = fastf1.get_event_schedule(year)
                
                # Print events to debug
                print(f"Events for {year}:")
                print(event_schedule)
                
                # Iterate through events
                for index, event in event_schedule.iterrows():
                    try:
                        # Use event name from the schedule
                        event_name = event['EventName']
                        
                        # Load race session
                        race_session = fastf1.get_session(year, event_name, 'R')
                        race_session.load()
                        
                        # Extract results
                        results = race_session.results
                        
                        # Print results to debug
                        print(f"Results for {event_name} {year}:")
                        print(results)
                        
                        # Feature extraction for each driver
                        for index, driver_result in results.iterrows():
                            driver = driver_result['Driver']
                            
                            # Safely extract driver laps
                            try:
                                driver_laps = race_session.laps.pick_driver(driver)
                            except Exception as lap_error:
                                print(f"Error getting laps for {driver}: {lap_error}")
                                continue
                            
                            # Feature extraction with error handling
                            race_features = {
                                'Year': year,
                                'Grand Prix': event_name,
                                'Driver': driver,
                                'Team': driver_result.get('Team', 'Unknown'),
                                'Average_Lap_Time': driver_laps['LapTime'].mean().total_seconds() if len(driver_laps) > 0 else np.nan,
                                'Fastest_Lap': driver_laps.pick_fastest()['LapTime'].total_seconds() if len(driver_laps) > 0 else np.nan,
                                'Number_of_Laps': len(driver_laps),
                                'Qualifying_Position': driver_result.get('GridPosition', np.nan),
                                'Final_Position': driver_result.get('Position', np.nan),
                                'Finished_Race': 1 if driver_result.get('Status', '') == 'Finished' else 0
                            }
                            
                            all_race_data.append(race_features)
                    
                    except Exception as event_error:
                        print(f"Error processing {event_name} in {year}: {event_error}")
            
            except Exception as year_error:
                print(f"Error processing year {year}: {year_error}")
        
        # Convert to DataFrame
        self.race_data = pd.DataFrame(all_race_data)
        
        # Print DataFrame to debug
        print("Collected Race Data:")
        print(self.race_data)
        print(self.race_data.columns)
        
        return self.race_data
    
    def prepare_data(self):
        """
        Prepare data for machine learning model
        """
        # Drop rows with NaN values
        race_data_cleaned = self.race_data.dropna()
        
        # Ensure we have data
        if len(race_data_cleaned) == 0:
            raise ValueError("No valid data found after cleaning")
        
        # One-hot encode categorical features
        race_data_encoded = pd.get_dummies(
            race_data_cleaned, 
            columns=['Driver', 'Team', 'Grand Prix']
        )
        
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
        
        try:
            X_train, X_test, y_train, y_test = self.prepare_data()
        except Exception as prep_error:
            print("Error preparing data:", prep_error)
            return None
        
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