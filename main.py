from google.colab import drive 

drive.mount('/content/drive') 

 

# --- Data Loading --- 

# IMPORTANT: Please upload your CSV file named 'lungs.csv' to Colab's file system,  

# OR replace 'lungs.csv' below with the actual name/path of your dataset. 

# If your dataset is in Google Drive, use its full path (e.g., '/content/drive/MyDrive/your_folder/your_dataset_name.csv'). 

 

import pandas as pd 

try: 

    df = pd.read_csv('/content/lungs.csv') # Changed placeholder filename to 'lungs.csv' 

    print ("Dataset loaded successfully!") 

    display(df.head()) 

    display(df.info()) 

except FileNotFoundError: 

    print ("Error: Dataset 'lungs.csv' not found. Please upload your CSV file to Colab's file system") 

    print ("or provide the correct path if it's in Google Drive or has a different name.") 

    print ("As a temporary measure, initializing an empty DataFrame.") 

    df = pd.DataFrame() 

 



# Check if df is not empty before processing 

if not df.empty: 

    print ("Starting data preprocessing...") 

 

    # Example: Drop columns that are not useful for prediction (modify as per your dataset) 

    # Assuming 'Patient Id' and similar columns are identifiers and not features. 

    if 'Patient Id' in df.columns: 

        df = df.drop('Patient Id', axis=1) 

     

    # Handle missing values (example: fill with mode for categorical, mean/median for numerical) 

    # This is a generic approach; inspect your data for better strategies. 

    for column in df.columns: 

        if df[column].dtype == 'object': # Categorical columns 

            df[column] = df[column].fillna(df[column].mode()[0]) 

        else: # Numerical columns 

            df[column] = df[column].fillna(df[column].mean()) 

 

    # Encode categorical features 

    from sklearn.preprocessing import LabelEncoder 

    for column in df.columns: 

        if df[column].dtype == 'object': 

            le = LabelEncoder() 

            df[column] = le.fit_transform(df[column]) 

     

    # Define features (X) and target (y) 

    # Assuming 'lung disease' or 'Level' (like in your original code) is the target variable. 

    # IMPORTANT: Replace 'target_column_name' with the actual name of your target column. 

    target_column_name = 'Result' # Changed from 'Level' to 'Result' to match the provided lungs.csv 

     

    if target_column_name in df.columns: 

        X = df.drop(target_column_name, axis=1) 

        y = df[target_column_name] 

 

        # Split data into training and testing sets 

        from sklearn.model_selection import train_test_split 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

         

        print ("Data preprocessing completes. Data split into training and testing sets. 

        print (f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}") 

        print (f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}") 

    else: 

        print (f"Error: Target column '{target_column_name}' not found in the dataset. Please adjust `target_column_name`.") 

        X_train, X_test, y_train, y_test = None, None, None, None 

else: 

    print ("Preprocessing skipped: DataFrame is empty. Please load your dataset first. 

    X_train, X_test, y_train, y_test = None, None, None, None 

 

if X_train is not None and y_train is not None: 

    print ("Starting model training...") 

    from sklearn.ensemble import RandomForestClassifier 

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 

 

    # Initialize and train the Random Forest Classifier 

    model = RandomForestClassifier(n_estimators=100, random_state=42) 

    model.fit(X_train, y_train) 

 

    print ("Model training completes.") 

 

    # Make predictions on the test set 

    y_pred = model.predict(X_test) 

 

    # Evaluate the model 

    accuracy = accuracy_score(y_test, y_pred) 

    print (f"\nModel Accuracy: {accuracy:.4f}") 

    print ("\nClassification Report:") 

    print (classification_report(y_test, y_pred)) 

    print ("\nConfusion Matrix:") 

    print (confusion_matrix(y_test, y_pred)) 

else: 

    print ("Model training skipped: Training data (X_train, y_train) is not available.") 

 


if 'model' in locals() and not df.empty: 

    print ("Demonstrating prediction on a sample new data point.") 

    # Create a sample new data point. It should have the same features as X_train. 

    # For demonstration, we'll take the first row of X_test as an example of 'new data'. 

    # In a real scenario, this would be actual new patient data. 

    sample_new_data = X_test.iloc[[0]] # Using iloc[[0]] to keep it as a DataFrame 

 

    # Make a prediction 

    new_prediction = model.predict(sample_new_data) 

 

    print("\nSample New Data Point:") 

    display(sample_new_data) 

    print(f"Predicted Lung Disease Level for the sample data: {new_prediction[0]}") 

 

    # You can also get prediction probabilities 

    new_prediction_proba = model.predict_proba(sample_new_data) 

    print(f"Prediction Probabilities: {new_prediction_proba}") 

else: 

    print("Prediction skipped: Model not trained or DataFrame is empty. Please ensure that data is loaded and the model is trained. 

 


