from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from flasgger import Swagger

app = Flask(__name__)

# Swagger config
app.config['SWAGGER'] = {
    'title': 'Cambridge Single Family Assessed Value Prediction API',
    'uiversion': 3
}
swagger = Swagger(app)

# SQLite DB setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///assessments.db'
db = SQLAlchemy(app)

# Define a database model
class Assessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    assessedvalue = db.Column(db.Float, nullable=False)
    interior_bedrooms = db.Column(db.Integer, nullable=False)
    interior_fullbaths = db.Column(db.Float, nullable=False)
    interior_halfbaths = db.Column(db.Integer, nullable=False)
    condition_overallcondition = db.Column(db.String(100), nullable=False)

# Create the database
with app.app_context():
    db.create_all()

def preprocess_data(df):
    # Clean the assessedvalue column
    df['assessedvalue'] = df['assessedvalue'].astype(float)

    # Drop rows where any of the key fields are NaN
    df = df.dropna(subset=['assessedvalue', 'interior_bedrooms', 'interior_fullbaths', 'interior_halfbaths', 'condition_overallcondition'])

    # One more time, fill any missing numerical values with the median, just in case
    df['interior_bedrooms'] = df['interior_bedrooms'].fillna(df['interior_bedrooms'].median())
    df['interior_fullbaths'] = df['interior_fullbaths'].fillna(df['interior_fullbaths'].median())
    df['interior_halfbaths'] = df['interior_halfbaths'].fillna(df['interior_halfbaths'].median())

    # Fill missing categorical values (condition_overallcondition) with the most frequent value
    df['condition_overallcondition'] = df['condition_overallcondition'].fillna(df['condition_overallcondition'].mode()[0])

    # One-hot encode the 'condition_overallcondition' column
    encoder = OneHotEncoder(sparse_output=False)
    condition_overallcondition_encoded = encoder.fit_transform(df[['condition_overallcondition']])

    # Create a DataFrame for the one-hot encoded overall condition
    condition_overallcondition_encoded_df = pd.DataFrame(condition_overallcondition_encoded, columns=encoder.get_feature_names_out(['condition_overallcondition']))

    # Concatenate the encoded condition_overallcondition with the original dataframe
    df = pd.concat([df, condition_overallcondition_encoded_df], axis=1).drop(columns=['condition_overallcondition'])

    # Drop any rows that still have NaN values at this point (forcefully)
    df = df.dropna()
    return df, encoder

# Global variables for model and encoder
model = None
encoder = None

@app.route('/reload', methods=['POST'])
def reload_data():
    '''
    Reload data from the Cambridge Assessed Value dataset, clear the database, load new data, and return summary stats
    ---
    responses:
      200:
        description: Summary statistics of reloaded data
    '''
    global model, encoder

    # Step 1: Download and decompress data
    url = 'https://data.cambridgema.gov/resource/eey2-rv59.csv?$limit=40000&$offset=150'
    response = requests.get(url)
  
    # Step 2: Load data into pandas
    assessments = pd.read_csv(BytesIO(response.content))

    # Step 3: Clear the database
    db.session.query(Assessment).delete()

    # Step 4: Process data and insert it into the database
    assessments = assessments[['assessedvalue', 'interior_bedrooms', 'interior_fullbaths', 'interior_halfbaths', 'condition_overallcondition']].dropna()
    assessments['assessedvalue'] = assessments['assessedvalue'].astype(float)

    for _, row in assessments.iterrows():
        new_assessments = Assessment(
            assessedvalue=row['assessedvalue'],
            interior_bedrooms=int(row['interior_bedrooms']),
            interior_fullbaths=row['interior_fullbaths'],
            interior_halfbaths=int(row['interior_halfbaths']),
            condition_overallcondition=row['condition_overallcondition']
        )
        db.session.add(new_assessments)
    db.session.commit()

    # Step 5: Preprocess and train model
    df, encoder = preprocess_data(assessments)
    X = df.drop(columns='assessedvalue')
    y = df['assessedvalue']
    model = LinearRegression()
    model.fit(X, y)

    # Step 6: Generate summary statistics
    summary = {
        'total_assessments': len(assessments),
        'average_assessedvalue': assessments['assessedvalue'].mean(),
        'min_assessedvalue': assessments['assessedvalue'].min(),
        'max_assessedvalue': assessments['assessedvalue'].max(),
        'average_interior_bedrooms': assessments['interior_bedrooms'].mean(),
        'average_interior_fullbaths': assessments['interior_fullbaths'].mean(),
        'top_condition_overallconditions': assessments['condition_overallcondition'].value_counts().head().to_dict()
    }

    return jsonify(summary)
@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the assessed value for a single family home in Cambridge based on the input features
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            interior_bedrooms:
              type: integer
            interior_fullbaths:
              type: number
            interior_halfbaths:
              type: integer
            condition_overallcondition:
              type: string
    responses:
      200:
        description: Predicted assessed value
    '''
    global model, encoder  # Ensure that the encoder and model are available for prediction

    # Define the list of valid condition_overallcondition
    valid_condition_overallcondition = [
        "Average", "Excellent", "Fair", "Good", "Poor", "Superior", "Very Good"
    ]

    # Check if the model and encoder are initialized
    if model is None or encoder is None:
        return jsonify({"error": "The data has not been loaded. Please refresh the data by calling the '/reload' endpoint first."}), 400

    data = request.json
    try:
        interior_bedrooms = pd.to_numeric(data.get('interior_bedrooms'), errors='coerce')
        interior_fullbaths = pd.to_numeric(data.get('interior_fullbaths'), errors='coerce')
        interior_halfbaths = pd.to_numeric(data.get('interior_halfbaths'), errors='coerce')
        condition_overallcondition = data.get('condition_overallcondition')

        if None in [interior_bedrooms, interior_fullbaths, interior_halfbaths, condition_overallcondition]:
            return jsonify({"error": "Missing or invalid required parameters"}), 400

        # Check if the condition_overallcondition is valid
        if condition_overallcondition not in valid_condition_overallcondition:
            return jsonify({"error": f"Invalid condition_overallcondition. Please choose one of the following: {', '.join(valid_condition_overallcondition)}"}), 400

        # Check for NaN values in the converted inputs
        if pd.isna(interior_bedrooms) or pd.isna(interior_fullbaths) or pd.isna(interior_halfbaths):
            return jsonify({"error": "Invalid numeric values for interior_bedrooms, interior_fullbaths, or interior_halfbaths"}), 400

        # Transform the input using the global encoder
        condition_overallcondition_encoded = encoder.transform([[condition_overallcondition]])
        input_data = np.concatenate(([interior_bedrooms, interior_fullbaths, interior_halfbaths], condition_overallcondition_encoded[0]))
        input_data = input_data.reshape(1, -1)

        # Predict the assessedvalue
        predicted_assessedvalue = model.predict(input_data)[0]

        return jsonify({"predicted_assessedvalue": predicted_assessedvalue})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
