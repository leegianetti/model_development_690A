import pytest
from app import app, db

# Define valid input for prediction test
valid_input = {
    "interior_bedrooms": 2,
    "interior_fullbaths": 1,
    "interior_halfbaths": 1,
    "condition_overallcondition": "Good"
}

# Define invalid condition input for prediction test
invalid_condition_overallcondition_input = {
    "interior_bedrooms": 2,
    "interior_fullbaths": 1,
    "interior_halfbaths": 1,
    "condition_overallcondition": "good"
}

# Define missing field input for prediction test
missing_field_input = {
    "interior_bedrooms": 2,
    "interior_fullbaths": 1,
    "condition_overallcondition": "Good"
}


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client


def test_reload_data(client):
    """Test the reload endpoint that loads the data."""
    response = client.post('/reload')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'total_assessments' in json_data
    assert 'average_assessedvalue' in json_data

def test_predict_after_reload(client):
    """Test prediction endpoint after reloading the data."""
    # Reload the data first
    client.post('/reload')

    # Test valid prediction
    response = client.post('/predict', json=valid_input)
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'predicted_assessedvalue' in json_data


def test_invalid_condition_overallcondition(client):
    """Test prediction with an invalid condition_overallcondition."""
    # Reload the data first
    client.post('/reload')

    # Test invalid condition_overallcondition
    response = client.post('/predict', json=invalid_condition_overallcondition_input)
    assert response.status_code == 400
    json_data = response.get_json()
    assert "Invalid condition_overallcondition" in json_data['error']


def test_missing_fields(client):
    """Test prediction with missing fields."""
    # Reload the data first
    client.post('/reload')

    # Test with missing fields
    response = client.post('/predict', json=missing_field_input)
    assert response.status_code == 400
    json_data = response.get_json()
    assert "Invalid numeric values for interior_bedrooms, interior_fullbaths, or interior_halfbaths" in json_data['error']

