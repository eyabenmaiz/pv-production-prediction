import pytest
from app import app

@pytest.fixture
def client():
	app.config["TESTING"] = True
	with app.test_client() as client:
		yield client

def test_health_endpoint(client):
	response = client.get("/health")
	assert response.status_code == 200
	data = response.get_json()
	assert data["model_loaded"] is True
def test_predict_valid_data(client):
	payload = {
        "Capacite_kWp": 10,
        "Tavg": 25,
        "Tmax": 30,
        "Tmin": 20,
        "Wind_mps": 3,
        "Solar_Wm2": 6,
        "Humidity": 50,
        "Type_centrale": "Résidentiel",
        "Region": "Tunisie/Sfax"
    }
	response = client.post("/predict", json=payload)
	assert response.status_code == 200
	data = response.get_json()
	assert data["status"] == "success"
	assert data["production_spec_kWh_per_kWp"] >= 0

def test_predict_missing_field(client):
	payload= {
        "Capacite_kWp": 10,
        "Tavg": 25
    }
	response = client.post("/predict", json=payload)
	assert response.status_code == 400

def test_predict_out_of_range(client):
    payload = {
        "Capacite_kWp": 10,
        "Tavg": 999,
        "Tmax": 30,
        "Tmin": 20,
        "Wind_mps": 3,
        "Solar_Wm2": 6,
        "Humidity": 50
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
