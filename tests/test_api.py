import pytest
from app.api import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_add_endpoint(client):
    r = client.post("/add", json={"a": 2, "b": 3})
    assert r.status_code == 200
    assert r.get_json()["result"] == 5


def test_divide_endpoint(client):
    r = client.post("/divide", json={"a": 10, "b": 2})
    assert r.status_code == 200
    assert r.get_json()["result"] == 5


def test_divide_by_zero_endpoint(client):
    r = client.post("/divide", json={"a": 1, "b": 0})
    assert r.status_code == 400


def test_factorial_endpoint(client):
    r = client.post("/factorial", json={"n": 5})
    assert r.status_code == 200
    assert r.get_json()["result"] == 120
