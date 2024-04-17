import io
from fastapi.testclient import TestClient
from predict_api.main import app, BASE_DIR, accepted_extensions
from PIL import Image

client = TestClient(app)


def test_get_home():
    response = client.get("/")
    assert response.status_code == 200  # successful


def test_no_file_upload_error():
    response = client.post("/predict") # requests.post("/predict") # python requests
    assert response.status_code == 422  # no data sent
    assert "application/json" in response.headers['content-type']


def test_file_upload_and_prediction():
    test_saved_path = BASE_DIR / "test_files"
    for path in test_saved_path.glob("*"):
        image_extensions = True
        if not path.suffix in accepted_extensions:
            image_extensions = False
        try:
            img = Image.open(path)
        except:
            img = None
        response = client.post("/predict",
            files={"img_file": open(path, 'rb')},
        )

        if not image_extensions:
            assert response.status_code == 400  # invalid file extensions
        if img is None:
            assert response.status_code == 400  # couldn't upload image
        else:
            # Returning a valid image
            assert response.status_code == 200  # successful upload
            data = response.json()
            assert len(data.keys()) == 1    # true output form(1 json file for the classification)