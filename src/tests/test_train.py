
import os
import pickle
import pytest

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../model.pkl')
MODEL_PATH = os.path.abspath(MODEL_PATH)


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), f"Le fichier modèle {MODEL_PATH} n'existe pas."


def test_model_load():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Le modèle n'a pas encore été généré.")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    assert hasattr(model, 'predict'), "Le modèle n'a pas de méthode 'predict'."
