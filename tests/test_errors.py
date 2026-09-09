import pytest

from google.genai.errors import ClientError

from errors import classify_error
from schemas import LLMError, LLMErrorType


def test_classify_rate_limit(monkeypatch):

    error = ClientError(
        429,
        {"error": {"message": "Resource exhausted"}}
    )

    monkeypatch.setattr(error, "code", 429)

    resultado = classify_error(error)

    assert isinstance(resultado, LLMError)
    assert resultado.error_type == LLMErrorType.RATE_LIMIT
    assert resultado.message == (
        "El proveedor alcanzó el límite de solicitudes."
    )


def test_classify_unknown_error():

    error = ValueError("Error de prueba")

    resultado = classify_error(error)

    assert isinstance(resultado, LLMError)
    assert resultado.error_type == LLMErrorType.UNKNOWN
    assert resultado.message == (
        "Ocurrió un error inesperado al comunicarse con el modelo."
    )