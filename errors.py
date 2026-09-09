from google.genai.errors import ClientError

from schemas import LLMError, LLMErrorType


# La clasificación de RATE_LIMIT no es requerida por la consigna,
# pero se agrega para manejar específicamente el error 429 de Gemini.
def classify_error(error: Exception) -> LLMError:
    # ClientError expone el código HTTP mediante el atributo "code".
    if isinstance(error, ClientError) and getattr(error, "code", None) == 429:
        return LLMError(
            LLMErrorType.RATE_LIMIT, "El proveedor alcanzó el límite de solicitudes."
        )

    return LLMError(
        LLMErrorType.UNKNOWN,
        "Ocurrió un error inesperado al comunicarse con el modelo.",
    )