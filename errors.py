from google.genai.errors import ClientError

from schemas import LLMError, LLMErrorType


# No solicitado en la tarea pero RATE_LIMIT es el error mas comun con genai porque la versión gratuita es muy limitada
def classify_error(error: Exception) -> LLMError:
    # El test permitió verificar que ClientError utiliza "code" para el código HTTP.
    if isinstance(error, ClientError) and getattr(error, "code", None) == 429:
        return LLMError(
            LLMErrorType.RATE_LIMIT, "El proveedor alcanzó el límite de solicitudes."
        )

    return LLMError(
        LLMErrorType.UNKNOWN,
        "Ocurrió un error inesperado al comunicarse con el modelo.",
    )
