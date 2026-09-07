from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from prompt_config import prompt
from schemas import RespuestaLLM

parser_llm = PydanticOutputParser(pydantic_object=RespuestaLLM)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", temperature=0.4, max_tokens=500
)

# Cadena LCEL
chain = prompt | llm | parser_llm
