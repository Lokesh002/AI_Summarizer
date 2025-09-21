from models.llms.google_llm import gemini_2_5_flash as llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.config import Config

def summarize_pdf(transcript) -> str:
    """Summarizes a transcript using the Gemini model."""
    prompt = ChatPromptTemplate.from_template(Config.Prompts.SUMMARIZER_PROMPT)
    chain = prompt | llm | StrOutputParser()
    try:
        summary = chain.invoke({"transcript": transcript})
        return summary
    except Exception as e:
        raise RuntimeError(f"An error occurred during summarization: {e}")