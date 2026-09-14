"""
Call an LLM with the retrieved context to answer the user's question.

Two backends - pick one with the LLM_BACKEND env var in .env:
  - "gemini": Google's free-tier Gemini API (needs GEMINI_API_KEY)
  - "ollama": a fully local model via Ollama (no API key at all)
"""
import os
from dotenv import load_dotenv

load_dotenv()

LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini").strip().lower()

PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know - don't make anything up.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, retrieved_chunks: list) -> str:
    context = "\n\n---\n\n".join(chunk for chunk, _score in retrieved_chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(question: str, retrieved_chunks: list) -> str:
    prompt = build_prompt(question, retrieved_chunks)

    if LLM_BACKEND == "gemini":
        return _generate_with_gemini(prompt)
    elif LLM_BACKEND == "ollama":
        return _generate_with_ollama(prompt)
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")


def _generate_with_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set it in your environment or .env file, "
            "or switch LLM_BACKEND=ollama."
        )

    from google import genai
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


def _generate_with_ollama(prompt: str, model: str = "llama3.2") -> str:
    # Needs Ollama installed and running locally (ollama.com), and the model pulled:
    #   ollama pull llama3.2
    import ollama
    response = ollama.generate(model=model, prompt=prompt)
    return response["response"]
