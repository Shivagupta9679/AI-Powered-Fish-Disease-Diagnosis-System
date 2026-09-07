import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def generate_diagnosis(disease_prediction, rag_results):

    rag_context = "\n\n".join(rag_results)

    prompt = f"""
You are an AI assistant for aquaculture
fish disease analysis.

AI image-based prediction:
{disease_prediction}

Retrieved knowledge from the RAG database:
{rag_context}

Generate a clear and structured report.

Include:

1. AI predicted disease/class
2. Explanation of the prediction
3. Relevant symptoms
4. Evidence from the knowledge base
5. Recommended management actions
6. Prevention recommendations
7. Important caution

Rules:

- Use the retrieved knowledge as the factual source.
- Do not invent medicines.
- Do not invent dosages.
- Do not invent treatment procedures.
- Do not claim certainty.
- Clearly state that this is an AI-assisted prediction.
- Recommend professional aquaculture/veterinary
  confirmation when appropriate.
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return response.output_text