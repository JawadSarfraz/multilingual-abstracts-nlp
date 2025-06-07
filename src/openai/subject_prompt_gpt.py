import os
from openai import OpenAI
from dotenv import load_dotenv

# ✅ Load from .env file if available
load_dotenv()

def predict_subjects(abstract):
    """
    Sends an abstract to OpenAI's GPT model and returns 3–5 guessed subject keywords.
    """
    # ✅ Initialize OpenAI client with API key
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Construct the prompt
    prompt = f"""
You are a subject classification assistant for academic economics papers.
Given the following abstract, guess the most relevant 3–5 subject keywords.
Return the result as a JSON array of strings.

Abstract:
\"\"\"{abstract}\"\"\"
Subjects:
"""

    # Send to OpenAI Chat API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=100,
    )


    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    test_abstract = (
        "This paper examines the effects of inflation targeting on employment "
        "and long-run growth across OECD countries."
    )
    predicted_subjects = predict_subjects(test_abstract)
    print("🔍 Predicted Subjects:\n", predicted_subjects)
