import os
from openai import OpenAI
from dotenv import load_dotenv

# ✅ Load from .env file if available
load_dotenv()

def predict_subjects(abstract):
    """
    Sends an abstract to OpenAI's GPT model and returns 5–7 guessed subject keywords.
    """
    # ✅ Initialize OpenAI client with API key
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Construct the prompt
    prompt = f"""
You are a subject classification assistant for academic economics papers.
Given the following abstract, guess the most relevant 5–7 subject keywords.
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
        "This paper considers whether a replacement rate cut can be income equality enhancing and with what conditions. The logical answer to the question is yes, if the propensity of moving from low income state to high income state is high enough. The main contribution of this paper is to derive an analytical expression of income equality improving elasticity. It specifies the limit, after which replacement rate cut is equality enhancing measured by Gini coefficient.s"
    )
    predicted_subjects = predict_subjects(test_abstract)
    print("🔍 Predicted Subjects:\n", predicted_subjects)
