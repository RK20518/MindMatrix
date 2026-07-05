from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def analyze(user_text, retrieved_context):

    prompt = f"""
You are MindMatrix, an AI-powered Mental Health Assessment System.

Retrieved Similar Cases:
{retrieved_context}

User Input:
{user_text}

Instructions:
1. Use retrieved cases only as supporting evidence.
2. Do not assume the user has the same condition as retrieved examples.
3. Assess carefully.
4. If there are signs of crisis or self-harm, mention that clearly.
5. Keep the response professional.

Return STRICTLY in this format:

Possible Condition: <condition>

Confidence Score: <0-100>

Risk Level: <Low/Moderate/High/Critical>

Explanation:
<brief explanation>

Recommendations:
- point 1
- point 2
- point 3
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content