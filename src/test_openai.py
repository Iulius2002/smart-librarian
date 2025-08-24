from openai import OpenAI
from .config import OPENAI_API_KEY, CHAT_MODEL

def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("Nu ai setat OPENAI_API_KEY în .env")

    client = OpenAI(api_key=OPENAI_API_KEY)

    msg = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Ești un asistent care răspunde foarte pe scurt."},
            {"role": "user", "content": "Spune-mi un salut de 3 cuvinte."}
        ],
        temperature=0.2
    )

    print("Răspuns:", msg.choices[0].message.content)

if __name__ == "__main__":
    main()
