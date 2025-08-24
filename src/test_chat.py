# src/test_chat.py
from .chat import chat_once

def main():
    print("Smart Librarian — întreabă-mă ce carte cauți (Ctrl+C pentru ieșire).")
    while True:
        try:
            q = input("\nTu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nLa revedere!")
            break
        if not q:
            continue

        ans = chat_once(q, k=3)
        print("\nAsistent:", ans)

if __name__ == "__main__":
    main()
