from .vector_store import query_similar

def main():
    print("Interogare demo. Exemple:")
    print(" - 'o carte despre libertate și control social'")
    print(" - 'prietenie și magie'")
    print(" - 'război și destine'")

    while True:
        try:
            q = input("\nCaut: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nLa revedere!")
            break

        if not q:
            continue

        res = query_similar(q, n_results=3)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        if not docs:
            print("Nicio potrivire.")
            continue

        print("\nTop 3 rezultate (distanță mai mică = mai relevant):")
        for i, (m, d) in enumerate(zip(metas, dists), start=1):
            title = m.get("title", "N/A")
            print(f"{i}) {title}   [dist={d:.4f}]")

if __name__ == "__main__":
    main()
