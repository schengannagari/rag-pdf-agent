import os

from google import genai
from app.pdf_tool import PDFKnowledgeBase, EMBED_MODEL

PDF_PATH = "/home/swaminath/python/hello-world/app/cricket.pdf"
QUERY = "How many balls can an over contain?"
TOP_K = 3
MODEL = "gemini-3-flash-preview" 

def preview(vec, n=5):
    """First n numbers of a vector, rounded -- full vectors have hundreds
    of dimensions, so printing all of them isn't useful."""
    return [round(float(x), 4) for x in vec[:n]]


def line(char="=", width=70):
    print(char * width)

def main():
    line()
    print("STEP 0 - Connect to Gemini")
    line()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    print(f"Embedding model: {EMBED_MODEL}\n")

    # Building PDFKnowledgeBase runs _load_and_chunk() AND _embed()
    # internally, so kb.chunks and kb.embeddings are both ready right after this.
    kb = PDFKnowledgeBase(client, pdf_path=PDF_PATH)
    line()
    print(f"STEP 1 - Load + chunk: {os.path.basename(PDF_PATH)}")
    line()
    print(f"Total chunks created: {len(kb.chunks)}\n")
    for i, chunk in enumerate(kb.chunks):
        snippet = chunk[:90].replace("\n", " ")
        print(f"  Chunk {i:>2}  |  {len(chunk):>4} chars  |  \"{snippet}...\"")

    print()
    line()
    print("STEP 2 - Embeddings for each chunk")
    line()
    print(f"Embedding matrix shape: {kb.embeddings.shape}  "
          f"({kb.embeddings.shape[0]} chunks x {kb.embeddings.shape[1]} dimensions)\n")
    for i, vec in enumerate(kb.embeddings):
        print(f"  Chunk {i:>2} embedding preview: {preview(vec)} ...")

    print()
    line()
    print("STEP 3 - Embed the query")
    line()
    print(f'Query: "{QUERY}"')
    query_vec = kb.embed_query(QUERY)
    print(f"Query embedding shape: {query_vec.shape}")
    print(f"Query embedding preview: {preview(query_vec)} ...\n")

    line()
    print("STEP 4 - Cosine similarity: query vs every chunk")
    line()
    sims = kb.similarities(query_vec)
    for i, s in enumerate(sims):
        bar = "#" * int(max(s, 0) * 40)
        print(f"  Chunk {i:>2}  similarity = {s:.4f}  {bar}")

    print()
    line()
    print(f"STEP 5 - Rank and pick top {TOP_K}")
    line()
    top_idx = kb.rank(sims, TOP_K)
    print(f"Winning indices (best first): {top_idx.tolist()}")
    for rank_num, i in enumerate(top_idx, start=1):
        print(f"  #{rank_num}  Chunk {i}  score={sims[i]:.4f}")

    print()
    line()
    print("STEP 6 - Retrieved TEXT handed back (not vectors)")
    line()
    retrieved_text = "\n\n---\n\n".join(kb.chunks[i] for i in top_idx)
    print(retrieved_text)

    print()
    line()
    print("STEP 7 - Sanity check against kb.search()")
    line()
    same_result = kb.search(QUERY, top_k=TOP_K)
    print("kb.search() output matches the manual steps above:",
          same_result == retrieved_text)

    print()
    line()
    print("STEP 8 - Reasoning: hand the retrieved text to the LLM")
    line()
    # No tools, no agent, no function calling -- just a plain prompt built
    # directly from the text we already retrieved in Step 6.
    prompt = f"""Answer the question using ONLY the context below. If the
        context doesn't contain the answer, say so.

        Context:
        {retrieved_text}

        Question: {QUERY}"""

    chat = client.chats.create(model=MODEL)
    response = chat.send_message(prompt)
    print(f"Question: {QUERY}\n")
    print(f"Answer: {response.text}")


if __name__ == "__main__":
    main()