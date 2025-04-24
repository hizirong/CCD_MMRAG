def rerank(query: str, raw_result: dict, k: int = 5) -> dict:
    """
    之後用 cross‑encoder MS‑MARCO or Cohere‑rerank.
    目前直接回傳原結果。
    """
    return raw_result
