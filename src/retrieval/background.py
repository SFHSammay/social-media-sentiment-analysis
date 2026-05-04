import re
import time

import google.generativeai as genai

from src.config import GEMINI_API_KEY, GEMINI_MODEL, GEMMA_MODEL, MAX_RETRIES

genai.configure(api_key=GEMINI_API_KEY)


# Calls gemma model
def call_gemma(prompt: str) -> str:
    model = genai.GenerativeModel(GEMMA_MODEL)
    response = model.generate_content(prompt)
    return response.text


# Calls gemini model
def call_gemini(prompt: str) -> str:
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text


# Build prompt for gemma model
def build_gemma_prompt(query: str, texts: list[str]) -> str:
    tweets = "\n".join(f"{i}. {text}" for i, text in enumerate(texts, start=1))
    return f"""Label relevance between the query and each tweet.
        Return only one Python list of integers.
        Return:
        1 = relevant, clearly about the same topic or entity as the query
        0 = not relevant
        -1 = unsure or borderline

        No explanation. No bullet points. No analysis. No extra text.
        Be strict. Use 1 only if clearly relevant. If uncertain, choose 0 or -1.
        The list length MUST equal the number of tweets.
        The output list order MUST exactly match the tweet order above.
        The first output value corresponds to tweet 1, the second to tweet 2, and so on.

        Query: "{query}"

        Tweets:
        {tweets}

        Example output:
        [1, 0, -1, 1, 0, 0, 1, 0, 0, 1]
        """


# Build prompt for gemini model
def build_gemini_prompt(query: str, texts: list[str]) -> str:
    tweets = "\n".join(f"{i}. {text}" for i, text in enumerate(texts, start=1))
    return f"""Label relevance between the query and each tweet.
        Return only one Python list of integers.
        No explanation. No bullet points. No analysis. No extra text.
        Return:
        1 = relevant
        0 = not relevant

        Be strict. If uncertain, choose 0.
        The list length MUST equal the number of tweets.
        The output list order MUST exactly match the tweet order above.
        The first output value corresponds to tweet 1, the second to tweet 2, and so on.

        Query: "{query}"

        Tweets:
        {tweets}

        Example output:
        [1, 0, 0]
    """


# Calls the LLM until a valid label list is obtained
def call_llm_until_valid(
    call_fn, prompt: str, expected_count: int, allowed_values: set[int]
) -> list[int]:
    for _ in range(MAX_RETRIES):
        response_text = call_fn(prompt)
        text = response_text.strip()
        candidates = re.findall(r"\[[^\[\]]*\]", text)

        for candidate in reversed(candidates):
            content = candidate[1:-1].strip()
            parts = [] if not content else [part.strip() for part in content.split(",")]

            try:
                labels = [int(part) for part in parts]
            except ValueError:
                continue

            if len(labels) == expected_count and all(
                label in allowed_values for label in labels
            ):
                return labels
        time.sleep(1)

    raise ValueError("LLM output was not a valid label list after max retries.")


# Label a query with relevance scores
def label_query(query: str, results: list[dict]) -> list[dict]:
    # Extract document IDs and texts from the results
    doc_ids = [result["doc_id"] for result in results]
    texts = [result["text"] for result in results]

    labels = call_llm_until_valid(
        call_gemma,
        build_gemma_prompt(query, texts),
        len(texts),
        {-1, 0, 1},  # -1 = unsure, 0 = not relevant, 1 = relevant
    )

    # Handle unsure labels by re-evaluating them with gemini model
    unsure_positions = [i for i, label in enumerate(labels) if label == -1]
    if unsure_positions:
        unsure_texts = [texts[i] for i in unsure_positions]
        gemini_labels = call_llm_until_valid(
            call_gemini,
            build_gemini_prompt(query, unsure_texts),
            len(unsure_texts),
            {0, 1},
        )
        for i, new_label in zip(unsure_positions, gemini_labels):
            labels[i] = new_label

    return [(doc_id, int(label)) for doc_id, label in zip(doc_ids, labels)]


# Label all queries in a list
def label_all(query_results_list: list[dict]) -> list[dict]:
    labeled_rows = []
    for item in query_results_list:
        labeled_rows.extend(label_query(item["query"], item["results"]))
    return labeled_rows
