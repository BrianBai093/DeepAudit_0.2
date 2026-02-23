from __future__ import annotations

GUIDE_SYSTEM_PROMPT = (
    "Role: You are a meticulous Software Test Engineer with expertise in machine learning. "
    "Task: analyze paper content and select Code-Level Guidance only. "
    "Return JSON only."
)

GUIDE_USER_PROMPT_TEMPLATE = (
    "Guide Extraction Stage\n"
    "Goal: identify sentence indices containing code-level reproducibility guidance.\n"
    "Select only items that describe: data/task, preprocessing, hyperparameters, architecture, algorithms, or metrics.\n"
    "Ignore abstract claims, future/related work, and generic background.\n"
    "Input Paper Segment (numbered):\n{paper_segment}\n\n"
    "Return ONLY a JSON array of integers, e.g. [1, 4, 9]."
)

ATOMIC_SYSTEM_PROMPT = (
    "Role: You are an expert technical writer specializing in executable specifications. "
    "Task: decompose facts into atomic verifiable criteria. Return JSON only."
)

ATOMIC_USER_PROMPT_TEMPLATE = (
    "Atomicity Standardization Stage\n"
    "Instructions:\n"
    "1) Decompose complex statements into smallest verifiable units.\n"
    "2) Keep equations/loss formulas intact as a single fact.\n"
    "3) Every criterion MUST contain <fact>...</fact> and <scope>...</scope>.\n"
    "Input Facts:\n{extracted_sentences}\n\n"
    "Return a JSON list of objects: [{{\"criterion\": \"...<fact>...</fact>...<scope>...</scope>...\"}}]."
)

FILTER_SYSTEM_PROMPT = (
    "Role: You are a QA Lead Engineer. "
    "Task: select the best representative among semantically similar criteria for direct verifiability. "
    "Return JSON only."
)

FILTER_USER_PROMPT_TEMPLATE = (
    "Filtering & Deduplication Stage\n"
    "Selection principles:\n"
    "1) Directly verifiable in code/logs.\n"
    "2) Unambiguous wording.\n"
    "3) Complete scope.\n"
    "Input Group:\n{cluster_of_similar_criteria}\n\n"
    "Return JSON object: {{\"selected_index\": int, \"reason\": str}}."
)
