from __future__ import annotations

GUIDE_SYSTEM_PROMPT = (
    "Role: You are a meticulous ML reproducibility auditor. "
    "Task: select sentences/tables that contain numeric results or execution parameters. "
    "Return JSON only."
)

GUIDE_USER_PROMPT_TEMPLATE = (
    "Guide Extraction Stage\n"
    "Goal: identify sentence indices containing reported numeric results OR execution parameters.\n"
    "Select ONLY items that describe:\n"
    "  - Reported numeric results: accuracy, F1, loss, AUC, BLEU, precision, recall, MSE, MAE, or any metric with a concrete number\n"
    "  - Execution parameters: dataset name/size/split, epochs, learning rate, batch size, seed, optimizer, dropout, weight decay\n"
    "IGNORE: method descriptions, architecture explanations, algorithm narratives, related work, abstract claims, references, author info, background.\n"
    "Input Paper Segment (numbered):\n{paper_segment}\n\n"
    "Return ONLY a JSON array of integers, e.g. [1, 4, 9]."
)

ATOMIC_SYSTEM_PROMPT = (
    "Role: You are an ML reproducibility auditor extracting verifiable facts. "
    "Task: extract numeric results and execution parameters as atomic criteria. Return JSON only."
)

ATOMIC_USER_PROMPT_TEMPLATE = (
    "Atomic Extraction Stage\n"
    "Instructions:\n"
    "1) Extract ONLY numeric results (metric = value) and concrete execution parameters (setting = value).\n"
    "2) Each criterion MUST contain <fact>...</fact> and <scope>...</scope>.\n"
    "3) For results: fact = 'metric_name = value' (e.g. 'accuracy = 92.3%'). scope = dataset/experiment context.\n"
    "4) For execution params: fact = 'parameter = value' (e.g. 'learning rate = 0.001'). scope = experiment context.\n"
    "5) SKIP method descriptions, architecture details, algorithm explanations.\n"
    "Input Facts:\n{extracted_sentences}\n\n"
    'Return a JSON list: [{{"criterion": "...<fact>...</fact>...<scope>...</scope>..."}}].'
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
