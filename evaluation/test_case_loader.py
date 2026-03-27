"""Decode compressed test cases from the dataset's answer field.

Pipeline: base64 → zlib decompress → pickle loads → JSON parse
Result: list of {"input": str, "output": str} dicts.
"""

import base64
import json
import pickle
import zlib
from typing import List, Dict


def load_test_cases(answer_field: str) -> List[Dict[str, str]]:
    """Decode a problem's answer field into test cases."""
    raw = base64.b64decode(answer_field)
    decompressed = zlib.decompress(raw)
    json_str = pickle.loads(decompressed)
    test_cases = json.loads(json_str)
    return test_cases


def load_all_test_cases(problems: list) -> Dict[int, List[Dict[str, str]]]:
    """Load test cases for all problems, keyed by rating."""
    result = {}
    for problem in problems:
        rating = problem["rating"]
        result[rating] = load_test_cases(problem["answer"])
    return result
