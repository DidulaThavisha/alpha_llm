"""Load and manage Codeforces competitive programming problems."""

import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class CodeProblem:
    prompt: str
    python_solution: str
    answer: str  # compressed test cases
    dataset: str
    problem_type: str
    rating: int

    @property
    def title(self) -> str:
        """Extract problem title from prompt."""
        for line in self.prompt.split("\n"):
            if line.startswith("Title:"):
                return line.replace("Title:", "").strip()
        return "Unknown"


def load_problems(path: str = "codeforces_one_per_rating.json") -> List[CodeProblem]:
    """Load all problems from the dataset JSON file."""
    with open(path) as f:
        raw = json.load(f)

    problems = []
    for entry in raw:
        problems.append(CodeProblem(
            prompt=entry["prompt"],
            python_solution=entry["python"],
            answer=entry["answer"],
            dataset=entry["dataset"],
            problem_type=entry["type"],
            rating=entry["rating"],
        ))

    # Sort by rating (easy → hard)
    problems.sort(key=lambda p: p.rating)
    return problems


def get_problems_by_max_rating(problems: List[CodeProblem], max_rating: int) -> List[CodeProblem]:
    """Filter problems up to a maximum rating (for curriculum learning)."""
    return [p for p in problems if p.rating <= max_rating]
