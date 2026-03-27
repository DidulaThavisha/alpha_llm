"""Sandboxed code execution via subprocess.

Runs generated Python code against test cases with timeout and captures stdout.
"""

import subprocess
import tempfile
import os
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    passed: int
    total: int
    errors: List[str]
    outputs: List[Optional[str]]


class CodeExecutor:
    def __init__(self, timeout: int = 10, max_memory_mb: int = 256):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    def run_single(self, code: str, input_data: str) -> Tuple[Optional[str], Optional[str]]:
        """Run code with given input. Returns (stdout, error_msg)."""
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode != 0:
                return None, result.stderr[:500]
            return result.stdout, None
        except subprocess.TimeoutExpired:
            return None, "TIMEOUT"
        except Exception as e:
            return None, str(e)[:500]

    def evaluate(self, code: str, test_cases: List[Dict[str, str]]) -> ExecutionResult:
        """Run code against all test cases. Returns ExecutionResult."""
        passed = 0
        errors = []
        outputs = []

        for tc in test_cases:
            stdout, error = self.run_single(code, tc["input"])
            outputs.append(stdout)

            if error:
                errors.append(error)
                continue

            actual = stdout.strip() if stdout else ""
            expected = tc["output"].strip()

            if actual == expected:
                passed += 1
            else:
                errors.append(f"Expected: {expected[:100]}, Got: {actual[:100]}")

        return ExecutionResult(
            passed=passed,
            total=len(test_cases),
            errors=errors[:10],  # cap error messages
            outputs=outputs,
        )
