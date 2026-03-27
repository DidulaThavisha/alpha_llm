"""Prompt formatting for the code generation model."""


def format_prompt(problem_text: str) -> str:
    """Format a problem into a prompt for the model.

    Returns a string that ends with the start of a Python code block,
    so the model's generation IS the solution code.
    """
    return (
        f"Solve the following competitive programming problem in Python. "
        f"Read input from stdin and print output to stdout.\n\n"
        f"{problem_text}\n\n"
        f"```python\n"
    )


def extract_code(full_text: str) -> str:
    """Extract Python code from model output.

    Handles the case where the model generates a closing ``` marker.
    """
    # Find the code after ```python\n
    marker = "```python\n"
    idx = full_text.rfind(marker)
    if idx >= 0:
        code = full_text[idx + len(marker):]
    else:
        code = full_text

    # Remove trailing ``` if present
    end_marker = "```"
    end_idx = code.rfind(end_marker)
    if end_idx >= 0:
        code = code[:end_idx]

    return code.strip()
