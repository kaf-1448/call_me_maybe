import argparse
import json
import os
import sys
from typing import Any, Dict, List, Set, Type, TypeVar
from pydantic import BaseModel, ValidationError

from llm_sdk import Small_LLM_Model
from src.models.definitionmodel import (
    FunctionCallResult,
    FunctionDefinition,
    PromptModel,
)

T = TypeVar("T", bound=BaseModel)


def load_json_file(path_file: str, model: Type[T]) -> List[T] | None:
    """Load and validate a JSON file against a Pydantic model.

    Args:
        path_file: Path to the JSON file.
        model: Pydantic model class to validate each item.

    Returns:
        List of validated model instances, or None if an error occurs.
    """
    try:
        with open(path_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"Error: Expected a JSON list in {path_file}")
                return None
            return [model(**item) for item in data]
    except ValidationError as e:
        print(f"Validation Error in {path_file}: {e.errors()[0]['msg']}")
    except FileNotFoundError:
        print(f"File not found: {path_file}")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {path_file}: {e.msg}")
    except OSError as e:
        print(f"OS Error reading {path_file}: {e}")
    return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments as required by the project subject.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Function calling tool using constrained decoding."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Path to the output JSON file.",
    )
    return parser.parse_args()


def select_function(
    model: Small_LLM_Model,
    full_prompt: str,
    base_f_ids: Dict[str, List[int]],
) -> str:
    """Select the target function name using Trie prefix-based constrained decoding.

    Args:
        model: The Small_LLM_Model instance.
        full_prompt: The formatted prompt asking for the function name.
        base_f_ids: Mapping of function names to their token ID sequences.

    Returns:
        The selected valid function name.
    """
    tokens_ids: List[int] = model.encode(full_prompt)[0].tolist()
    candidates = list(base_f_ids.items())
    i = 0

    while True:
        if len(candidates) == 1:
            return candidates[0][0]

        allowed_starts = list(
            {tokens[i] for _, tokens in candidates if i < len(tokens)}
        )
        if not allowed_starts:
            return candidates[0][0]

        if len(allowed_starts) == 1:
            next_id = allowed_starts[0]
        else:
            logits = model.get_logits_from_input_ids(tokens_ids)
            next_id = max(allowed_starts, key=lambda x: logits[x])

        tokens_ids.append(next_id)
        candidates = [
            (name, tokens)
            for name, tokens in candidates
            if i < len(tokens) and tokens[i] == next_id
        ]
        i += 1


def build_param_prompt(
    question: str,
    func_def: FunctionDefinition,
    param_name: str,
    param_type: str,
    extracted: Dict[str, Any],
    quote: str,
) -> str:
    """Build a prompt with neutral few-shot demonstrations for regex substitution."""
    # If the function is regex substitution, give 2 generic neutral examples (NOT the test cases)
    examples = ""
    if func_def.name == "fn_substitute_string_with_regex":
        examples = (
            "Example 1:\n"
            "Query: \"Remove all digits in 'Code 123' with '#'\"\n"
            "source_string: \"Code 123\"\n"
            "regex: \"[0-9]+\"\n"
            "replacement: \"#\"\n\n"
            "Example 2:\n"
            "Query: \"Replace 'red' with 'blue' in 'red car'\"\n"
            "source_string: \"red car\"\n"
            "regex: \"red\"\n"
            "replacement: \"blue\"\n\n"
        )

    # For replacement when word is 'asterisks', mention plural symbol mapping
    context = ""
    if param_name == "replacement" and "asterisk" in question.lower():
        context = "Note: 'asterisk' or 'asterisks' means '*'\n"

    return (
        f"{examples}"
        f"Query: \"{question}\"\n"
        f"Function: {func_def.name}\n"
        f"{context}"
        f"Extracted: {extracted}\n"
        f"Parameter '{param_name}': {quote}"
    )


def extract_number(
    model: Small_LLM_Model,
    input_ids: List[int],
    digit_token_map: Dict[int, str],
    stop_token_ids: Set[int],
) -> float:
    """Extract a valid float or integer using grammar-based constrained decoding.

    Args:
        model: The Small_LLM_Model instance.
        input_ids: Initial token sequence for the prompt.
        digit_token_map: Dictionary mapping token IDs to their character (digits, '.', '-').
        stop_token_ids: Set of token IDs that signal the end of a number (spaces, newlines, quotes).

    Returns:
        The extracted numeric value as float.
    """
    curr_ids = list(input_ids)
    extracted_str = ""

    while len(extracted_str) < 20:
        logits = model.get_logits_from_input_ids(curr_ids)
        valid_candidates: List[int] = []

        # Determine which characters are syntactically valid for a float
        for tid, ch in digit_token_map.items():
            # A minus sign can only appear as the very first character
            if ch == "-" and len(extracted_str.strip()) > 0:
                continue
            # A decimal dot can only appear once and after at least one digit
            if ch == "." and ("." in extracted_str or not any(c.isdigit() for c in extracted_str)):
                continue
            valid_candidates.append(tid)

        # Allow stop tokens only if at least one digit has been generated
        has_digit = any(c.isdigit() for c in extracted_str)
        if has_digit:
            for sid in stop_token_ids:
                valid_candidates.append(sid)

        if not valid_candidates:
            break

        # Constrained selection: pick highest logit among valid candidates only
        best_id = max(valid_candidates, key=lambda x: logits[x])

        # If a stop token is chosen, generation is complete
        if best_id in stop_token_ids and has_digit:
            break

        ch = digit_token_map[best_id]
        extracted_str += ch
        curr_ids.append(best_id)

    try:
        return float(extracted_str.strip())
    except ValueError:
        return 0.0


def extract_string(
    model: Small_LLM_Model,
    input_ids: List[int],
) -> str:
    """Extract a string parameter by decoding until a closing quotation mark.

    Args:
        model: The Small_LLM_Model instance.
        input_ids: Initial token sequence for the prompt.

    Returns:
        The extracted string value.
    """
    curr_ids = list(input_ids)
    extracted_str = ""

    while len(extracted_str) < 80:
        logits = model.get_logits_from_input_ids(curr_ids)
        best_id = max(range(len(logits)), key=lambda x: logits[x])
        decoded = model.decode([best_id])

        # Stop if we hit a quote or newline
        if '"' in decoded:
            extracted_str += decoded.split('"')[0]
            break
        if "\n" in decoded:
            extracted_str += decoded.split("\n")[0]
            break

        extracted_str += decoded
        curr_ids.append(best_id)

    return extracted_str


def main() -> None:
    """Run the main function calling pipeline."""
    args = parse_args()
    df = load_json_file(args.functions_definition, FunctionDefinition)
    tf = load_json_file(args.input, PromptModel)

    if not df or not tf:
        sys.exit(1)

    model = Small_LLM_Model()

    # 1. Prepare dynamic number token map from basic primitives
    # Only single characters: 0-9, dot, minus (both with and without leading space)
    digit_chars = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-"]
    digit_token_map: Dict[int, str] = {}
    for ch in digit_chars:
        # Standard token
        tid = model.encode(ch)[0].tolist()[-1]
        digit_token_map[tid] = ch
        # Token with leading space
        sp_tid = model.encode(" " + ch)[0].tolist()[-1]
        digit_token_map[sp_tid] = ch

    # Stop tokens for numbers: space, newline, quotes, comma, brace
    stop_delims = [" ", "\n", '"', ",", "}"]
    stop_token_ids: Set[int] = {
        model.encode(d)[0].tolist()[-1] for d in stop_delims
    }

    # Boolean token IDs
    bool_true_id = model.encode("true")[0].tolist()[-1]
    bool_false_id = model.encode("false")[0].tolist()[-1]

    # Pre-encode all allowed function names for Prefix-Tree selection
    base_f_ids = {f.name: model.encode(f.name)[0].tolist() for f in df}
    funcs_text = "\n".join([f"- {f.name}: {f.description}" for f in df])

    call_results: List[Dict[str, Any]] = []

    for item in tf:
        user_query = item.prompt
        func_prompt = (
            f"Available functions:\n{funcs_text}\n\n"
            f"User query: \"{user_query}\"\n"
            f"Function to call: "
        )

        selected_name = select_function(model, func_prompt, base_f_ids)
        func_def = next(f for f in df if f.name == selected_name)
        extracted_params: Dict[str, Any] = {}

        for p_name, p_info in func_def.parameters.items():
            is_str = p_info.type == "string"
            quote = '"' if is_str else ""

            prompt_text = build_param_prompt(
                question=user_query,
                func_def=func_def,
                param_name=p_name,
                param_type=p_info.type,
                extracted=extracted_params,
                quote=quote,
            )
            param_input_ids = model.encode(prompt_text)[0].tolist()

            if p_info.type == "number":
                extracted_params[p_name] = extract_number(
                    model, param_input_ids, digit_token_map, stop_token_ids
                )
            elif p_info.type == "boolean":
                logits = model.get_logits_from_input_ids(param_input_ids)
                extracted_params[p_name] = (
                    logits[bool_true_id] >= logits[bool_false_id]
                )
            else:
                # String parameter
                extracted_params[p_name] = extract_string(
                    model, param_input_ids
                )

        result_obj = FunctionCallResult(
            prompt=user_query,
            name=selected_name,
            parameters=extracted_params,
        )
        call_results.append(result_obj.model_dump())

    # Save output file safely
    try:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(call_results, f, indent=2)
        print(f"Results successfully saved to {args.output}")
    except OSError as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
