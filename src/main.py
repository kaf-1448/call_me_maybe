import argparse
import json
import os
import sys
from typing import Any, Dict, List, Type, TypeVar
import numpy as np
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
                print(
                    f"Error: Expected a JSON list in {path_file}",
                    file=sys.stderr
                )
                return None
            return [model(**item) for item in data]
    except ValidationError as e:
        print(
            f"Validation Error in {path_file}: {e.errors()[0]['msg']}",
            file=sys.stderr
        )
    except FileNotFoundError:
        print(f"File not found: {path_file}", file=sys.stderr)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {path_file}: {e.msg}", file=sys.stderr)
    except OSError as e:
        print(f"OS Error reading {path_file}: {e}", file=sys.stderr)
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
    """Select the target function name using token-by-token
    constrained decoding.

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


def extract_number(
    model: Small_LLM_Model,
    convert_par: List[int],
    digit_ids: List[int],
    stop_ids: set[int],
) -> float:
    """Extract a valid float parameter (positive or negative)
    using constrained decoding."""
    value = ""
    while True:
        logits = model.get_logits_from_input_ids(convert_par)
        valid_ids = []
        for tid in digit_ids:
            tok_str = model.decode([tid])
            if "-" in tok_str and len(value.strip()) > 0:
                continue

            if "." in tok_str and ("." in value or not
                                   any(c.isdigit() for c in value)):
                continue
            valid_ids.append(tid)

        value_id = max(valid_ids, key=lambda x: logits[x])

        if value_id in stop_ids and any(c.isdigit() for c in value):
            break

        value += model.decode([value_id])
        convert_par.append(value_id)

        if "." in value and len(value.split(".")[1]) >= 2:
            break
        if len(value) > 15:
            break

    try:
        return float(value.strip().replace(" ", ""))
    except ValueError:
        return 0.0


def main() -> None:
    """Run the main function calling pipeline."""
    args = parse_args()
    df = load_json_file(args.functions_definition, FunctionDefinition)
    tf = load_json_file(args.input, PromptModel)

    if df is None or tf is None or len(df) == 0:
        sys.exit(1)

    funcs_text = "\n".join([f"- {f.name}: {f.description}" for f in df])
    call_results: List[Dict[str, Any]] = []

    l_model = Small_LLM_Model()

    vocab_path = l_model.get_path_to_vocab_file()
    try:
        with open(vocab_path, "r", encoding="utf-8") as vf:
            _ = json.load(vf)
    except OSError:
        pass

    digit_strs = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        " 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
        ".", "-", " -", " ", "\n",
        "-1", "-2", "-3", "-5", "-2.5", "-3.5", "-10",
        " -1", " -2", " -3", " -2.5", " -3.5"
    ]
    conv_all_par = list({l_model.encode(d)[0].tolist()[0] for d in digit_strs})
    stop_num_ids = {
        l_model.encode(" ")[0].tolist()[0],
        l_model.encode("\n")[0].tolist()[0],
        220,
    }
    bool_true_id = l_model.encode("true")[0].tolist()[0]
    bool_false_id = l_model.encode("false")[0].tolist()[0]

    base_f_ids = {f.name: l_model.encode(f.name)[0].tolist() for f in df}

    for pt in tf:
        user_question = pt.prompt
        full_prompt = (
            f"Available functions:\n{funcs_text}\n"
            f"User query: {user_question}\n"
            f"Function to call: "
        )

        name_function = select_function(l_model, full_prompt, base_f_ids)
        selected_function = next(f for f in df if f.name == name_function)
        extracted_parameters: Dict[str, Any] = {}

        for para_name, para_info in selected_function.parameters.items():
            if para_info.type == "number":
                prompt_par = (
                    'Query: "Calculate 10 plus -3.5"\n'
                    "Function: fn_add_numbers\n"
                    "Extracted: {'a': 10.0}\n"
                    "Parameter 'b': -3.5\n\n"
                    f'Query: "{user_question}"\n'
                    f"Function: {name_function}\n"
                    f"Extracted: {extracted_parameters}\n"
                    f"Parameter '{para_name}': "
                )
                convert_par = l_model.encode(prompt_par)[0].tolist()
                extracted_parameters[para_name] = extract_number(
                    l_model, convert_par, conv_all_par, stop_num_ids
                )

            elif para_info.type == "boolean":
                prompt_par = (
                    f'Query: "{user_question}"\n'
                    f"Function: {name_function}\n"
                    f"Extracted: {extracted_parameters}\n"
                    f"Parameter '{para_name}' (true/false): "
                )
                convert_par = l_model.encode(prompt_par)[0].tolist()
                logits = l_model.get_logits_from_input_ids(convert_par)
                extracted_parameters[para_name] = (
                    logits[bool_true_id] >= logits[bool_false_id]
                )

            else:
                prompt_par = (
                    'Query: "Replace all numbers in \'Hi 12\' with X"\n'
                    "Extracted: {'source_string': 'Hi 12'}\n"
                    'Parameter \'regex\': "[0-9]+"\n\n'
                    'Query: "Replace all vowels in \'Hello\' with asterisks"\n'
                    "Extracted: {'source_string': 'Hello'}\n"
                    'Parameter \'regex\': "[aeiouAEIOU]"\n\n'
                    'Query: "Replace all vowels in \'Hello\' with asterisks"\n'
                    "Extracted: {'source_string': 'Hello', 'regex': "
                    "'[aeiouAEIOU]'}\n"
                    'Parameter \'replacement\': "*" not " *"\n\n'
                    'Query: "Replace \'cat\' with \'dog\' in \'the cat\'"\n'
                    "Extracted: {'source_string': 'the cat'}\n"
                    'Parameter \'regex\': "cat"\n\n'
                    f'Query: "{user_question}"\n'
                    f"Extracted: {extracted_parameters}\n"
                    f"Parameter '{para_name}': \""
                )
                convert_par = l_model.encode(prompt_par)[0].tolist()
                value = ""

                while True:
                    logits = l_model.get_logits_from_input_ids(convert_par)
                    value_id = int(np.argmax(logits))
                    new_value_id = l_model.decode([value_id])

                    if '"' in new_value_id:
                        value += new_value_id.split('"')[0]
                        break
                    if "\n" in new_value_id or len(value) > 60:
                        break

                    value += new_value_id
                    convert_par.append(value_id)

                extracted_parameters[para_name] = value

        print(extracted_parameters)
        result_obj = FunctionCallResult(
            prompt=user_question,
            name=name_function,
            parameters=extracted_parameters,
        )
        call_results.append(result_obj.model_dump())

    try:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(call_results, f, indent=2)

    except OSError as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.", file=sys.stderr)
        sys.exit(130)
