import json
import os
import sys
import numpy as np
from typing import Any, List
from pydantic import ValidationError, BaseModel
from src.models.definitionmodel import FunctionDefinition, PromptModel
from src.models.definitionmodel import FunctionCallResult
from llm_sdk import Small_LLM_Model


def load_json_file(path_file: str,
                   model: type[BaseModel]) -> List[Any] | None:
    try:
        with open(path_file, 'r', encoding="utf-8") as f:
            data = json.load(f)
            return [model(**item) for item in data]
    except ValidationError as e:
        print(f"Error : {e.errors()[0]['msg']}")
    except FileNotFoundError:
        print(f"File not found : {path_file}")
    except json.JSONDecodeError as e:
        print(f"Message: {e.msg}")
    return None


def main():
    definition_file = sys.argv[1]
    test_file = sys.argv[2]
    df = load_json_file(definition_file, FunctionDefinition)
    tf = load_json_file(test_file, PromptModel)

    if not df or not tf:
        sys.exit(1)

    allowd_function = [f.name for f in df]
    funcs_text = "\n".join([f"- {f.name}: {f.description}" for f in df])
    prompt_rs = [pt.prompt for pt in tf]
    call_restult = []

    l = Small_LLM_Model()

    # Pre-encode digits AND function names ONCE outside the loop
    digit = ["0", "1", "2", "3", "4", "5", "6",
             "7", "8", "9", ".", "-", " ", " -", "\n"]
    conv_all_par = list({l.encode(d)[0].tolist()[0] for d in digit})
    stop_num_ids = {l.encode(" ")[0].tolist()[0],
                    l.encode("\n")[0].tolist()[0], 220}

    base_f_ids = {f.name: l.encode(f.name)[0].tolist() for f in df}

    for user_question in prompt_rs:
        full_prompt = (
            f"Available functions:\n{funcs_text}\n"
            f"User query: {user_question}\n"
            f"Function to call: "
        )

        tokens_ids = l.encode(full_prompt)[0].tolist()
        # Keep track of (function_name, token_id_list)
        candidates = list(base_f_ids.items())
        i = 0

        while True:
            # FAST EXIT: If only 1 function matches the prefix so far, pick it immediately!
            if len(candidates) == 1:
                name_funtion = candidates[0][0]
                break

            allowed_starts = list(
                {tokens[i] for _, tokens in candidates if i < len(tokens)})

            # If all candidates share the exact same next token (e.g. "fn_"), skip LLM call!
            if len(allowed_starts) == 1:
                next_ids = allowed_starts[0]
            else:
                logits = l.get_logits_from_input_ids(tokens_ids)
                next_ids = max(allowed_starts, key=lambda x: logits[x])

            tokens_ids.append(next_ids)
            candidates = [
                (name, tokens) for name, tokens in candidates
                if i < len(tokens) and tokens[i] == next_ids
            ]
            i += 1

            # Fallback check if exact match reached
            decoded_so_far = l.decode(
                tokens_ids[len(l.encode(full_prompt)[0]):])
            if decoded_so_far in allowd_function:
                name_funtion = decoded_so_far
                break

        selected_function = next(f for f in df if f.name == name_funtion)
        extracted_parameters = {}

        for para_name, para_info in selected_function.parameters.items():
            quote = '"' if para_info.type == "string" else ""
            value = ""

            # Short, fast prompt (adds a 1-line hint only when extracting regex)
            regex_hint = (
                'Example: vowels -> "[aeiouAEIOU]", digits -> "[0-9]"\n'
                if para_name == "regex" else ""
            )
            prompt_par = (
                f'{regex_hint}Query: "{user_question}"\n'
                f'Function: {name_funtion}\n'
                f'Extracted: {extracted_parameters}\n'
                f"Parameter '{para_name}': {quote}"
            )
            convert_par = l.encode(prompt_par)[0].tolist()

            if para_info.type == "number":
                while True:
                    logits = l.get_logits_from_input_ids(convert_par)
                    value_id = max(conv_all_par, key=lambda x: logits[x])

                    if value_id in stop_num_ids and any(c.isdigit() for c in value):
                        break

                    value += l.decode([value_id])
                    convert_par.append(value_id)

                    if "." in value and len(value.split(".")[1]) >= 2:
                        break
                    if len(value) > 15:
                        break

                extracted_parameters[para_name] = float(value.strip())

            else:
                while True:
                    logits = l.get_logits_from_input_ids(convert_par)
                    # np.argmax is ~100x faster than max(range(len(logits)))
                    value_id = int(np.argmax(logits))
                    new_value_id = l.decode([value_id])

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
            name=name_funtion,
            parameters=extracted_parameters
        )
        call_restult.append(result_obj.model_dump())

    os.makedirs("data/output", exist_ok=True)
    with open("data/output/function_calls.json", "w", encoding="utf-8") as f:
        json.dump(call_restult, f, indent=2)

    print("Saved all results to data/output/function_calls.json!")


if __name__ == "__main__":
    main()
