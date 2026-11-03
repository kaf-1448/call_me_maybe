import json
import numpy as np
from pydantic import ValidationError, BaseModel
import sys
from src.models.definitionmodel import FunctionDefinition, PromptModel
from src.models.definitionmodel import FunctionCallResult
from llm_sdk import Small_LLM_Model
from typing import Any, Dict, List


def load_json_file(path_file: str,
                   model: type[BaseModel]) -> List[Any] | None:
    try:
        with open(path_file, 'r', encoding="utf-8") as f:
            data = json.load(f)
            fde = [model(**item) for item in data]

        return fde

    except ValidationError as e:
        print(f"Error : {e.errors()[0]['msg']}")

    except FileNotFoundError:
        print(f"File not found : {path_file}")
    except json.JSONDecodeError as e:
        print(f"Message: {e.msg}")


def build_param_prompt(
    question: str,
    func_def: FunctionDefinition,
    param_name: str,
    param_type: str,
    extracted: dict,
    quote: str,
) -> str:
    all_params = ", ".join(
        [f"{k} ({v.type})" for k, v in func_def.parameters.items()]
    )

    lines = [
        "Extract the parameter value from the Query.",
        "If the parameter is 'regex' and the query describes a pattern (like numbers, digits, vowels, or spaces), write the regular expression pattern.",
        "",
        "Query: What is 7 plus -2.5?",
        "Function: fn_add_numbers - Add two numbers",
        "All parameters: a (number), b (number)",
        "Extracted so far: {'a': 7.0}",
        "Target parameter: b (number)",
        "Value: -2.5",
        "",
        "Query: Replace all vowels in 'Hello World' with *",
        "Function: fn_substitute_string_with_regex - Replace regex matches in a string",
        "All parameters: source_string (string), regex (string), replacement (string)",
        "Extracted so far: {'source_string': 'Hello World'}",
        "Target parameter: regex (string)",
        'Value: "[aeiouAEIOU]"',
        "",
        "Query: Replace all vowels in 'Hello World' with *",
        "Function: fn_substitute_string_with_regex - Replace regex matches in a string",
        "All parameters: source_string (string), regex (string), replacement (string)",
        "Extracted so far: {'source_string': 'Hello World', 'regex': '[aeiouAEIOU]'}",
        "Target parameter: replacement (string)",
        'Value: "*"',
        "",
        "Query: In 'the fox ran', replace 'fox' with 'bear'",
        "Function: fn_substitute_string_with_regex - Replace regex matches in a string",
        "All parameters: source_string (string), regex (string), replacement (string)",
        "Extracted so far: {'source_string': 'the fox ran'}",
        "Target parameter: regex (string)",
        'Value: "fox"',
        "",
        "Query: In 'the fox ran', replace 'fox' with 'bear'",
        "Function: fn_substitute_string_with_regex - Replace regex matches in a string",
        "All parameters: source_string (string), regex (string), replacement (string)",
        "Extracted so far: {'source_string': 'the fox ran', 'regex': 'fox'}",
        "Target parameter: replacement (string)",
        'Value: "bear"',
        "",
        f"Query: {question}",
        f"Function: {func_def.name} - {func_def.description}",
        f"All parameters: {all_params}",
        f"Extracted so far: {extracted}",
        f"Target parameter: {param_name} ({param_type})",
        f"Value: {quote}",
    ]
    return "\n".join(lines)


def main():
    definition_file = sys.argv[1]
    test_file = sys.argv[2]
    df = load_json_file(definition_file, FunctionDefinition)
    tf = load_json_file(test_file, PromptModel)
    allowd_function = [f.name for f in df]

    funcs_text = "\n".join([f"- {f.name}: {f.description}" for f in df])

    prompt_rs = [pt.prompt for pt in tf]
    call_restult = []

    l = Small_LLM_Model()
    # ===========================================================
    digit = ["0", "1", "2", "3", "4",
             "5", "6", "7", "8", "9", ".", " ", " -"]
    conv_all_par = [l.encode(d)[0].tolist()[0] for d in digit]
    # ===========================================================

    for user_question in prompt_rs:

        full_prompt = f"""Available functions:
        {funcs_text}
        User query: {user_question}
        Function to call: """

        input_ids = l.encode(full_prompt)

        f_ids = []
        for f in allowd_function:
            f_ids_conv = l.encode(f)
            f_ids.append(f_ids_conv[0].tolist())

        # print(f_ids)
        name_funtion = ""
        tokens_ids = input_ids[0].tolist()
        # print(tokens_ids)
        i = 0

        while 1:

            logits = l.get_logits_from_input_ids(tokens_ids)

            allowed_starts = [sublist[i]
                              for sublist in f_ids if i < len(sublist)]

            next_ids = max(allowed_starts, key=lambda x: logits[x])

            # print(next_ids)
            next_token = l.decode([next_ids])
            # print(f"next token is {next_token}")
            name_funtion += next_token
            tokens_ids.append(next_ids)

            # print(f_ids)

            f_ids = [li for li in f_ids if i <
                     len(li) and li[i] == next_ids]
            # print(tokens_ids)
            i += 1

            if name_funtion in allowd_function:
                break

        # print(name_funtion)

        selected_function = next(f for f in df if f.name == name_funtion)
        extracted_parameters = {}

        for para_name, para_info in selected_function.parameters.items():
            quote = '"' if para_info.type == "string" else ""
            value = ""
            # prompt_par = f'Query: "{user_question}"\nParameter "{para_name}" value: {quote}'
            prompt_par = f"""Query: "{user_question}"
            Function: {name_funtion}
            Extracted: {extracted_parameters}
            Parameter '{para_name}': {quote}"""
            convert_par = l.encode(prompt_par)[0].tolist()

            if para_info.type == "number":

                # print("space-minus:",
                #       float(logits[481]), "two:", float(logits[17]))

                while 1:

                    logits = l.get_logits_from_input_ids(convert_par)

                    value_id = max(conv_all_par, key=lambda x: logits[x])
                    # value_id = int(np.argmax(logits))

                    value += l.decode([value_id])
                    # print("next is ", value)
                    convert_par.append(value_id)

                    if value_id == 220 and len(value.strip()) > 0:
                        break
                    elif "." in value and len(value.split(".")[1]) > 0:
                        break

                extracted_parameters[para_name] = float(value.strip())

            else:

                while 1:

                    logits = l.get_logits_from_input_ids(convert_par)

                    value_id = max(range(len(logits)), key=lambda x: logits[x])

                    new_value_id = l.decode([value_id])
                    # if '"' in new_value_id or '\n' in new_value_id:
                    #     break
                    # لداخل فـ الـ while د string:
                    if len(value) > 60 or '"' in new_value_id:
                        break

                    value += new_value_id
                    # print("next is ", value)
                    convert_par.append(value_id)

                extracted_parameters[para_name] = value

        print(extracted_parameters)
        result_obj = FunctionCallResult(
            prompt=user_question,
            name=name_funtion,
            parameters=extracted_parameters
        )
        call_restult.append(result_obj.model_dump())

    import os
    os.makedirs("data/output", exist_ok=True)

    with open("data/output/function_calls.json", "w", encoding="utf-8") as f:
        json.dump(call_restult, f, indent=2)

    print("Saved all results to data/output/function_calls.json!")


if __name__ == "__main__":
    main()
