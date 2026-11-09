from llm_sdk import Small_LLM_Model
import json


llm = Small_LLM_Model()
path = llm.get_path_to_vocab_file()
print(path)


with open(path, 'r') as pv:
    vacab = json.load(pv)

print(type(vacab))


digit_vocab = {}
valid_numbers = set("123456789.-")
for token, token_id in vacab.items():
    if token in valid_numbers:
        digit_vocab[token] = token_id


# print(digit_vocab)
digit_str_ids = list(digit_vocab.values())
print(digit_str_ids)
