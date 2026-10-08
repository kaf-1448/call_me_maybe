# Project Structure — Call Me Maybe

This document outlines the architecture, directory structure, and responsibilities of each module for the **Call Me Maybe** function calling project.

---

## 📁 Directory Tree Overview

```text
call_me_maybe/
├── .gitignore                      # Git ignore rules (ignore cache, output/, venv)
├── Makefile                        # Automation (install, run, debug, clean, lint)
├── pyproject.toml                  # Project metadata & dependencies (uv / pip)
├── uv.lock                         # Locked dependency versions
├── README.md                       # Comprehensive documentation required by subject
├── structure.md                    # Project architecture & directory structure
│
├── llm_sdk/                        # Provided LLM SDK package (copied as-is)
│   ├── __init__.py
│   └── small_llm_model.py          # Small_LLM_Model class & vocab helpers
│
├── data/
│   ├── input/
│   │   ├── functions_definition.json   # Available functions and their schemas
│   │   └── function_calling_tests.json # Input natural language test prompts
│   └── output/                         # Output directory (GENERATED AT RUNTIME, DO NOT COMMIT)
│       └── function_calling_results.json
│
├── src/                            # Main application source code
│   ├── __init__.py
│   ├── __main__.py                 # Entrypoint for `python -m src`
│   │
│   ├── cli/                        # Command-line interface & argument parsing
│   │   ├── __init__.py
│   │   └── args.py                 # Parses --functions_definition, --input, --output
│   │
│   ├── models/                     # Pydantic data schemas & validation
│   │   ├── __init__.py
│   │   ├── function_schema.py      # FunctionDefinition, ParameterSchema, ReturnType
│   │   ├── test_prompt.py          # TestPrompt schema
│   │   └── result_schema.py        # FunctionCallResult schema (prompt, name, parameters)
│   │
│   ├── engine/                     # Core LLM & Constrained Decoding logic
│   │   ├── __init__.py
│   │   ├── vocabulary.py           # Vocabulary manager & token-to-string mappings
│   │   ├── logit_masker.py         # Logit masking and negative infinity assignment
│   │   ├── grammar_fsm.py          # Finite State Machine for valid JSON generation
│   │   └── decoder.py              # Main constrained decoding loop (token-by-token)
│   │
│   ├── pipeline/                   # End-to-end execution pipeline
│   │   ├── __init__.py
│   │   └── runner.py               # Orchestrates loading -> inference -> validation -> export
│   │
│   └── utils/                      # Helper utilities & robust error handling
│       ├── __init__.py
│       ├── io_helpers.py           # Safe JSON reader/writer with context managers
│       └── errors.py               # Custom exceptions & friendly error reporting
│
└── tests/                          # Unit & integration tests (optional for grading, great for dev)
    ├── __init__.py
    ├── test_pydantic_models.py     # Schema validation tests
    ├── test_logit_masker.py        # Masking and token filtering tests
    └── test_end_to_end.py          # Full pipeline runs with mock/real prompts
```

---

## 🔍 Detailed Component Responsibilities

### 1. Root Configuration Files

* **`Makefile`**: Contains mandatory targets:
  * `install`: Installs packages via `uv sync` or `pip install`.
  * `run`: Executes default pipeline via `uv run python -m src`.
  * `debug`: Runs the project using Python's built-in debugger `pdb`.
  * `clean`: Deletes `__pycache__`, `.mypy_cache`, `.pytest_cache`, and temporary files.
  * `lint`: Runs `flake8 .` and `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`.
  * `lint-strict`: (Optional) Runs `flake8 .` and `mypy . --strict`.

* **`pyproject.toml`**: Configures dependencies:
  * Runtime: `pydantic >= 2.0`, `numpy`
  * Dev/Linting: `flake8`, `mypy`, `pytest`
  * *(Forbidden: `transformers`, `pytorch`, `dspy`, `outlines`, etc.)*

* **`.gitignore`**:
  * Must strictly ignore `data/output/`, `__pycache__/`, `.mypy_cache/`, `.venv/`.

---

### 2. `src/models/` (Data Validation Layer)

All classes must use **Pydantic** for validation:
* **`function_schema.py`**:
  * Defines parameters (`type`: string, number, boolean, etc.), descriptions, function names.
  * Validates structure of `functions_definition.json`.
* **`result_schema.py`**:
  * Represents each result entry:
    ```python
    class FunctionCallResult(BaseModel):
        prompt: str
        name: str
        parameters: dict[str, Any]
    ```

---

### 3. `src/engine/` (Constrained Decoding Core)

This is the central algorithmic component:
* **`vocabulary.py`**:
  * Loads token mapping from `Small_LLM_Model.get_path_to_vocab_file()`.
  * Builds fast lookups (Token ID $\leftrightarrow$ Token String representation).
* **`grammar_fsm.py`**:
  * Tracks generation state:
    1. Function name selection state (only tokens matching valid function names).
    2. JSON punctuation states (`{`, `}`, `"`, `:`, `,`).
    3. Parameter key states (matching exact argument names for the selected function).
    4. Parameter value states (restricted according to argument types: number, string, boolean).
* **`logit_masker.py`**:
  * Computes binary or $-\infty$ mask over the vocabulary logits tensor.
  * Forces invalid tokens to `-float('inf')` before token selection.
* **`decoder.py`**:
  * Executes the token-by-token loop with `Small_LLM_Model.get_logits_from_input_ids()`.
  * Stops generation when the closing JSON token `}` is reached.

---

### 4. `src/pipeline/` & `src/cli/` (Execution & Orchestration)

* **`src/cli/args.py`**:
  * Uses `argparse` to handle:
    * `--functions_definition` (default: `data/input/functions_definition.json`)
    * `--input` (default: `data/input/function_calling_tests.json`)
    * `--output` (default: `data/output/function_calling_results.json`)
* **`src/pipeline/runner.py`**:
  * Coordinates safe loading of inputs $\rightarrow$ iterative model calling $\rightarrow$ results parsing $\rightarrow$ saving to target output path.

---

### 5. `src/utils/` (Error Handling & Robustness)

* **`io_helpers.py`**:
  * Uses Python context managers (`with open(...)`) for all file I/O.
  * Handles missing files, permissions issues, and malformed input JSON gracefully without crashing.
* **`errors.py`**:
  * Custom exception classes with user-friendly error messages.

---

## 🚀 Execution Commands

```bash
# 1. Install dependencies
uv sync

# 2. Run with default paths
uv run python -m src

# 3. Run with custom arguments
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json

# 4. Check code standards (Flake8 & Mypy)
make lint
```
