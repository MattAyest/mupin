# langgraph-self-healing-code-agent

An autonomous, self-healing code generation microservice built with LangGraph and FastAPI. 

This agent architects solutions, generates implementations alongside test suites, and iteratively verifies its output within an isolated Docker environment. When tests fail, the system analyzes the trace and modifies its own logic until the criteria are met.

## Key Features

* **State Graph Architecture:** Utilizes specialized LangGraph nodes (architect, synthesizer, analyzer, distiller) to manage complex coding tasks.
* **Self-Healing Execution Loop:** Code is executed in a sandboxed Docker container using `pytest` and `hypothesis`. Errors are intercepted, distilled, and fed back into the synthesis node.
* **Hybrid LLM Routing:** Combines cloud models (e.g., OpenAI GPT-4o) for heavy synthesis with local models (via Ollama) for rapid error diagnostics and environment planning.
* **Asynchronous API:** Tasks are triggered asynchronously, allowing clients to poll for status, graph node progression, and final outputs via a REST interface.

## Architecture Flow

1. **Speculative Router:** Evaluates prompt complexity to determine if a dedicated architectural plan is required.
2. **Architect & Environment Node:** Generates a project blueprint and dynamically resolves dependencies into a `requirements.txt`.
3. **Synthesizer:** Writes the Python implementation and corresponding test suite, formatted via XML block extraction.
4. **Static Analyzer:** Validates AST parsing to catch syntax errors prior to runtime execution.
5. **Deterministic Verifier:** Mounts generated files into a lightweight Python Docker container and executes the test matrix. 
6. **Error Distiller:** If tests fail, the traceback is condensed into actionable instructions for the synthesizer. Repeated regressions trigger a rollback and force the architect to draft a new plan.

## Getting Started

### Prerequisites

* Python 3.11+
* Docker (Required for the `deterministic_verifier` sandbox)
* Ollama (Running locally with `qwen2.5-coder:3b` or equivalent)
* Cloud LLM API Keys (e.g., `OPENAI_API_KEY`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/langgraph-self-healing-code-agent.git
   cd langgraph-self-healing-code-agent
   ```

2. Configure the virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r Coding-Module/requirements.txt
   ```

3. Ensure local models are available:
   ```bash
   ollama pull qwen2.5-coder:3b
   ```

### Running the Service

Start the FastAPI backend:
```bash
uvicorn Coding-Module.src.api:app --reload
```

### Usage

Submit a generation request:
```bash
curl -X POST "http://127.0.0.1:8000/task" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Write a Python script that calculates projectile motion and include a pytest suite."}'
```

Poll task status:
```bash
curl -X GET "http://127.0.0.1:8000/task/<task_id>"
```
The response will track the `current_node`, `loop_count`, and `regression_count`. Upon completion, the final, verified file manifest will be provided in the response body.
