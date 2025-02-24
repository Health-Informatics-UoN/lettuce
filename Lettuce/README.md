# Lettuce
This directory contains the main Lettuce program, an AI assistant for making OMOP mappings.

For detailed instructions follow the [Lettuce docs](https://health-informatics-uon/github.io/lettuce).

## Dependencies
Lettuce now uses `uv` for dependency management. To install `uv` (MacOS / Linux) run: 
```bash 
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then set up the virtual environment in the root of the package (the same folder as `pyproject.toml`) and generate the locked requirements file and install the dependencies.  
```bash
uv venv --python 3.12 
uv pip compile -r pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
```
## Running the CLI
Lettuce has a command-line interface, run it with `uv run lettuce-cli`

## Starting the API
Running `uv run python app.py` will start up the Lettuce API on port 8000
