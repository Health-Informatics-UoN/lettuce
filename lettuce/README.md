# Lettuce
This directory contains the main Lettuce program, an AI assistant for making OMOP mappings.

For detailed instructions follow the [Lettuce docs](https://health-informatics-uon/github.io/lettuce).

## Dependencies
Lettuce now uses `uv` for dependency management. To install `uv` (MacOS / Linux) run: 
```bash 
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then run the following command: 
```bash
uv sync  --all-extras
```
This command will create a `.venv` folder (if it doesn't already exist) at the root of the project and install the main and developer dependencies. Omit the `--all-extras` flag to install the main package only. It will also re-lock the project by generating a `uv.lock` file. See the [`uv` documentation](https://docs.astral.sh/uv/reference/cli/#uv-sync) for further details.
## Running the CLI
Lettuce has a command-line interface, run it with `uv run lettuce-cli`

## Starting the API
Running `uv run python app.py` will start up the Lettuce API on port 8000
