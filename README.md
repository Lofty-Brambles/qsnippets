# QSnippets

A collection of Jupyter notebooks containing snippets showing some basic quantum computing algorithms.

# Setup and Usage

## Create Virtual Environment and Install Dependencies

```bash
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Environment Variables

Copy the `.example.env` file to `.env` and fill in your API keys:

```bash
cp .example.env .env
```

Then edit `.env` with your credentials. Here, we use only blue qubit, so filling in the IBM key is optional:

-   `BLUE_QUBIT_KEY`: Your Blue Qubit API key
-   `IBM_RUNTIME_KEY`: Your IBM Quantum Runtime API key

## Run Notebooks

```bash
jupyter notebook
```

This will open a browser window where you can navigate to and open the notebook files.

## Deactivate Environment

```bash
deactivate
```
