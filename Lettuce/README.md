# LLettuce

LLettuce is a FastAPI application designed to assist in identifying medications by converting informal names to formal names and searching the OMOP (Observational Medical Outcomes Partnership) database.

## Table of Contents

- [Lettuce](#lettuce)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Setup](#setup)
  - [Usage](#usage)
  - [API](#api)
    - [Run Pipeline](#run-pipeline)
  - [Functionality](#functionality)
    - [Informal Name to Formal Name Conversion](#informal-name-to-formal-name-conversion)
    - [OMOP Search](#omop-search)

## Installation

To install Lettuce, you will need to clone the repository and install the dependencies listed in the `requirements.txt` file.

```bash
git clone https://UniversityOfNottingham@dev.azure.com/UniversityOfNottingham/DRS/_git/brc-llm
cd Carrot-Assistant
pip install -r requirements.txt
```

Alternatively, you can use Docker to containerize the application.

```bash
docker compose up
```

## Setup

Before running the application, make sure to set up the necessary environment variables and initialize the database.

1. **Environment Variables**: Copy the `.env.example` to `.env` and configure the required variables.
   
2. **Database Details**: You can proved a local or remote database details in the .env file and the application will connect to the database.

## Usage

To run the application locally, use the following command:

```bash
python app.py
```

To run the application using Docker:

```bash
docker run -p 8000:8000 Lettuce
```

To run the UI:

```bash
streamlit run ui.py
```
The UI will be available at `http://localhost:8501`.

## API

### Run Pipeline

**Endpoint**: `/run`  
**Method**: `POST`

**Request Body**:
```json
{
  "informal_name": "Betnovate Scalp Application"
}
```

**Response**:
```json
{
  "event": "llm_output",
  "data": {
    "reply": formal_name: str,
    "meta": LLM metadata: List,
  }
}
```
```json
{
  "event": "omop_output",
  "data": [
    {
      "search_term": search_term: str,
      "CONCEPT": [concept_data: Dict]
    }
  ]
}
```

## Functionality

### Informal Name to Formal Name Conversion

The informal name provided in the request is processed by the `assistant.run` function. This function uses a language model to convert the informal name to a formal name.

### OMOP Search

The formal name obtained from the previous step is then used to search the OMOP database using the `OMOP_match.run` function. This function retrieves relevant information from the OMOP database based on the formal name.

---

For more detailed information, please refer to the source code and comments within the respective files. If you encounter any issues, feel free to open an issue on the repository or contact the maintainer.

