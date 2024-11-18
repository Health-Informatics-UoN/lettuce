Usage
=====

Installation
------------

To use Llettuce, you must first clone the repository

.. code-block:: console

   $ git clone <url>
   $ cd Carrot-Assistant

Then install the dependencies.

We recommend using `poetry <https://python-poetry.org/>`_ then running the commands using the poetry shell

Alternatively, dependencies can be installed either using pip

.. code-block:: console

   $ pip install -r requirements.txt

or conda

.. code-block:: console

  $ conda create -f environment.yml

Connecting to a database
-------------------------

Llettuce works by querying a database with the OMOP schema, so you should have access to one. Your database access credentials should be kept in `.env`. An example of the format can be found in `/Carrot-Assistant/.env.example`:

.. literalinclude:: ../Carrot-Assistant/.env.example

Running the API
----------------

The simplest way to get a formal name from an informal name is to use the API and the GUI. To start a Llettuce server:

.. code-block:: console

  $ python app.py


Or run the application using Docker

.. code-block:: console

   $ docker run -p 8000:8000 Lettuce

Then start another terminal, and start the GUI

.. code-block:: console

   $ streamlit run ui.py

The GUI makes calls to the API equivalent to the curl request below.

Run pipeline
************

To get a response without the GUI, a request can be made using curl, e.g. for Betnovate scalp application

.. code-block:: console
   
   $ curl -X POST "http://127.0.0.1:8000/run" -H "Content-Type: application/json" -d '{"names": ["Betnovate Scalp Application", "Panadol"]}'

The API endpoint is `/run`, and uses a `POST` method

The request body should have the format

.. code-block::

   {
    "names": <Drug informal names>,
    "pipeline_options": {
      <options>
    }
   }


Refer to `app.py` in the API reference for the available pipeline options.

The response will be provided in the format

.. code-block::

   {
    "event": "llm_output",
    "data": {
       "reply": formal_name: str,
       "meta": LLM metadata: List,
     }
   }

   {
    "event": "omop_output",
    "data": [
       {
         "search_term": search_term: str,
         "CONCEPT": [concept_data: Dict]
       }
     ]
   }

The response will be streamed asynchronously so the llm_output will arrive before any omop_output
