<p align="center">
  <picture>
    <img alt="Lettuce Logo" src="/website/public/lettuce-logo.svg" width="280"/>
  </picture>
  </a>
</p>
<div align="center">
  <strong>
  LLM for Efficient Translation and Transformation into Uniform Clinical Encoding
  <br />
  </strong>
</div>

Lettuce is an application for medical researchers that matches the source terms supplied by the user to concepts in the [Observational Health Data Sciences and Informatics](https://www.ohdsi.org) (OMOP) [standardised vocabularies](https://github.com/OHDSI/Vocabulary-v5.0/wiki)

The application can be used as an [HTTP API](https://health-informatics-uon.github.io/lettuce/deployment), a [CLI](https://health-informatics-uon.github.io/lettuce/api_reference/cli), or run with a [graphical user interface (GUI)](https://health-informatics-uon.github.io/lettuce/api_reference/ui).

## Overview

Lettuce uses vector search, a large language model, and text search features to help find the matching concept for a source term.
A full pipeline uses the vector search results for retrieval-augmented generation, then the answer provided by an LLM to run text search against a configured OMOP-CDM database.
Users can use the full pipeline, or only components of it, depending on requirements.

<div align="center">
    <picture>
        <img alt="Lettuce workflow" src="/website/public/Lettuce_Architecture.png" />
    </picture>
</div>

## Installation

To use Lettuce, follow [the quickstart](https://health-informatics-uon.github.io/lettuce/quickstart)

### Connecting to a database

Lettuce works by querying a database with the OMOP schema, so you should have access to one. Your database access credentials should be kept in `.env`. An example of the format can be found in `/Lettuce/.env.example`

## Published Images
Development Docker images for the Lettuce project are available on GitHub Container Registry (GHCR):

- **Registry**: `ghcr.io/health-informatics-uon/lettuce`
- **Weights Image** (pre-loaded LLaMA-3.1-8B weights):
  - `dev-weights-llama-3.1-8B-sha-<hash>` (e.g., `-sha-a1b2c3d`)
  - `dev-weights-llama-3.1-8B-edge` (latest)
  - Pull: `docker pull ghcr.io/health-informatics-uon/lettuce:dev-weights-llama-3.1-8B-edge`
- **Base Image** (lightweight, no weights):
  - `dev-base-sha-<hash>` (e.g., `-sha-a1b2c3d`)
  - `dev-base-edge` (latest)
  - Pull: `docker pull ghcr.io/health-informatics-uon/lettuce:dev-base-edge`

See [GitHub Packages](https://github.com/Health-Informatics-UoN/lettuce/pkgs/container/lettuce) for all tags.

## Contact

If there are any bugs, please raise an issue or [email us.](mailto:james.mitchell-white1@nottingham.ac.uk)
