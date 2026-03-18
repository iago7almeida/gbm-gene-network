# GitHub Copilot / AI Agent Instructions for gbm-gene-network

Purpose: make an AI coding agent immediately productive working on this repo.

- **Big picture:** This repo builds a GBM gene co-expression pipeline:
  - `src/extract_gdc.py` and `src/extract_clinical.py` pull raw data from the GDC API.
  - `src/process_matrix.py` cleans counts, selects top HVGs and produces an edge list (CSV).
  - `src/load_graph.py` ingests nodes/edges into Neo4j (Aura); creates unique constraint and uploads in batches.
  - `src/analyze_network.py` runs Cypher analyses and writes CSV reports to `data/reports/`.
  - `src/deploy_dashboard.py` loads `neo4j/neodash_config.json` into Neo4j as a dashboard object.

- **Data flow & file conventions:**
  - Raw downloaded matrices go to `data/raw/` (example output: `data/raw/matriz_gbm_bruta.csv`).
  - Processed outputs are written to `data/processed/` (`gbm_genes_nodes.csv`, `gbm_network_edges.csv`).
  - Reports are saved under `data/reports/`.
  - CSVs use headers with `gene_id`, `gene_source`, `gene_target`, `weight` as shown in `src/process_matrix.py`.

- **Runtime / environment:**
  - Use a Python 3.11+ virtualenv and `pip install -r requirements.txt` (see `requirements.txt`).
  - Environment variables: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` are required by Neo4j scripts; `.env.example` is present but empty — prefer creating a local `.env` before running.
  - Dockerfile and `buildspec.yml` are present but empty; do not assume containerized CI is configured.

- **Typical developer tasks & exact commands:**
  - Create venv and install:
    - `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
  - Download and build raw matrix (small test):
    - `python src/extract_gdc.py` (script limits via `max_files` parameter in code)
  - Process matrix to edges/nodes:
    - `python src/process_matrix.py` (reads `data/raw/matriz_gbm_bruta.csv` and writes `data/processed/`)
  - Ingest into Neo4j (ensure env vars set):
    - `python src/load_graph.py` (creates uniqueness constraint and loads in batches; batch size tuned in source)
  - Run analysis / export reports:
    - `python src/analyze_network.py`
  - Deploy dashboard JSON to Neo4j:
    - `python src/deploy_dashboard.py` (uses `neo4j/neodash_config.json`)

- **Project-specific patterns & gotchas for agents:**
  - Data size concerns: `process_matrix.py` reduces from ~60k genes to top-N HVGs (default 2000). Avoid changing this without considering memory.
  - Neo4j ingestion: `load_graph.py` uses `MERGE` and creates a uniqueness constraint `g.id` — do not insert duplicate node ids; rely on the constraint.
  - Batch upload pattern: edges are sent as Python lists of dicts in batches (see `load_edges`). Keep batch sizes modest during dry runs.
  - File index conventions: many scripts expect the first column of CSVs to be the index (`index_col=0`) — preserve that when producing test CSVs.
  - Minimal error handling is present; prefer small, reproducible runs (reduce `max_files`, set lower correlation thresholds) when developing.

- **Where to look for examples:**
  - `src/extract_gdc.py` — API call patterns and tar extraction.
  - `src/process_matrix.py` — HVG selection, Pearson correlation -> edge list and deduplication.
  - `src/load_graph.py` — Neo4j connection pattern, `CREATE CONSTRAINT`, batch uploads.

- **When making edits or adding features:**
  - Keep scripts idempotent: preserve `MERGE` semantics in DB writes and maintain the uniqueness constraint.
  - Add feature toggles via function parameters (e.g., `max_files`, `top_n_genes`, `correlation_threshold`) rather than hardcoded constants.
  - When touching data formats, include a short example CSV snippet in the PR description showing header + 2 rows.

If anything here is unclear or you'd like more details (CI commands, container setup, or an example `.env.example`), tell me which area to expand and I will iterate.
