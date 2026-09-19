# Fraud Sentinel

## 1. What this solution does

The pipeline implements:

CSV ingestion -> data forensics -> robust cleaning -> relational integration ->
prompt-injection defense -> behavioral/relational features -> deterministic
risk baseline + Isolation Forest -> optional Qwen2.5-0.5B SLM -> strict JSON
validation -> deterministic fallback -> `predictions.json`.

The supplied `transactions.csv` contains 1,000 rows and 29 columns; `accounts.csv`
contains 178 rows and 21 columns; `customers.csv` contains 124 rows and 26
columns. The transaction schema has no fraud/label column and no note/memo/
description/comment/message field, so the supplied files do not contain a
ground-truth fraud label or an obvious transaction-note prompt-injection field.

## 2. Model

Default SLM:

`meta-llama/Llama-3.2-1B-Instruct`

This is the approved default model for the current run. It is a compact
1B-parameter Llama model under the hackathon constraint and is compatible with
the existing local-path override and deterministic fallback logic.

Official model:
https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

If model download/inference is unavailable, the pipeline does not crash; it
falls back to the deterministic/statistical baseline.

For the default direct run, use:

```bash
python fraud_sentinel.py
```

## 2. Install dependencies

Use the project requirements before running the script:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Direct run

This is the supported path for the project:

```bash
python fraud_sentinel.py
```

The script reads the data files from `./data/` and writes predictions to both:

```text
./output/predictions.json
./predictions.json
```

The pipeline automatically uses the deterministic fallback when the SLM model or
inference path is not available, so a direct run is the intended default execution.

## 4. Data placement

The project expects these files in the repository:

```text
./data/transactions.csv
./data/accounts.csv
./data/customers.csv
```

## 5. Output schema

Every prediction is validated to:

```json
{
  "transaction_id": "TXN_00001",
  "is_fraud": true,
  "confidence": 0.92,
  "justification": "One concise sentence based on observed evidence."
}
```

`is_fraud` is a JSON boolean, `confidence` is constrained to `[0, 1]`, and
`transaction_id` must match the original input row.

## 6. Notes

- The provided dataset does not contain a true fraud-label column.
- The model path is optional and the script falls back safely when needed.
- For a normal run, use the direct command above and do not add extra setup steps.
- The goal is a simple, reproducible execution from the repo root.

## 13. 90-minute strategy

1. Run the deterministic pipeline first.
2. Confirm `predictions.json` is produced.
3. Enable SLM inference.
4. If GPU/time permits, run the compact LoRA pass.
5. Never allow model download or malformed generation to block final output.
6. Keep the final JSON validator/fallback in the execution path.

This prioritizes correctness, security, reproducibility and a guaranteed output
over an expensive model-training experiment.
