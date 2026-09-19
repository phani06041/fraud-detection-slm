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

You can override it at runtime with either a remote model id or a local model directory:

```bash
python fraud_sentinel.py --model-name meta-llama/Llama-3.2-1B-Instruct
```

or:

```bash
python fraud_sentinel.py --model-name /path/to/Llama-3.2-1B-Instruct
```

If model download/inference is unavailable, the pipeline does not crash; it
falls back to the deterministic/statistical baseline.

## 3. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Dataset placement

For the hackathon runner:

```text
/app/environment/data/transactions.csv
/app/environment/data/accounts.csv
/app/environment/data/customers.csv
```

The script also searches `./data/` and the current directory.

## 5. Run immediately

```bash
python fraud_sentinel.py
```

Output:

```text
/app/output/predictions.json
```

By default, the pipeline runs in deterministic mode so it finishes reliably within a 90-minute hackathon. Use the approved Llama model only when a valid local cache or fast GPU environment is available:

```bash
python fraud_sentinel.py --use-slm --model-name meta-llama/Llama-3.2-1B-Instruct
```

To keep the submission fast and robust when model loading is not practical:

```bash
python fraud_sentinel.py --no-use-slm
```

## 6. Run LoRA fine-tuning

Because the supplied data has no genuine fraud labels, fine-tuning uses only
high-confidence pseudo-labels generated from the deterministic/anomaly
baseline. These are explicitly NOT ground truth.

```bash
python fraud_sentinel.py --finetune
```

The adapter is saved under:

```text
./fraud_sentinel_lora/
```

For a 90-minute hackathon, use one epoch / capped steps as implemented rather
than attempting full-model fine-tuning.

## 7. Output schema

Every prediction is validated to:

```json
{
  "transaction_id": "TXN_00001",
  "is_fraud": true,
  "confidence": 0.92,
  "justification": "One concise sentence based on observed evidence."
}
```

`is_fraud` is a JSON boolean, `confidence` is constrained to [0, 1], and
`transaction_id` must match the input.

## 8. Data-quality handling

The pipeline preserves suspicious conditions as features instead of silently
discarding rows:

- malformed/missing amounts
- invalid timestamps
- negative/zero amounts
- duplicate transaction IDs
- missing/unmatched account relationships
- missing/unmatched customer relationships
- missing device/authentication information
- extreme amounts
- high velocity
- unusual account-average ratios
- new device
- foreign transaction
- negative post-transaction balance

Amounts such as `INR 62,146.26` and `41,677.41` are normalized into numeric
values while parse failures remain visible through `amount_parse_error`.

## 9. Prompt-injection defense

Transaction notes are treated as untrusted data if a note-like column exists.
The detector looks for layered instruction-like patterns such as:

- ignore previous instructions
- system prompt
- developer message
- override instructions
- jailbreak
- you are now
- act as
- classify this as safe
- mark this legitimate

The raw suspicious text is never placed in the system/authoritative instruction.
The model receives only compact structured evidence inside explicit
`<UNTRUSTED_TRANSACTION_EVIDENCE>` delimiters.

The supplied transaction file has no note-like column, so no note injection is
present in the provided schema.

## 10. Fraud logic

No genuine fraud label exists in the supplied transaction schema. Therefore
the solution does not claim supervised fraud accuracy.

The baseline combines transparent risk rules with Isolation Forest. Examples:

- extreme amount
- negative amount
- credit-limit exceedance
- high utilization
- negative post-transaction balance
- high 24h/7d velocity
- rapid transactions
- large home-distance
- amount/account-average ratio
- new device
- foreign transaction
- missing authentication/device
- relational/data-quality anomalies
- prompt-injection detection when applicable

The baseline is an anomaly/risk score, not a calibrated real-world fraud
probability.

## 11. Hybrid decision

When the SLM is available:

`final_risk = 0.75 * baseline_risk + 0.25 * SLM_risk`

The weights are deliberately documented as a hackathon heuristic because there
is no genuine labeled validation set in the supplied data. They are not claimed
to be optimal.

If SLM output is malformed, inconsistent with the schema, times out, or fails,
the deterministic prediction is used.

## 12. Reproducibility

Seed:

`42`

LoRA configuration:

- rank: 8
- alpha: 16
- dropout: 0.05
- target modules: q/k/v/o projection layers
- one epoch
- capped at 150 steps
- batch size: 2
- gradient accumulation: 4
- learning rate: 2e-4

## 13. 90-minute strategy

1. Run the deterministic pipeline first.
2. Confirm `predictions.json` is produced.
3. Enable SLM inference.
4. If GPU/time permits, run the compact LoRA pass.
5. Never allow model download or malformed generation to block final output.
6. Keep the final JSON validator/fallback in the execution path.

This prioritizes correctness, security, reproducibility and a guaranteed output
over an expensive model-training experiment.
