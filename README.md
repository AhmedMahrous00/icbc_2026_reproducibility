# Paper Metrics Bundle

Minimal bundle to run `reproduce_paper_metrics.py`.

## Quick start
```bash
cd /home/mahrouaa/pump_and_dump/paper_metrics_bundle
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (no APIs)
```bash
python reproduce_paper_metrics.py --reports_dir reports_local
```

## Run with APIs
```bash
# set OPENAI_API_KEY / DEEPSEEK_API_KEY / GEMINI_API_KEY first
python reproduce_paper_metrics.py --run_api --preflight_checks --reports_dir reports_api
```

Outputs are written under `reports_*/`.
