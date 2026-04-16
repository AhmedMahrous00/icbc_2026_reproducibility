## Reproducibility
This repository reproduces main results from the paper "PumpSense: Detection and Target Extraction of Cryptocurrency Pump-and-Dump on Telegram" accepted at ICBC 2026.

## Quick start
```bash
cd reproducibility_folder
python3 -m venv .venv
source .venv/bin/activate
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

## Citation

If you use this repository, please cite our paper:

@inproceedings{mahrous2026pumpsense,
  title={PumpSense: Detection and Target Extraction of Cryptocurrency Pump-and-Dump on Telegram},
  author={Mahrous, Ahmed and others},
  booktitle={Proceedings of the IEEE International Conference on Blockchain and Cryptocurrency (ICBC)},
  year={2026}
}
