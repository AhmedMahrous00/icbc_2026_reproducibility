# PumpSense Reproducibility

Code and data accompanying **PumpSense: Real-Time Detection and Target Extraction of Crypto Pump-and-Dumps on Telegram**, by Ahmed Mahrous and Roberto Di Pietro (IEEE ICBC 2026, pp. 1–9).

[Published paper](https://doi.org/10.1109/ICBC67748.2026.11575494) · [Paper page](https://amahrous.com/papers/pumpsense/) · [Preprint](https://arxiv.org/abs/2605.09431) · [Archived code & data (v1.1)](https://doi.org/10.5281/zenodo.19616802)

The repository includes Telegram channel data in `chat_data/`, detection and extraction scripts, and `reproduce_paper_metrics.py` for producing evaluation reports.

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

```bibtex
@inproceedings{mahrous2026pumpsense,
  title={PumpSense: Real-Time Detection and Target Extraction of Crypto Pump-and-Dumps on Telegram},
  author={Mahrous, Ahmed and Di Pietro, Roberto},
  booktitle={2026 IEEE International Conference on Blockchain and Cryptocurrency (ICBC)},
  year={2026},
  pages={1--9},
  doi={10.1109/ICBC67748.2026.11575494},
  url={https://doi.org/10.1109/ICBC67748.2026.11575494}
}
```
