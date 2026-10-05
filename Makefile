.PHONY: install test quick experiments tables report market all

install:
	pip install -r requirements.txt

test:
	python -m pytest -v

quick:            # tiny-settings smoke test of the whole experiment suite
	python -m experiments.run_experiments --quick --out /tmp/out --figures /tmp/figs

experiments:      # full run, regenerates results/ and docs/figures/ (~15 min, one CPU core)
	python -m experiments.run_experiments

tables:           # rebuild tables and figures from the stored results.json
	python -m experiments.run_experiments --tables-only

market:           # real-market walk-forward study (needs internet and yfinance)
	python -m experiments.real_market_study --placebo

report:           # rebuild paper/technical_report.pdf from results/experiments.json (needs pdflatex)
	python paper/make_numbers.py
	cd paper && pdflatex -interaction=nonstopmode technical_report.tex && pdflatex -interaction=nonstopmode technical_report.tex

all: test experiments report
