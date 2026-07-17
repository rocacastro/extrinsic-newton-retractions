PYTHON ?= python
LATEXMK ?= latexmk

.PHONY: install results compare verify supplement checksums clean

install:
	$(PYTHON) -m pip install -r requirements.txt

results:
	$(PYTHON) src/generate_article_tables.py --output-root results_local

compare:
	$(PYTHON) scripts/compare_residuals.py --candidate-root results_local

verify:
	$(PYTHON) scripts/verify_repository.py

supplement:
	cd supplement && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error supplementary_material.tex

checksums:
	$(PYTHON) scripts/generate_checksums.py

clean:
	rm -rf results_local __pycache__ src/__pycache__ scripts/__pycache__ .tmp
	cd supplement && $(LATEXMK) -C supplementary_material.tex || true
