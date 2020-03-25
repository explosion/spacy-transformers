SHELL := /bin/bash
PYVER := 3.6
VENV := ./env$(PYVER)

version := $(shell "bin/get-version.sh")

dist/spacy-trf-$(version).pex : wheelhouse/spacy-trf-$(version).stamp
	$(VENV)/bin/pex -f ./wheelhouse --no-index --disable-cache -o $@ spacy_transformers==$(version) jsonschema
	chmod a+rx $@

wheelhouse/spacy-trf-$(version).stamp : $(VENV)/bin/pex setup.py spacy_transformers/*.py* spacy_transformers/*/*.py*
	$(VENV)/bin/pip wheel . -w ./wheelhouse
	$(VENV)/bin/pip wheel jsonschema -w ./wheelhouse
	touch $@

$(VENV)/bin/pex :
	python$(PYVER) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip setuptools pex wheel

.PHONY : clean

clean : setup.py
	rm -rf dist/*
	rm -rf ./wheelhouse
	rm -rf $(VENV)
	python setup.py clean --all
