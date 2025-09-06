.PHONY: black check

PROJECT_DIR := .
DATASET_SRIPT := allgorythms/scripts/generate_dataset.py
TRAIN_SCRIPT := allgorythms/model/train.py
PREDICT_SCRIPT := allgorythms/model/predict.py
FIND_ALGOS := allgorythms/scripts/find_algos.py

black:
	black $(PROJECT_DIR)

check:
	black --color --check --diff $(PROJECT_DIR)

dataset:
	python $(DATASET_SRIPT)

train:
	python $(TRAIN_SCRIPT)

predict:
ifndef FILE
	$(error "Usage: make predict FILE=path/to/code_file")
endif
	python $(PREDICT_SCRIPT) $(FILE)

find:
ifndef URL
	$(error "Usage: make predict URL=url/to/code_file")
endif
	python $(FIND_ALGOS) $(URL)
