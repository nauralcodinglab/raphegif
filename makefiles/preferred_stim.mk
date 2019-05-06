PROJECT_ROOT=$(shell pwd)
DATA_PATH=./data/processed/5HT_fastnoise
SCRIPT_PATH=./analysis/preferred_stimuli

.PHONY : all
all : $(DATA_PATH)/5HT_goodcells_spktrig.ldat

$(DATA_PATH)/%_spktrig.ldat : $(DATA_PATH)/%.ldat $(SCRIPT_PATH)/extract_spktrig.py | $(DATA_PATH)
	PYTHONPATH="$(PROJECT_ROOT)" python $(SCRIPT_PATH)/extract_spktrig.py -v $< $@

$(DATA_PATH) :
	mkdir -p $(DATA_PATH)

.PHONY : clean
clean :
	rm $(DATA_PATH)/5HT_goodcells_spktrig.ldat
