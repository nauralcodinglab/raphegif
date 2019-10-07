# Preprocess 5HT noisy current data.
# Must be run from project root with appropriate conda environment active!

# Define paths.
PROJECT_ROOT = $(shell pwd)
DATA_PATH = data/processed/5HT_fastnoise
BENCHMARK_PATH = data/processed/benchmarks
MODEL_PATH = data/models/5HT
SCRIPT_PATH = analysis/regression_tinkering
SRC_PATH = grr

# Final targets.
.PHONY : all
all : $(SCRIPT_PATH)/inspect_model_coefficients.ipynb $(SCRIPT_PATH)/inspect_benchmark_results.ipynb

# Run notebooks to inspect results.
$(SCRIPT_PATH)/inspect_model_coefficients.ipynb : $(MODEL_PATH)/*.lmod
	jupyter nbconvert --to notebook --execute $@ --inplace

$(SCRIPT_PATH)/inspect_benchmark_results.ipynb : $(BENCHMARK_PATH)/*_benchmark.dat
	jupyter nbconvert --to notebook --execute $@ --inplace

# Benchmark models.
$(BENCHMARK_PATH)/%_benchmark.dat : $(MODEL_PATH)/%.lmod $(DATA_PATH)/5HT_goodcells.ldat \
$(SCRIPT_PATH)/compute_Md.py | $(BENCHMARK_PATH)
	PYTHONPATH="$(PROJECT_ROOT)" python $(SCRIPT_PATH)/compute_Md.py -v $(DATA_PATH)/5HT_goodcells.ldat $< $@ --precision 8.0

$(SCRIPT_PATH)/compute_Md.py : $(SRC_PATH)/Tools.py

$(BENCHMARK_PATH) :
	mkdir -p $(BENCHMARK_PATH)

# Fit models.
$(MODEL_PATH)/sergifs.lmod : $(SCRIPT_PATH)/fit_sergifs.py $(DATA_PATH)/5HT_goodcells.ldat | $(MODEL_PATH)
	PYTHONPATH="$(PROJECT_ROOT)" python $<

$(SCRIPT_PATH)/fit_sergifs.py : $(SRC_PATH)/GIF.py $(SRC_PATH)/Filter_Exps.py

$(MODEL_PATH)/serkgifs.lmod : $(SCRIPT_PATH)/fit_serkgifs.py $(DATA_PATH)/5HT_goodcells.ldat | $(MODEL_PATH)
	PYTHONPATH="$(PROJECT_ROOT)" python $<

$(SCRIPT_PATH)/fit_serkgifs.py : $(SRC_PATH)/AugmentedGIF.py $(SRC_PATH)/Filter_Exps.py $(SCRIPT_PATH)/model_evaluation.py

$(MODEL_PATH) :
	mkdir -p $(MODEL_PATH)

# Preprocess data.
$(DATA_PATH)/5HT_goodcells.ldat : data/raw/5HT/fast_noise/index.csv $(SCRIPT_PATH)/preprocess_fast_noise.py | $(DATA_PATH)
	PYTHONPATH="$(PROJECT_ROOT)" python $(SCRIPT_PATH)/preprocess_fast_noise.py

$(DATA_PATH) :
	mkdir -p $(DATA_PATH)

# Other commands.
.PHONY : clean
clean :
	rm $(DATA_PATH)/5HT_goodcells.ldat $(MODEL_PATH)/serkgifs.lmod $(MODEL_PATH)/sergifs.lmod $(BENCHMARK_PATH)/serkgifs_benchmark.dat $(BENCHMARK_PATH)/sergifs_benchmark.dat
