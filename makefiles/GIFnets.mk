SIM_PATH=$(data/simulations/GIF_network/)

.PHONY : all
all : data/simulations/GIF_network/median_gifs.dat data/simulations/GIF_network/sergif_manual.dat data/simulations/GIF_network/sergif_noIA.dat

data/models/GIF_network/%.mod : analysis/GIF_network/generate_gif_models.py
	python -m analysis.GIF_network.generate_gif_models

data/simulations/GIF_network/%.dat : data/models/GIF_network/%.mod
	@echo Floopabloop > $@
