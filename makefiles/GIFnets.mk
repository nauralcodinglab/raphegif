SIM_PATH=$(data/simulations/GIF_network/)

.PHONY : all
all : data/simulations/GIF_network/median_gifs.dat data/simulations/GIF_network/sergif_manual.dat data/simulations/GIF_network/sergif_noIA.dat

data/models/GIF_network/median_gifs.mod : analysis/GIF_network/generate_gif_models.py
	python -m analysis.GIF_network.generate_gif_models

data/models/GIF_network/sergif_manual.mod : analysis/GIF_network/generate_gif_models.py
	python -m analysis.GIF_network.generate_gif_models

data/models/GIF_network/sergif_noIA.mod : analysis/GIF_network/generate_gif_models.py
	python -m analysis.GIF_network.generate_gif_models

data/simulations/GIF_network/%.dat : data/models/GIF_network/%.mod
	python -m analysis.GIF_network.gifnet_sim $< $@ -v

.PHONY : clean
clean :
	rm data/models/GIF_network/*; rm data/simulations/GIF_network/*;
