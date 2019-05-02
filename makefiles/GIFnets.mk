SIM_PATH=data/simulations/GIF_network
MOD_PATH=data/models/GIF_network
GEN_PATH=analysis/GIF_network/model_generators

.PHONY : all
all : \
$(SIM_PATH)/median_gifs.ldat $(SIM_PATH)/sergif_manual.ldat $(SIM_PATH)/sergif_noIA.ldat \
$(SIM_PATH)/fuzzy_mean_gifnet.ldat $(SIM_PATH)/fuzzy_manual_gifnet.ldat $(SIM_PATH)/fuzzy_manual_noIA_gifnet.mod

# Rules for constructing models.
# Incomplete because dependency on fitted GIFs isn't accounted for...
$(MOD_PATH)/median_gifs.mod $(MOD_PATH)/sergif_manual.mod $(MOD_PATH)/sergif_noIA.mod : $(GEN_PATH)/point_gifnets.py
	PYTHONPATH="$(shell pwd)" python $<

$(MOD_PATH)/fuzzy_mean_gifnet.mod $(MOD_PATH)/fuzzy_manual_gifnet.mod $(MOD_PATH)/fuzzy_manual_noIA_gifnet.mod : $(GEN_PATH)/fuzzy_gifnets.py
	PYTHONPATH="$(shell pwd)" python $<

# Rule for running necessary simulations.
$(SIM_PATH)/%.ldat : $(MOD_PATH)/%.mod
	PYTHONPATH="$(shell pwd)" python analysis/GIF_network/gifnet_sim.py $< $@ -v

.PHONY : clean
clean :
	rm $(MOD_PATH)/*; rm $(SIM_PATH)/*;
