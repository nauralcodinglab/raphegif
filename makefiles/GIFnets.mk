SIM_PATH=data/simulations/GIF_network
MOD_PATH=data/models/GIF_network
GEN_PATH=analysis/GIF_network/model_generators
INGEN_PATH=./analysis/GIF_network/input_generators

.PHONY : all
all : $(SIM_PATH)/subsample.ldat $(SIM_PATH)/no_gaba_subsample.ldat

# Rules for constructing models.
# Incomplete because dependency on fitted GIFs isn't accounted for...
#$(MOD_PATH)/median_gifs.mod $(MOD_PATH)/sergif_manual.mod $(MOD_PATH)/sergif_noIA.mod : $(GEN_PATH)/point_gifnets.py
#	PYTHONPATH="$(shell pwd)" python $<

#$(MOD_PATH)/fuzzy_mean_gifnet.mod $(MOD_PATH)/fuzzy_manual_gifnet.mod $(MOD_PATH)/fuzzy_manual_noIA_gifnet.mod : $(GEN_PATH)/fuzzy_gifnets.py
#	PYTHONPATH="$(shell pwd)" python $<

# Rule for running necessary simulations.
$(SIM_PATH)/subsample.ldat : $(MOD_PATH)/subsample.mod  | $(SIM_PATH)
	PYTHONPATH="$(shell pwd)" python analysis/GIF_network/gifnet_sim.py $< $@ -v

$(SIM_PATH)/no_gaba_subsample.ldat : $(MOD_PATH)/no_gaba_subsample.mod | $(SIM_PATH)
	PYTHONPATH="$(shell pwd)" python analysis/GIF_network/no_gaba_gifnet_sim.py $< $@ -v

$(MOD_PATH)/no_gaba_subsample.mod : $(GEN_PATH)/no_gaba_subsample.py
	PYTHONPATH="$(shell pwd)" python $<


#$(SIM_PATH)/subsample_input.ldat : $(INGEN_PATH)/OU_noise.py | $(SIM_PATH)
#	PYTHONPATH="$(shell pwd)" python $<

$(SIM_PATH) :
	mkdir -p $(SIM_PATH)

.PHONY : clean
clean :
	rm $(MOD_PATH)/*; rm $(SIM_PATH)/*;
