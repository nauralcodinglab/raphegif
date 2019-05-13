SIMDATA_PATH=data/simulations/GIF_network
GIFNET_PATH=data/models/GIF_network
GEN_PATH=analysis/GIF_network/model_generators
GIFMOD_PATH=data/models

.PHONY : all
all : $(SIMDATA_PATH)/subsample.ldat $(SIMDATA_PATH)/no_gaba_subsample.ldat $(SIMDATA_PATH)/no_IA_subsample.ldat

# Rules for constructing models.
# Deprecated for now because point_gifnets.py and fuzzy_gifnets.py depend on fitted GIF paths that are out of date.
#$(GIFNET_PATH)/median_gifs.mod $(GIFNET_PATH)/sergif_manual.mod $(GIFNET_PATH)/sergif_noIA.mod : $(GEN_PATH)/point_gifnets.py
#	PYTHONPATH="$(shell pwd)" python $<

#$(GIFNET_PATH)/fuzzy_mean_gifnet.mod $(GIFNET_PATH)/fuzzy_manual_gifnet.mod $(GIFNET_PATH)/fuzzy_manual_noIA_gifnet.mod : $(GEN_PATH)/fuzzy_gifnets.py
#	PYTHONPATH="$(shell pwd)" python $<

# Rule for running necessary simulations.
$(SIMDATA_PATH)/subsample.ldat : $(GIFNET_PATH)/subsample.mod  | $(SIMDATA_PATH)
	PYTHONPATH="$(shell pwd)" python analysis/GIF_network/gifnet_sim.py $< $@ -v

$(SIMDATA_PATH)/no_IA_subsample.ldat : $(GIFNET_PATH)/no_IA_subsample.mod  | $(SIMDATA_PATH)
	PYTHONPATH="$(shell pwd)" python analysis/GIF_network/gifnet_sim.py $< $@ -v

$(SIMDATA_PATH)/no_gaba_subsample.ldat : $(GIFNET_PATH)/no_gaba_subsample.mod | $(SIMDATA_PATH)
	PYTHONPATH="$(shell pwd)" python analysis/GIF_network/no_gaba_gifnet_sim.py $< $@ -v

# Rules to generate GIFnet models.
$(GIFNET_PATH)/subsample.mod : $(GEN_PATH)/subsample.py $(GIFMOD_PATH)/5HT/serkgifs.lmod $(GIFMOD_PATH)/GABA/gaba_gifs.mod | $(GIFNET_PATH)
	PYTHONPATH="$(shell pwd)" python $<

$(GIFNET_PATH)/no_IA_subsample.mod : $(GEN_PATH)/no_IA_subsample.py $(GIFMOD_PATH)/5HT/serkgifs.lmod | $(GIFNET_PATH)
	PYTHONPATH="$(shell pwd)" python $<

$(GIFNET_PATH)/no_gaba_subsample.mod : $(GEN_PATH)/no_gaba_subsample.py $(GIFMOD_PATH)/5HT/serkgifs.lmod | $(GIFNET_PATH)
	PYTHONPATH="$(shell pwd)" python $<

# Rules to create needed folders.
$(SIMDATA_PATH) $(GIFNET_PATH) :
	mkdir -p $@

.PHONY : clean
clean :
	rm $(GIFNET_PATH)/*; rm $(SIMDATA_PATH)/*;
