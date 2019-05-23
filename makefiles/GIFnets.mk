SIMDATA_PATH=data/simulations/GIF_network
GIFNET_PATH=data/models/GIF_network
GEN_PATH=analysis/GIF_network/model_generators
GIFMOD_PATH=data/models

.PHONY : all
all : $(SIMDATA_PATH)/subsample.ldat $(SIMDATA_PATH)/no_gaba_subsample.ldat $(SIMDATA_PATH)/no_IA_subsample.ldat

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
$(SIMDATA_PATH) $(GIFNET_PATH) $(SIMDATA_PATH)/input :
	mkdir -p $@

.PHONY : test
test : $(SIMDATA_PATH)/subsample_test.hdf5

$(SIMDATA_PATH)/subsample_test.hdf5 : $(GIFNET_PATH)/subsample.mod $(SIMDATA_PATH)/input/OU_test.ldat analysis/GIF_network/gifnet_sim.py | $(SIMDATA_PATH)
	PYTHONPATH="$(shell pwd)" python analysis/GIF_network/gifnet_sim.py $< $(SIMDATA_PATH)/input/OU_test.ldat $@ -v

$(SIMDATA_PATH)/input/OU_test.ldat : analysis/GIF_network/input_generators/OU_noise.py | $(SIMDATA_PATH)/input
	PYTHONPATH="$(shell pwd)" python $< $@ -d 1000. -v

.PHONY : clean
clean :
	rm $(GIFNET_PATH)/*; rm $(SIMDATA_PATH)/*;
