SIMDATA_PATH=data/simulations/GIF_network
GIFNET_PATH=data/models/GIF_network
GEN_PATH=analysis/GIF_network/model_generators
GIFMOD_PATH=data/models

.PHONY : all
all : \
$(SIMDATA_PATH)/subsample_base_l_g.hdf5 $(SIMDATA_PATH)/subsample_base_l_ng.hdf5 \
$(SIMDATA_PATH)/subsample_base_m_g.hdf5 $(SIMDATA_PATH)/subsample_base_m_ng.hdf5 \
$(SIMDATA_PATH)/subsample_base_h_g.hdf5 $(SIMDATA_PATH)/subsample_base_h_ng.hdf5 \
$(SIMDATA_PATH)/subsample_noIA_l_g.hdf5 $(SIMDATA_PATH)/subsample_noIA_l_ng.hdf5 \
$(SIMDATA_PATH)/subsample_noIA_m_g.hdf5 $(SIMDATA_PATH)/subsample_noIA_m_ng.hdf5 \
$(SIMDATA_PATH)/subsample_noIA_h_g.hdf5 $(SIMDATA_PATH)/subsample_noIA_h_ng.hdf5 \
$(SIMDATA_PATH)/subsample_fixedIA_l_g.hdf5 $(SIMDATA_PATH)/subsample_fixedIA_l_ng.hdf5 \
$(SIMDATA_PATH)/subsample_fixedIA_m_g.hdf5 $(SIMDATA_PATH)/subsample_fixedIA_m_ng.hdf5 \
$(SIMDATA_PATH)/subsample_fixedIA_h_g.hdf5 $(SIMDATA_PATH)/subsample_fixedIA_h_ng.hdf5 \

# Rule for running necessary simulations.
$(SIMDATA_PATH)/subsample_%_l_g.hdf5 : analysis/GIF_network/gifnet_sim.py $(GIFNET_PATH)/subsample_%.mod  $(SIMDATA_PATH)/input/synpulse_low.dat | $(SIMDATA_PATH)
	PYTHONPATH="$(shell pwd)" python $^ $@ -v
$(SIMDATA_PATH)/subsample_%_l_ng.hdf5 : analysis/GIF_network/gifnet_sim.py $(GIFNET_PATH)/subsample_%.mod $(SIMDATA_PATH)/input/synpulse_low.dat | $(SIMDATA_PATH)
	PYTHONPATH="$(shell pwd)" python $^ $@ --no-gaba -v
$(SIMDATA_PATH)/subsample_%_m_g.hdf5 : analysis/GIF_network/gifnet_sim.py $(GIFNET_PATH)/subsample_%.mod $(SIMDATA_PATH)/input/synpulse_med.dat | $(SIMDATA_PATH)
	PYTHONPATH="$(shell pwd)" python $^ $@ -v
$(SIMDATA_PATH)/subsample_%_m_ng.hdf5 : analysis/GIF_network/gifnet_sim.py $(GIFNET_PATH)/subsample_%.mod $(SIMDATA_PATH)/input/synpulse_med.dat | $(SIMDATA_PATH)
	PYTHONPATH="$(shell pwd)" python $^ $@ --no-gaba -v
$(SIMDATA_PATH)/subsample_%_h_g.hdf5 : analysis/GIF_network/gifnet_sim.py $(GIFNET_PATH)/subsample_%.mod $(SIMDATA_PATH)/input/synpulse_hi.dat | $(SIMDATA_PATH)
	PYTHONPATH="$(shell pwd)" python $^ $@ -v
$(SIMDATA_PATH)/subsample_%_h_ng.hdf5 : analysis/GIF_network/gifnet_sim.py $(GIFNET_PATH)/subsample_%.mod $(SIMDATA_PATH)/input/synpulse_hi.dat | $(SIMDATA_PATH)
	PYTHONPATH="$(shell pwd)" python $^ $@ --no-gaba -v

# Rules to generate model inputs.
$(SIMDATA_PATH)/input/synpulse_low.dat : analysis/GIF_network/input_generators/synaptic_pulse.py | $(SIMDATA_PATH)
	PYTHONPATH="$(shell pwd)" python $< $@ --baseline-ser -0.01 --baseline-gaba -0.01 --sigma-ser-background 0.005 --sigma-gaba-background 0.005
$(SIMDATA_PATH)/input/synpulse_med.dat : analysis/GIF_network/input_generators/synaptic_pulse.py | $(SIMDATA_PATH)
	PYTHONPATH="$(shell pwd)" python $< $@ --baseline-ser 0.    --baseline-gaba 0.    --sigma-ser-background 0.005 --sigma-gaba-background 0.005
$(SIMDATA_PATH)/input/synpulse_hi.dat : analysis/GIF_network/input_generators/synaptic_pulse.py | $(SIMDATA_PATH)
	PYTHONPATH="$(shell pwd)" python $< $@ --baseline-ser 0.01  --baseline-gaba 0.01  --sigma-ser-background 0.005 --sigma-gaba-background 0.005

# Rules to generate GIFnet models.
$(GIFNET_PATH)/subsample.mod : $(GEN_PATH)/subsample.py $(GIFMOD_PATH)/5HT/serkgifs.lmod $(GIFMOD_PATH)/GABA/gaba_gifs.mod | $(GIFNET_PATH)
	PYTHONPATH="$(shell pwd)" python $<

# All models named subsample_[something] are descended from subsample.mod. subsample_base.mod is a copy of subsample.mod
$(GIFNET_PATH)/subsample_%.mod : $(GEN_PATH)/subsample_%.py $(GIFNET_PATH)/subsample.mod | $(GIFNET_PATH)
	PYTHONPATH="$(shell pwd)" python $<


# Rules to create needed folders.
$(SIMDATA_PATH) $(GIFNET_PATH) $(SIMDATA_PATH)/input :
	mkdir -p $@


.PHONY : test
test : $(SIMDATA_PATH)/subsamp_test.hdf5

$(SIMDATA_PATH)/subsamp_test.hdf5 : $(GIFNET_PATH)/subsample.mod $(SIMDATA_PATH)/input/OU_test.ldat analysis/GIF_network/gifnet_sim.py | $(SIMDATA_PATH)
	PYTHONPATH="$(shell pwd)" python analysis/GIF_network/gifnet_sim.py $< $(SIMDATA_PATH)/input/OU_test.ldat $@ -v

$(SIMDATA_PATH)/input/OU_test.ldat : analysis/GIF_network/input_generators/OU_noise.py | $(SIMDATA_PATH)/input
	PYTHONPATH="$(shell pwd)" python $< $@ -d 1000. -v

.PHONY : clean
clean :
	rm $(GIFNET_PATH)/*; rm $(SIMDATA_PATH)/*;
