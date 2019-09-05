SIMDATA_PATH=data/simulations/GIF_network
GIFNET_PATH=data/models/GIF_network
GEN_PATH=analysis/GIF_network/model_generators
GIFMOD_PATH=data/models

REPS = 15

.PHONY : all
all : $(SIMDATA_PATH)/subsample_base $(SIMDATA_PATH)/subsample_noIA $(SIMDATA_PATH)/subsample_fixedIA

# Rule for running necessary simulations.
$(SIMDATA_PATH)/subsample_% : analysis/GIF_network/gifnet_sim.py $(GIFNET_PATH)/subsample_%.mod $(SIMDATA_PATH)/input/synpulse_low.dat $(SIMDATA_PATH)/input/synpulse_med.dat | $(SIMDATA_PATH)
	if [ ! -d "$@" ]; then \
		mkdir -p "$@" ; \
	fi; \
	for r in $$(seq 1 $(REPS)); do \
		PYTHONPATH="$(shell pwd)" python $< $(word 2, $^) $(SIMDATA_PATH)/input/synpulse_low.dat "$@/l_g_$$r.hdf5" --sigma-background 0.010 --seed-background $$r & \
	done ; \
	echo "Waiting for synpulse_low simulations with GABA..." ; \
	wait $$(jobs -rp) ; \
	for r in $$(seq 1 $(REPS)); do \
		PYTHONPATH="$(shell pwd)" python $< $(word 2, $^) $(SIMDATA_PATH)/input/synpulse_med.dat "$@/m_g_$$r.hdf5" --sigma-background 0.010 --seed-background $$r & \
	done ; \
	echo "Waiting for synpulse_med simulations with GABA..." ; \
	wait $$(jobs -rp) ; \
	for r in $$(seq 1 $(REPS)); do \
		PYTHONPATH="$(shell pwd)" python $< $(word 2, $^) $(SIMDATA_PATH)/input/synpulse_low.dat "$@/l_ng_$$r.hdf5" --sigma-background 0.010 --seed-background $$r --no-gaba & \
	done ; \
	echo "Waiting for synpulse_low simulations without GABA..." ; \
	wait $$(jobs -rp) ; \
	for r in $$(seq 1 $(REPS)); do \
		PYTHONPATH="$(shell pwd)" python $< $(word 2, $^) $(SIMDATA_PATH)/input/synpulse_med.dat "$@/m_ng_$$r.hdf5" --sigma-background 0.010 --seed-background $$r --no-gaba & \
	done ; \
	echo "Waiting for synpulse_med simulations without GABA..." ; \
	wait $$(jobs -rp) ; \

$(SIMDATA_PATH)/condgrad_l.hdf5 : analysis/GIF_network/gifnet_sim.py $(GIFNET_PATH)/condgrad.mod $(SIMDATA_PATH)/input/synpulse_low.dat | $(SIMDATA_PATH)
	PYTHONPATH="$(shell pwd)" python $^ $@ --sigma-background 0. --no-gaba -v

# Rules to generate model inputs.
$(SIMDATA_PATH)/input/synpulse_low.dat : analysis/GIF_network/input_generators/synaptic_pulse.py | $(SIMDATA_PATH)
	PYTHONPATH="$(shell pwd)" python $< $@ \
		--baseline-ser 0. --min-ser 0.02 --max-ser 0.070 \
		--baseline-gaba 0. --min-gaba 0.02 --max-gaba 0.070
$(SIMDATA_PATH)/input/synpulse_med.dat : analysis/GIF_network/input_generators/synaptic_pulse.py | $(SIMDATA_PATH)
	PYTHONPATH="$(shell pwd)" python $< $@ \
		--baseline-ser 0.01 --min-ser 0.02 --max-ser 0.070 \
	     	--baseline-gaba 0. --min-gaba 0.02 --max-gaba 0.070

# Rules to generate GIFnet models.
$(GIFNET_PATH)/subsample.mod : $(GEN_PATH)/subsample.py $(GIFMOD_PATH)/5HT/serkgifs.lmod $(GIFMOD_PATH)/GABA/gaba_gifs.mod | $(GIFNET_PATH)
	PYTHONPATH="$(shell pwd)" python $<

# All models named subsample_[something] are descended from subsample.mod. subsample_base.mod is a copy of subsample.mod
$(GIFNET_PATH)/subsample_%.mod : $(GEN_PATH)/subsample_%.py $(GIFNET_PATH)/subsample.mod | $(GIFNET_PATH)
	PYTHONPATH="$(shell pwd)" python $<

$(GIFNET_PATH)/condgrad.mod : $(GEN_PATH)/cond_gradient.py $(GIFMOD_PATH)/5HT/serkgifs.lmod | $(GIFNET_PATH)
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
