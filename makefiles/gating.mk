# Makefile to fit gating curves.

NBPATH=./analysis/gating
PDPATH=./data/processed/gating
RDPATH=./data/gating

.PHONY : all
all : $(NBPATH)/inspect_gating_pdata.ipynb

$(NBPATH)/inspect_gating_pdata.ipynb : $(PDPATH)/gating_params.dat $(PDPATH)/peakact_pdata.dat \
$(PDPATH)/peakinact_pdata.dat $(PDPATH)/ss_pdata.dat $(PDPATH)/peakact_fittedpts.dat \
$(PDPATH)/peakinact_fittedpts.dat $(PDPATH)/ss_fittedpts.dat
	jupyter nbconvert --to notebook --execute $@ --inplace

$(PDPATH)/gating_params.dat $(PDPATH)/peakact_pdata.dat \
$(PDPATH)/peakinact_pdata.dat $(PDPATH)/ss_pdata.dat $(PDPATH)/peakact_fittedpts.dat \
$(PDPATH)/peakinact_fittedpts.dat $(PDPATH)/ss_fittedpts.dat : ./analysis/gating/v_steps_analysis.py $(RDPATH)
	PYTHONPATH="$(shell pwd)" python ./analysis/gating/v_steps_analysis.py

