RAWDATA=../../data/raw
PROCDATA=../../data/processed
MODS=../../data/models

.PHONY : all
all : benchmarks

.PHONY : benchmarks
benchmarks : models run_benchmarks.py
	sh ./run_all_benchmarks.sh

# Fit single cell models.
.PHONY : models
models : $(MODS)/5HT/5HT_GIFs.lmod $(MODS)/5HT/5HT_AugmentedGIFs.lmod $(MODS)/5HT/5HT_iGIF_NPs.lmod $(MODS)/5HT/5HT_iGIF_VRs.lmod \
	$(MODS)/GABA/GABA_GIFs.lmod $(MODS)/GABA/GABA_AugmentedGIFs.lmod $(MODS)/GABA/GABA_iGIF_NPs.lmod $(MODS)/GABA/GABA_iGIF_VRs.lmod \
	$(MODS)/mPFC/mPFC_GIFs.lmod $(MODS)/mPFC/mPFC_AugmentedGIFs.lmod $(MODS)/mPFC/mPFC_iGIF_NPs.lmod $(MODS)/mPFC/mPFC_iGIF_VRs.lmod

$(MODS)/5HT/5HT_GIFs.lmod : fit_mods.py $(PROCDATA)/5HT_fastnoise/5HT_goodcells.ldat 5HT_fit_opts.json
	python $< $(PROCDATA)/5HT_fastnoise/5HT_goodcells.ldat "$@" 5HT_fit_opts.json -m GIF -v
$(MODS)/5HT/5HT_AugmentedGIFs.lmod : fit_mods.py $(PROCDATA)/5HT_fastnoise/5HT_goodcells.ldat 5HT_fit_opts.json
	python $< $(PROCDATA)/5HT_fastnoise/5HT_goodcells.ldat "$@" 5HT_fit_opts.json -m AugmentedGIF -v
$(MODS)/5HT/5HT_iGIF_NPs.lmod : fit_mods.py $(PROCDATA)/5HT_fastnoise/5HT_goodcells.ldat 5HT_fit_opts.json
	python $< $(PROCDATA)/5HT_fastnoise/5HT_goodcells.ldat "$@" 5HT_fit_opts.json -m iGIF_NP -v
$(MODS)/5HT/5HT_iGIF_VRs.lmod : fit_mods.py $(PROCDATA)/5HT_fastnoise/5HT_goodcells.ldat 5HT_fit_opts.json
	python $< $(PROCDATA)/5HT_fastnoise/5HT_goodcells.ldat "$@" 5HT_fit_opts.json -m iGIF_VR -v

$(MODS)/GABA/GABA_GIFs.lmod : fit_mods.py $(PROCDATA)/GABA_fastnoise/GABA_goodcells.ldat GABA_fit_opts.json
	python $< $(PROCDATA)/GABA_fastnoise/GABA_goodcells.ldat "$@" GABA_fit_opts.json -m GIF -v
$(MODS)/GABA/GABA_AugmentedGIFs.lmod : fit_mods.py $(PROCDATA)/GABA_fastnoise/GABA_goodcells.ldat GABA_fit_opts.json
	python $< $(PROCDATA)/GABA_fastnoise/GABA_goodcells.ldat "$@" GABA_fit_opts.json -m AugmentedGIF -v
$(MODS)/GABA/GABA_iGIF_NPs.lmod : fit_mods.py $(PROCDATA)/GABA_fastnoise/GABA_goodcells.ldat GABA_fit_opts.json
	python $< $(PROCDATA)/GABA_fastnoise/GABA_goodcells.ldat "$@" GABA_fit_opts.json -m iGIF_NP -v
$(MODS)/GABA/GABA_iGIF_VRs.lmod : fit_mods.py $(PROCDATA)/GABA_fastnoise/GABA_goodcells.ldat GABA_fit_opts.json
	python $< $(PROCDATA)/GABA_fastnoise/GABA_goodcells.ldat "$@" GABA_fit_opts.json -m iGIF_VR -v

$(MODS)/mPFC/mPFC_GIFs.lmod : fit_mods.py $(PROCDATA)/mPFC_fastnoise/mPFC_goodcells.ldat mPFC_fit_opts.json
	python $< $(PROCDATA)/mPFC_fastnoise/mPFC_goodcells.ldat "$@" mPFC_fit_opts.json -m GIF -v
$(MODS)/mPFC/mPFC_AugmentedGIFs.lmod : fit_mods.py $(PROCDATA)/mPFC_fastnoise/mPFC_goodcells.ldat mPFC_fit_opts.json
	python $< $(PROCDATA)/mPFC_fastnoise/mPFC_goodcells.ldat "$@" mPFC_fit_opts.json -m AugmentedGIF -v
$(MODS)/mPFC/mPFC_iGIF_NPs.lmod : fit_mods.py $(PROCDATA)/mPFC_fastnoise/mPFC_goodcells.ldat mPFC_fit_opts.json
	python $< $(PROCDATA)/mPFC_fastnoise/mPFC_goodcells.ldat "$@" mPFC_fit_opts.json -m iGIF_NP -v
$(MODS)/mPFC/mPFC_iGIF_VRs.lmod : fit_mods.py $(PROCDATA)/mPFC_fastnoise/mPFC_goodcells.ldat mPFC_fit_opts.json
	python $< $(PROCDATA)/mPFC_fastnoise/mPFC_goodcells.ldat "$@" mPFC_fit_opts.json -m iGIF_VR -v

# Quality control and preprocessing.
$(PROCDATA)/5HT_fastnoise/5HT_goodcells.ldat : AEC_QC.py 5HT_QC_opts.json $(shell find $(RAWDATA)/5HT/fast_noise -iname '*.abf' -o -iname '*.csv')
	python $< $(RAWDATA)/5HT/fast_noise/index.csv $@ 5HT_QC_opts.json -v

$(PROCDATA)/GABA_fastnoise/GABA_goodcells.ldat : AEC_QC.py GABA_QC_opts.json $(shell find $(RAWDATA)/GABA/OU_noise -iname '*.abf' -o -iname '*.csv')
	python $< $(RAWDATA)/GABA/OU_noise/index.csv $@ GABA_QC_opts.json -v

$(PROCDATA)/mPFC_fastnoise/mPFC_goodcells.ldat : AEC_QC.py mPFC_QC_opts.json $(shell find $(RAWDATA)/mPFC/mPFC_spiking -iname '*.abf' -o -iname '*.csv')
	python $< $(RAWDATA)/mPFC/mPFC_spiking/index.csv $@ mPFC_QC_opts.json -v

.PHONY : clean
clean :
	rm -f \
	$(PROCDATA)/5HT_fastnoise/5HT_goodcells.ldat $(PROCDATA)/GABA_fastnoise/GABA_goodcells.ldat $(PROCDATA)/mPFC_fastnoise/mPFC_goodcells.ldat \
	$(MODS)/5HT/5HT_GIFs.lmod $(MODS)/5HT/5HT_AugmentedGIFs.lmod $(MODS)/5HT/5HT_iGIF_NPs.lmod $(MODS)/5HT/5HT_iGIF_VRs.lmod \
	$(MODS)/GABA/GABA_GIFs.lmod $(MODS)/GABA/GABA_AugmentedGIFs.lmod $(MODS)/GABA/GABA_iGIF_NPs.lmod $(MODS)/GABA/GABA_iGIF_VRs.lmod \
	$(MODS)/mPFC/mPFC_GIFs.lmod $(MODS)/mPFC/mPFC_AugmentedGIFs.lmod $(MODS)/mPFC/mPFC_iGIF_NPs.lmod $(MODS)/mPFC/mPFC_iGIF_VRs.lmod \
	$(MODS)/5HT/5HT_goodcells_benchmark_sample_traces.pkl $(MODS)/5HT/5HT_goodcells_benchmark_Md_8.csv $(MODS)/5HT/5HT_goodcells_benchmark_R2_V.csv $(MODS)/5HT/5HT_goodcells_benchmark_R2_dV.csv \
	$(MODS)/GABA/GABA_goodcells_benchmark_sample_traces.pkl $(MODS)/GABA/GABA_goodcells_benchmark_Md_8.csv $(MODS)/GABA/GABA_goodcells_benchmark_R2_V.csv $(MODS)/GABA/GABA_goodcells_benchmark_R2_dV.csv \
	$(MODS)/mPFC/mPFC_goodcells_benchmark_sample_traces.pkl $(MODS)/mPFC/mPFC_goodcells_benchmark_Md_8.csv $(MODS)/mPFC/mPFC_goodcells_benchmark_R2_V.csv $(MODS)/mPFC/mPFC_goodcells_benchmark_R2_dV.csv
