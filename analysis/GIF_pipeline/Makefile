RAWDATA=../../data/raw
PROCDATA=../../data/processed

.PHONY : all
all : $(PROCDATA)/5HT_fastnoise/5HT_goodcells.ldat $(PROCDATA)/GABA_fastnoise/GABA_goodcells.ldat $(PROCDATA)/mPFC_fastnoise/mPFC_goodcells.ldat

$(PROCDATA)/5HT_fastnoise/5HT_goodcells.ldat : AEC_QC.py 5HT_QC_opts.json $(shell find $(RAWDATA)/5HT/fast_noise -iname '*.abf' -o -iname '*.csv')
	python $< $(RAWDATA)/5HT/fast_noise/index.csv $@ 5HT_QC_opts.json -v

$(PROCDATA)/GABA_fastnoise/GABA_goodcells.ldat : AEC_QC.py GABA_QC_opts.json $(shell find $(RAWDATA)/GABA/OU_noise -iname '*.abf' -o -iname '*.csv')
	python $< $(RAWDATA)/GABA/OU_noise/index.csv $@ GABA_QC_opts.json -v

$(PROCDATA)/mPFC_fastnoise/mPFC_goodcells.ldat : AEC_QC.py mPFC_QC_opts.json $(shell find $(RAWDATA)/mPFC/mPFC_spiking -iname '*.abf' -o -iname '*.csv')
	python $< $(RAWDATA)/mPFC/mPFC_spiking/index.csv $@ mPFC_QC_opts.json -v

.PHONY : clean
clean :
	rm -f $(PROCDATA)/5HT_fastnoise/5HT_goodcells.ldat $(PROCDATA)/GABA_fastnoise/GABA_goodcells.ldat $(PROCDATA)/mPFC_fastnoise/mPFC_goodcells.ldat
