PROJECT_ROOT=$(shell pwd)
SCRIPT_PATH=./figs/scripts/bhrd
IMG_PATH=./figs/ims/2019BHRD

.PHONY : all
all : $(IMG_PATH)/fig1_5HTphysiol.png $(IMG_PATH)/fig3_somphysiol.png $(IMG_PATH)/fig4_somassociation.png

$(IMG_PATH)/fig1_5HTphysiol.png : $(SCRIPT_PATH)/5HT_intro.py | $(IMG_PATH)
	PYTHONPATH="$(PROJECT_ROOT)" python $<

$(IMG_PATH)/fig3_somphysiol.png : $(SCRIPT_PATH)/som_intro.py | $(IMG_PATH)
	PYTHONPATH="$(PROJECT_ROOT)" python $<

$(IMG_PATH)/fig4_somassociation.png : $(SCRIPT_PATH)/somassociation.py | $(IMG_PATH)
	PYTHONPATH="$(PROJECT_ROOT)" python $<

$(IMG_PATH) : 
	mkdir -p $(IMG_PATH)
