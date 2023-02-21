# RapheGIF

<div style="text-align: center;">
<img
src="https://img.shields.io/badge/python-2.7-important.svg"
alt="Python 2.7"/>
<img
src="https://img.shields.io/badge/os-macOS%20|%20linux-informational.svg"
alt="macOS or Linux"/>
<a href="LICENSE.txt">
<img
src="https://img.shields.io/badge/license-MIT-green.svg"
alt="MIT license"/>
</a>
<a href="https://black.readthedocs.io/en/stable/">
<img
src="https://img.shields.io/badge/codestyle-black-black.svg"
alt="Codestyle: Black"/>
</a>
<br/>
<a href="https://doi.org/10.7554/eLife.72951">
<img
src="https://img.shields.io/badge/paper%20doi-10.7554%2FeLife.72951-informational.svg"
alt="Paper doi: 10.7554/eLife.72951"/>
</a>
<a href="https://doi.org/10.5061/dryad.66t1g1k2w">
<img
src="https://img.shields.io/badge/data%20doi-10.5061%2Fdryad.66t1g1k2w-informational.svg"
alt="Data doi: 10.5061/dryad.66t1g1k2w"/>
</a>
</div>

<figure>
    <img src="visual_abstract.png" alt="Visual abstract"/>
    <figcaption>
        <b>A</b> Augmented generalized integrate-and-fire (aGIF) models add
        biophysical realism to GIF models. <b>B</b> aGIFs can be trained to
        imitate individual neurons.  <b>C</b> Resampling a bank of aGIFs
        yields a population model with realistic heterogeneity. <b>D</b>,
        <b>E</b> Simulated 5-HT populations encode the derivative of their
        input.
    </figcaption>
</figure>

Code for building experimentally-contstrained spiking neural network models of
the dorsal raphe nucleus (DRN) used in [Harkin et al.,
2023](https://doi.org/10.7554/eLife.72951).

> Emerson F. Harkin, Michael B. Lynn, Alexandre Payeur, Jean-François Boucher,
> Léa Caya-Bissonnette, Dominic Cyr, Chloe Stewart, André Longtin, Richard
> Naud, and Jean-Claude Béïque. Temporal derivative computation in the dorsal
> raphe network revealed by an experimentally-driven augmented
> integrate-and-fire modeling framework. eLife, 2023. doi:
> 10.7554/eLife.72951

Are you an electrophysiologist interested in fitting spiking neuron models with
a limited set of Hodgkin-Huxley currents to your data? See ["A User's Guide to
Generalized Integrate-and-Fire
Models"](https://doi.org/10.1007/978-3-030-89439-9_3) ([open
access](https://neurodynamic.uottawa.ca/neuralcoding/images/HarkinGIF.pdf)).
For more detailed information about how the serotonin neuron model used here
was developed, see *[A Simplified Serotonin Neuron
Model](http://dx.doi.org/10.20381/ruor-22786)*. ["Patch-clamp recordings from
dorsal raphe neurons"](https://doi.org/10.5061/dryad.66t1g1k2w) used to contrain
our models are freely available on the Dryad data repository.


## Overview

The project is broken up into `grr`, a reusable library for fitting GIF neuron
models forked from the excellent [GIF Fitting
Toolbox](https://github.com/pozzorin/GIFFittingToolbox), `analysis` scripts for
reproducing the models and simulations from [our
paper](https://doi.org/10.7554/eLife.72951), and
`figs/scripts` for reproducing the figures in our paper from the results of
`analysis`.

    .
    ├── analysis                        # Data analysis + simulation scripts
    │   ├── GIF_pipeline                # Fit spiking neuron models
    │   ├── GIFnet_pipeline             # Run network simulations
    │   │   └── input_generators
    │   ├── gaba_synapses
    │   └── gating                      # Characterize I_A in 5-HT neurons
    ├── figs
    │   └── scripts                     # Notebooks to reproduce figures
    └── grr                             # Library for fitting + running models


To use the augmented GIF model in your own work, clone or download this repo and
install `grr` using `pip install . && pip install -r requirements.txt` from
inside the project.

To reproduce the results from our paper, follow these steps:

1. Clone or download this repo and install `grr`.
2. Get a copy of the [raw data](https://doi.org/10.5061/dryad.66t1g1k2w) and
   extract it.
3. Set the environment variable `DATA_PATH` to the root directory of the raw
   data (recommended: `data/raw`) and the variable `IMG_PATH` to the location
   where you would like figures to be saved (recommended: `figs/ims`).
4. Run the scripts in `analysis`. `GIF_pipeline` must be run before
   `GIFnet_pipeline`.
5. Run the notebooks in `figs/scripts`.


## Contributions

Christian Pozzorini wrote the [GIF Fitting
Toolbox](https://github.com/pozzorin/GIFFittingToolbox) that forms the
foundation of `grr`. All remaining code was written by [Emerson
Harkin](https://github.com/efharkin). [Alexandre
Payeur](https://github.com/apayeur) prototyped some of the models included in
`grr` and provided valuable input on all aspects of the project along with
[Richard Naud](http://www.neurodynamic.uottawa.ca/neuralcoding/index.html) and
[Michael Lynn](https://github.com/micllynn).


## License

Science thrives on openness. This work is released under the [MIT
license](LICENSE.txt) and is free to use for any purpose. If you find our work
useful, please cite [our paper](https://doi.org/10.7554/eLife.72951)!

    @article{harkin2023temporal,
      title={Temporal derivative computation in the dorsal raphe network
        revealed by an experimentally-driven augmented integrate-and-fire
        modeling framework},
      author={Harkin, Emerson F and Lynn, Michael B and Payeur, Alexandre
        and Boucher, Jean-Fran{\c{c}}ois and Caya-Bissonnette, L{\'e}a
        and Cyr, Dominic and Stewart, Chloe and Longtin, Andr{\'e}
        and Naud, Richard and B{\'e}{\"\i}que, Jean-Claude},
      journal={eLife},
      volume={12},
      pages={e72951},
      year={2023},
      publisher={eLife Sciences Publications, Ltd},
      langid={english},
      doi={10.7554/eLife.72951},
    }
