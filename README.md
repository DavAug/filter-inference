[![DOI](https://zenodo.org/badge/460385265.svg)](https://zenodo.org/badge/latestdoi/460385265)

# Filter inference: A scalable nonlinear mixed effects inference approach for snapshot time series data

This GitHub repository serves as documentation and reproduction source for the results published in XXX. It contains the raw data, the data derived during the analysis, the model specifications (SBML format) and executable scripts (Python scripts as well as Jupyter notebooks).

## Looking at the results

The results are documented by multiple notebooks. To open the notebooks, please follow the links below:

1. [Early cancer growth model inference results](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/results.ipynb)
2. [EGF pathway model inference results](https://github.com/DavAug/filter-inference/blob/main/results/2_egf_pathway/results.ipynb)

To inspect the scripts used to generate data, implement models and to estimate model parameters, please follow the links below:

### Datasets
Early cancer gowth datasets (no substructure):
1. [90 snapshot measurements](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/data/1_cancer_growth_data_15.csv)
2. [270 snapshot measurements](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/data/1_cancer_growth_data_45.csv)
3. [810 snapshot measurements](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/data/1_cancer_growth_data_135.csv)
4. [2430 snapshot measurements](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/data/1_cancer_growth_data_405.csv)

Early cancer growth datasets (slow and aggressive growth subpopulations):
1. [240 snapshot measurements](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/data/2_bimodal_cancer_growth_data_20.csv)
2. [3000 snapshot measurements](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/data/2_bimodal_cancer_growth_data_500.csv)

Epidermal growth factor signalling pathway dataset
1. [2400 snapshot measurements](https://github.com/DavAug/filter-inference/blob/main/results/2_egf_pathway/data/1_egf_pathway_data.csv)

### Model implementations

1. [Early cancer growth model](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/exponential_growth_model.py)
2. [EGF pathway model (SBML)](https://github.com/DavAug/filter-inference/blob/main/models/dixit_growth_factor_model.xml)
3. [EGF pathway model exposed to 2 constant EGF concentrations](https://github.com/DavAug/filter-inference/blob/main/results/2_egf_pathway/growth_factor_model.py)

### Data-generating scripts

Early cancer growth:
1. [Generate datasets with no substructure](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/1_generate_data.py)
2. [Generate datasets with slow and aggressive cancer growth](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/5_generate_bimodal_cancer_growth_data.py)

Epidermal growth factor signalling pathway:
1. [Generate dataset](https://github.com/DavAug/filter-inference/blob/main/results/2_egf_pathway/8_generate_data.py)

### Inference scripts
Early cancer growth:
1. [NLME inference from 90 snapshot measurements](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/2_run_nlme_inference.py)
2. [Filter inference from 90, 270, 810 and 2430 snapshot measurements](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/3_run_filter_inference.py)
3. [Filter inference from 2430 snapshot measurements with varying numbers of simulated individuals](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/4_run_information_loss_experiment_1.py)
4. [NLME inference from 240 snapshot measurements of structured population](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/6_run_nlme_inference_bimodal_cancer_growth.py)
5. [Filter infernece from 240 and 3000 snapshot measurements of structured population using different filters](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/7_run_filter_inference_bimodal_cancer_growth.py)

Epidermal growth factor signalling pathway:
1. [Filter inference from 2400 snapshot measurements](https://github.com/DavAug/filter-inference/blob/main/results/2_egf_pathway/9_run_filter_inference.py)

## Reproducing the results

To reproduce the results, the GitHub repository can be cloned, and the scripts
can be executed locally. For ease of execution, we prepared a `Makefile` that
runs the scripts in the correct order. Please find a step-by-step instruction
how to install the dependencies and how to reproduce the results, once the
repostory has been cloned.

#### 1. Install dependencies

- 1.1 Install CVODE (myokit uses CVODE to solve ODEs)

For Ubuntu:
```bash
apt-get update && apt-get install libsundials-dev
```
For MacOS:
 ```bash
brew update-reset && brew install sundials
```
For Windows:
    No action required. Myokit installs CVODE automatically.

- 1.2 Install Python dependencies

```bash
pip install -r requirements.txt
```

#### 2. Reproduce results

You can now reproduce the results by running

```bash
make all
```

This may take a while (hours to days), because you are re-running all scripts
sequentially. To reproduce only the plots from the existing data you can run

```bash
make plot_results
```

You can also run each script individually, but be aware that some scripts are
dependent on the data derived in other scripts.
