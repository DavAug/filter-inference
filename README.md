# Filter inference: A scalable nonlinear mixed effects inference approach for snapshot time series data

This GitHub repository serves as documentation and reproduction source for the results published in XXX. It contains the raw data, the data derived during the analysis, the model specifications (SBML format) and executable scripts (Python scripts as well as Jupyter notebooks).

## Looking at the results

The results are documented by multiple notebooks. To open the notebooks, please follow the links below:

1. [Early cancer model inference results](https://github.com/DavAug/filter-inference/blob/main/results/1_cancer_growth/results.ipynb)
2. [EGF pathway model inference results](https://github.com/DavAug/filter-inference/blob/main/results/2_egf_pathway/results.ipynb)

## Reproducing the results

To reproduce the results, the GitHub repository can be cloned, and the scripts
cane be executed locally. For ease of execution, we prepared a `Makefile` that
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
