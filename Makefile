# Makefile for paper (figures and data)
# The makefile deviates from the structure of the publication
# 	1. Illustrate NLME model, MCMC sampling, and filters (Figs 1, 2, 3)
#	2. Illustrate NLME and filter inference results of early cancer growth, the
#      relationship between filter inference and summary statistics-based ABC
#      and the information loss results (Figs 4, 5 & Fig 11 & 12, 13)
#   3. Illustrate filter inference results of EGF pathway, the computational
#      costs of NLME inference and filter inference and the efficiency of
#      filter inference using Metropolis-Hastings and NUTS
#      (Figs 6, 7 & Figs 8, 9 & Fig 10)
#
# The published figures can be generated using the plot_* commands. The
# experiments can be reproduced and then plotted using the reproduce_*
# commands.

.PHONY: plot_results

plot_results: plot_methods plot_cancer_growth_results plot_egf_pathway_results
plot_methods: results/1_cancer_growth/illustrations.ipynb
	jupyter nbconvert --to notebook --inplace --execute $<
plot_cancer_growth_results: results/1_cancer_growth/results.ipynb
	jupyter nbconvert --to notebook --inplace --execute $<
plot_egf_pathway_results: results/2_egf_pathway/results.ipynb
	jupyter nbconvert --to notebook --inplace --execute $<

reproduce_results: reproduce_cancer_growth_results reproduce_egf_pathway_results

reproduce_cancer_growth_results: generate_cancer_data run_nlme_inference_cancer_data run_filter_inference_cancer_data run_information_loss_experiment_1 generate_bimodal_cancer_data run_nlme_inference_bimodal_cancer_data run_filter_inference_bimodal_cancer_data
generate_cancer_data:
	python results/1_cancer_growth/1_generate_data.py
run_nlme_inference_cancer_data:
	python results/1_cancer_growth/2_run_nlme_inference.py
run_filter_inference_cancer_data:
	python results/1_cancer_growth/3_run_filter_inference.py
run_information_loss_experiment_1:
	python results/1_cancer_growth/4_run_information_loss_experiment_1.py
generate_bimodal_cancer_data:
	python results/1_cancer_growth/5_generate_bimodal_cancer_growth_data.py
run_nlme_inference_bimodal_cancer_data:
	python results/1_cancer_growth/6_run_nlme_inference_bimodal_cancer_growth.py
run_filter_inference_bimodal_cancer_data:
	python results/1_cancer_growth/7_run_filter_inference_bimodal_cancer_growth.py

reproduce_egf_pathway_results: generate_egf_pathway_data run_filter_inference_egf_pathway_data run_filter_inference_egf_reduced_pathway_data run_cancer_growth_scaling_experiment_1 run_cancer_growth_scaling_experiment_2 run_egf_pathway_scaling_experiment_1 run_egf_pathway_scaling_experiment_2
generate_egf_pathway_data:
	python results/2_egf_pathway/8_generate_data.py
run_filter_inference_egf_pathway_data:
	python results/2_egf_pathway/9_run_filter_inference.py
run_filter_inference_egf_reduced_pathway_data:
	python results/2_egf_pathway/10_run_filter_inference_reduced_model.py
run_cancer_growth_scaling_experiment_1:
	python results/1_cancer_growth/11_run_scaling_experiment_1.py
run_cancer_growth_scaling_experiment_2:
	python results/1_cancer_growth/12_run_scaling_experiment_2.py
run_egf_pathway_scaling_experiment_1:
	python results/2_egf_pathway/13_run_scaling_experiment_1.py
run_egf_pathway_scaling_experiment_2:
	python results/2_egf_pathway/14_run_scaling_experiment_2.py
run_filter_inference_using_metropolis_hastings_cancer_growth:
	python results/1_cancer_growth/15_run_filter_inference_metropolis_hastings.py
run_filter_inference_using_metropolis_hastings_egf_pathway:
	python results/2_egf_pathway/16_run_filter_inference_metropolis_hastings.py

