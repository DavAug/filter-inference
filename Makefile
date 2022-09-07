# Makefile for paper (figures and data)
.PHONY: plot_results clean_plots
plot_results: plot_cancer_growth_results plot_egf_pathway_results
plot_cancer_growth_results: results/1_cancer_growth/results.ipynb
	jupyter nbconvert --to notebook --inplace --execute $<
plot_egf_pathway_results: results/2_egf_pathway/results.ipynb
	jupyter nbconvert --to notebook --inplace --execute $<

# Reproduce results from scratch (may take several hours)
reproduce_results: reproduce_cancer_growth_results reproduce_egf_pathway_results

# 1. Cancer growth study
reproduce_cancer_growth_results: generate_cancer_data run_nlme_inference_cancer_data run_filter_inference_cancer_data run_information_loss_experiment_1 generate_bimodal_cancer_data run_nlme_inference_bimodal_cancer_data run_filter_inference_bimodal_cancer_data
generate_cancer_data:
	python results/1_cancer_growth/1_generate_cancer_growth_data.py
run_nlme_inference_cancer_data:
	python results/1_cancer_growth/2_run_nlme_inference_cancer_growth.py
run_filter_inference_cancer_data:
	python results/1_cancer_growth/3_run_filter_inference_cancer_growth.py
run_information_loss_experiment_1:
	python results/1_cancer_growth/4_run_information_loss_experiment_1.py
generate_bimodal_cancer_data:
	python results/1_cancer_growth/5_generate_bimodal_cancer_growth_data.py
run_nlme_inference_bimodal_cancer_data:
	python results/1_cancer_growth/6_run_nlme_inference_bimodal_cancer_growth.py
run_filter_inference_bimodal_cancer_data:
	python results/1_cancer_growth/7_run_filter_inference_bimodal_cancer_growth.py

# 2. EGF pathway study
reproduce_egf_pathway_results: generate_egf_pathway_data run_filter_inference_egf_pathway_data
generate_egf_pathway_data:
	python results/2_egf_pathway/1_generate_egf_pathway_data.py
run_filter_inference_egf_pathway_data:
	python results/2_egf_pathway/2_run_filter_inference_egf_pathway.py

# 2. TNF pathway study
reproduce_tnf_pathway_results: generate_tnf_pathway_data
generate_tnf_pathway_data:
	python results/3_tnf_pathway/1_generate_tnf_pathway_data.py
run_filter_inference_tnf_pathway_data:
	python results/3_tnf_pathway/2_run_filter_inference_tnf_pathway.py



# # Delete figures and derived data
# clean: clean_in_vitro_data clean_in_silico_data clean_figures
# clean_in_vitro_data:
# 	rm -f $(PWD)/results/in_vitro_study/derived_data/K_model_inference_data.nc
# 	rm -f $(PWD)/results/in_vitro_study/derived_data/KP_model_inference_data.nc
# 	rm -f $(PWD)/results/in_vitro_study/derived_data/KR_model_inference_data.nc
# clean_in_silico_data:
# 	rm -f $(PWD)/results/in_silico_study/derived_data/measurements.csv
# 	rm -f $(PWD)/results/in_silico_study/derived_data/K_model_posterior_10h.nc
# 	rm -f $(PWD)/results/in_silico_study/derived_data/K_model_posterior_15h.nc
# 	rm -f $(PWD)/results/in_silico_study/derived_data/K_model_posterior_20h.nc
# 	rm -f $(PWD)/results/in_silico_study/derived_data/K_model_posterior_30h.nc
# 	rm -f $(PWD)/results/in_silico_study/derived_data/KP_model_posterior_10h.nc
# 	rm -f $(PWD)/results/in_silico_study/derived_data/KP_model_posterior_15h.nc
# 	rm -f $(PWD)/results/in_silico_study/derived_data/KP_model_posterior_20h.nc
# 	rm -f $(PWD)/results/in_silico_study/derived_data/KP_model_posterior_30h.nc
# 	rm -f $(PWD)/results/in_silico_study/derived_data/KR_model_posterior_10h.nc
# 	rm -f $(PWD)/results/in_silico_study/derived_data/KR_model_posterior_15h.nc
# 	rm -f $(PWD)/results/in_silico_study/derived_data/KR_model_posterior_20h.nc
# 	rm -f $(PWD)/results/in_silico_study/derived_data/KR_model_posterior_30h.nc
# clean_figures:
# 	rm -f $(PWD)/results/in_vitro_study/in_vitro_data.pdf
# 	rm -f $(PWD)/results/in_vitro_study/model_weights_in_vitro_study.pdf
# 	rm -f $(PWD)/results/in_vitro_study/MLE_predictions.pdf
# 	rm -f $(PWD)/results/in_silico_study/log_reduction_doses.pdf