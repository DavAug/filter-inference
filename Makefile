# Makefile for paper (figures and data)
.PHONY: plot_results clean_plots
plot_results: plot_cancer_growth_results

plot_cancer_growth_results: results/1_cancer_growth/results.ipynb
	jupyter nbconvert --to notebook --inplace --execute $<

# Reproduce results from scratch (may take several hours)
reproduce_results: reproduce_cancer_growth_results

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


# # Run analysis of in vitro study
# in_vitro_study: format_data infer_K_model infer_KP_model infer_KR_model plot_results

# format_data: data/raw_data/format_data.ipynb
# 	jupyter nbconvert --to notebook --inplace --execute $<
# infer_K_model: results/in_vitro_study/infer_K_model.ipynb
# 	jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=1000 --execute $<
# infer_KP_model: results/in_vitro_study/infer_KP_model.ipynb
# 	jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=1000 --execute $<
# infer_KR_model: results/in_vitro_study/infer_KR_model.ipynb
# 	jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=1000 --execute $<
# plot_results: results/in_vitro_study/plot_results.ipynb
# 	jupyter nbconvert --to notebook --inplace --execute $<

# # Run analysis of in silico study
# in_silico_study: synthesise_data infer_K_model_after_10h infer_K_model_after_15h infer_K_model_after_20h infer_K_model_after_30h infer_KP_model_after_10h infer_KP_model_after_15h infer_KP_model_after_20h infer_KP_model_after_30h infer_KR_model_after_10h infer_KR_model_after_15h infer_KR_model_after_20h infer_KR_model_after_30h predict_doses

# synthesise_data: results/in_silico_study/synthesise_data.ipynb
# 	jupyter nbconvert --to notebook --inplace --execute $<
# infer_K_model_after_10h: results/in_silico_study/infer_K_model_after_10h.ipynb
# 	jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=10000 --execute $<
# infer_K_model_after_15h: results/in_silico_study/infer_K_model_after_15h.ipynb
# 	jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=10000 --execute $<
# infer_K_model_after_20h: results/in_silico_study/infer_K_model_after_20h.ipynb
# 	jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=10000 --execute $<
# infer_K_model_after_30h: results/in_silico_study/infer_K_model_after_30h.ipynb
# 	jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=10000 --execute $<
# infer_KP_model_after_10h: results/in_silico_study/infer_KP_model_after_10h.ipynb
# 	jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=10000 --execute $<
# infer_KP_model_after_15h: results/in_silico_study/infer_KP_model_after_15h.ipynb
# 	jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=10000 --execute $<
# infer_KP_model_after_20h: results/in_silico_study/infer_KP_model_after_20h.ipynb
# 	jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=10000 --execute $<
# infer_KP_model_after_30h: results/in_silico_study/infer_KP_model_after_30h.ipynb
# 	jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=10000 --execute $<
# infer_KR_model_after_10h: results/in_silico_study/infer_KR_model_after_10h.ipynb
# 	jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=10000 --execute $<
# infer_KR_model_after_15h: results/in_silico_study/infer_KR_model_after_15h.ipynb
# 	jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=10000 --execute $<
# infer_KR_model_after_20h: results/in_silico_study/infer_KR_model_after_20h.ipynb
# 	jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=10000 --execute $<
# infer_KR_model_after_30h: results/in_silico_study/infer_KR_model_after_30h.ipynb
# 	jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=10000 --execute $<
# predict_doses: results/in_silico_study/predict_doses.ipynb
# 	jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.timeout=10000 --execute $<

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