
import os

configfile: "config.yml"

#################################################################################
# FUNCTIONS                                                                     #
#################################################################################

def OPJ(*args):
    path = os.path.join(*args)
    return os.path.normpath(path)


#################################################################################
# GLOBALS                                                                       #
################################################################################# PROJECT_NAME = 'phenores' PROJECT_DIR = OPJ(os.path.dirname(__file__), os.pardir)

#################################################################################
# RULES                                                                         #
#################################################################################


# rule roary:
# 	input:
# 		"data/external/AAC.02140-16_zac003175944sd1.csv"
# 	output:
# 		"data/interim/accession_map.txt"
# 	shell:
# 		"""
# 		cut -f2 {input} | sort > data/interim/PMC5328538_original_genome_accessions.sort
# 		grep -v -P "^SRR"  data/interim/PMC5328538_original_genome_accessions.sort > data/interim/PMC5328538_assembly_ids.txt
# 		cut -f3,5 data/external/PRJNA242614_AssemblyDetails.txt | sort | join - data/interim/PMC5328538_assembly_ids.txt > data/interim/PMC5328538_assembly_biosample_ids.txt
# 		(
# 		cd data/interim
# 		./get_sra_accession.sh
# 		)
# 		(
# 		cd data/interim
# 		./merge.sh
# 		)
# 		sort -t' ' -k3 data/interim/PMC5328538_assembly_biosample_sra_ids.txt | join -t' ' -11 -23 -a1 -o1.1,2.1 data/interim/PMC5328538_sra_ids.txt - > {output}
# 		

# Matt's MIC value encoding method
rule bin_mics:
    input:
        "data/raw/GenotypicAMR.csv"
    output:
        "data/interim/mic_class_dataframe.pkl",
        "data/interim/mic_class_order_dict.pkl"
    script:
        "src/data/bin_mics.py"

# Create the kmer count database
rule count_kmers:
    input:
        "data/raw/genomes/"
    output:
        expand("data/interim/kmer_counts.k{k}.l{l}.db",
               k=config["k"], l=config["l"])
    script:
        "src/data/count_kmers.py"

# Convert metadata sheet to useable format
rule prepare_metadata:
    input:
        "data/raw/GenotypicAMR.csv",
        "data/interim/mic_class_dataframe.pkl"
    output:
        "data/interim/metadata/GenotypicAMR_{label}.pkl"
    script:
        "src/data/prepare_metadata.py"

# Create train/test data from metadata sheet and kmer count database, save data
rule gather_data:
    input:
        "data/raw/genomes/",
        expand("data/interim/kmer_counts.k{k}.l{l}.db",
               k=config["k"], l=config["l"]),
        expand("data/interim/metadata/GenotypicAMR_{label}.pkl",
               label=config['label_encoding'])
    output:
        "data/processed/{drug}/data.pkl",
        "data/processed/{drug}/target.pkl"
    script:
        "src/features/gather_data.py"

rule make_predictions:
    input:
        "data/processed/{drug}/data.pkl",
        "data/processed/{drug}/target.pkl",
    output:
       "results/{drug}/{model, (NN)|(SVM)}_predictions.pkl",
    script:
        "src/models/make_predictions.py"

rule cross_validate_svm:
    input:
        "data/processed/{drug}/data.pkl",
        "data/processed/{drug}/target.pkl",
    output:
       "results/{drug}/{model, (SVM)}_results.pkl",
       "results/{drug}/{model, (SVM)}_feature_coefs.pkl"
    script:
        'src/models/cross_validate.py'

rule cross_validate_nn:
    input:
        "data/processed/{drug}/data.pkl",
        "data/processed/{drug}/target.pkl",
    output:
       "results/{drug}/{model, (NN)}_results.pkl",
    script:
        'src/models/cross_validate.py'

# run_model for a drug with every model/label encoding combination
rule test_drug:
    input:
        expand("results/{{drug}}/{model}_results.pkl", model=['NN', 'SVM'])
    output:
        "results/{drug}.results"
    script:
        "src/results/combine_results.py"

# run_model for every drug/model/label encoding combination
rule test_all_drugs:
    input:
        expand("results/{drug}.results", drug=config["drugs"])
    output:
        "results/complete_results.pkl"
    script:
        "src/results/combine_results.py"

# Plot the accuracies of the models
rule plot_accuracies:
    input:
        "results/complete_results.pkl"
    output:
        "reports/figures/complete_results.pdf"
    script:
        "src/visualization/plot_accuracies.py"

# Save the predicted/true value combinations as csv
rule create_prediction_tables:
    input:
        "results/complete_predictions.pkl"

rule sample_distributions:
    input:
        "data/processed/{drug}/data.pkl",
        "data/processed/{drug}/target.pkl"
    output:
        "results/{drug}/sample_distributions.pkl"
    script:
        "src/data/sample_ditributions.py"

rule feature_importances:
    input:
        "results/{drug}/SVM_feature_coefs.pkl",
        "results/{drug}/sample_distributions.pkl"
    output:
        "results/{drug}/feature_importances.pkl"
    script:
        "src/results/feature_importances.py"
