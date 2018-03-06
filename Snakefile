
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
        expand("data/interim/metadata/GenotypicAMR_{label}.pkl",
               label=['regular', 'clean', 'bin'])
    script:
        "src/data/prepare_metadata.py"

# Create train/test data from metadata sheet and kmer count database, save data
rule gather_data:
    input:
        "data/raw/genomes/",
        expand("data/interim/kmer_counts.k{k}.l{l}.db",
               k=config["k"], l=config["l"]),
        "data/interim/metadata/GenotypicAMR_{label}.pkl"
    output:
        "data/processed/{drug}/{label}/data.pkl",
        "data/processed/{drug}/{label}/target.pkl"
    script:
        "src/features/gather_data.py"

# Perform cross validation if run_type == results
# Return predicted/true value comparison if run_type == predictions
rule run_model:
    input:
        "data/processed/{drug}/{label}/data.pkl",
        "data/processed/{drug}/{label}/target.pkl",
    output:
        "results/{drug}/{label}/{MLtype, (NN)|(SVM)}_{run_type}.pkl"
    script:
        "src/models/run_model.py"

# run_model for a drug with every model/label encoding combination
rule test_drug:
    input:
        expand("results/{{drug}}/{label}/{MLtype}_{{run_type}}.pkl",
               label=['clean', 'bin', 'regular'], MLtype=['NN', 'SVM'])
    output:
        "results/{drug}.{run_type}"
    script:
        "src/results/combine_{0}.py".format(wildcards.run_type)

# run_model for every drug/model/label encoding combination
rule test_all_drugs:
    input:
        expand("results/{drug}.{{run_type}}", drug=config["drugs"])
    output:
        "results/complete_{run_type}.pkl"
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
    output:
        "reports/tables/complete_predictions.csv"
    run:
        import pickle
        with open(input[0], 'rb') as f:
            data = pickle.load(f)
        data.to_csv(output[0], index=False, sep=',')

