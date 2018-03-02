
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


rule bin_mics: # Matt's MIC value encoding method
    input:
        "data/raw/GenotypicAMR.csv"
    output:
        "data/interim/mic_class_dataframe.pkl", "data/interim/mic_class_order_dict.pkl"
    script:
        "src/data/bin_mics.py"

rule count_kmers: # Create the kmer count database
    input:
        "data/raw/genomes/"
    output:
        expand("data/interim/kmer_counts.k{k}.l{l}.db", k=config["k"], l=config["l"])
    script:
        "src/data/count_kmers.py"

rule prepare_metadata: # Convert metadata sheet to useable format
    input:
        "data/raw/GenotypicAMR.csv",
        "data/interim/mic_class_dataframe.pkl"
    output:
        expand("data/interim/metadata/GenotypicAMR_{label}.pkl", label=['regular', 'clean', 'bin'])
    script:
        "src/data/prepare_metadata.py"

rule gather_data: # Create train/test data from metadata sheet and kmer count database, save data
    input:
        "data/raw/genomes/",
        expand("data/interim/kmer_counts.k{k}.l{l}.db", k=config["k"], l=config["l"]),
        "data/interim/metadata/GenotypicAMR_{label}.pkl"
    output:
        "data/processed/{drug}/{label}/data.pkl",
        "data/processed/{drug}/{label}/target.pkl"
    script:
        "src/features/gather_data.py"

rule cross_validate:
    input:
        "data/processed/{drug}/{label}/data.pkl",
        "data/processed/{drug}/{label}/target.pkl",
    output:
        "results/{drug}/{label}/{MLtype, (NN)|(SVM)}_results.pkl"
    script:
        "src/models/run_model.py"

rule make_predictions:
    input:
        "data/processed/{drug}/{label}/data.pkl",
        "data/processed/{drug}/{label}/target.pkl",
    output:
        "results/{drug}/{label}/{MLtype, (NN)|(SVM)}_predictions.pkl"
    script:
        "src/models/run_model.py"

rule test_drug:
    input:
        expand("results/{{drug}}/{label}/{MLtype}_results.pkl",
               label=['clean', 'bin', 'regular'], MLtype=['NN', 'SVM'])
    output:
        "results/{drug}.results"
    script:
        "src/results/combine_results.py"

rule test_all_drugs:
    input:
        expand("results/{drug}.results", drug=config["drugs"])
    output:
        "results/complete_results.pkl"
    script:
        "src/results/combine_results.py"

rule predictions_for_drug:
    input:
        expand("results/{{drug}}/{label}/{MLtype}_results.pkl",
               label=['clean', 'bin', 'regular'], MLtype=['NN', 'SVM'])
    output:
        "results/{drug}.predictions"
    script:
        "src/results/combine_predictions.py"

rule predictions_for_all_drugs:
    input:
        expand("results/{drug}.predictions", drug=config["drugs"])
    output:
        "results/complete_predictions.pkl"
    script:
        "src/results/combine_results.py"


