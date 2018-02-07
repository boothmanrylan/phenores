
import os

onstart:
    import yaml
    with open('config.yml', 'r') as f:
        data = yaml.load(f)
    data["train_splits"] = list(range(data["train_splits"]))
    data["train_runs"] = list(range(data["train_runs"]))
    with open('configuration.yml', 'w') as f:
        yaml.dump(data, f)

onsuccess:
    shell("rm configuration.yml")

onerror:
    shell("rm configuration.yml")

configfile: "configuration.yml"

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
        expand("data/interim/{{drug}}/{{label}}/{{MLtype}}_{{MLmethod}}/{k}/{file}.pkl",
               k=config["train_splits"],
               file=["x_train", "y_train", "x_test", "y_test"])
    script:
        "src/features/gather_data.py"

rule process_data: # Load train/test data, perform feature selection and data scaling, save processed data
    input:
        expand("data/interim/{{drug}}/{{label}}/{{MLtype}}_{{MLmethod}}/{k}/{file}.pkl",
               k=config["train_splits"],
               file=["x_train", "y_train", "x_test", "y_test"])
    output:
        expand("data/processed/{{drug}}/{{label}}/{{MLtype}}_{{MLmethod}}/{k}/{file}.pkl",
               k=config["train_splits"],
               file=["x_train", "y_train", "x_test", "y_test"])
    script:
        "src/features/process_data.py"

rule train_model: # Create model, pass train data to model, save model
    input:
        expand("data/processed/{{drug}}/{{label}}/{{MLtype}}_{{MLmethod}}/{k}/{file}.pkl",
               k=config["train_splits"],
               file=["x_train", "y_train"])
    output:
        expand("models/{{drug}}/{{label}}/{K}/{k}/{{MLtype, (NN)|(SVM)}}_{{MLmethod, (C|R)}}.h5",
               k=config["train_splits"],
               K=config["train_runs"])
    script:
        "src/models/train_{wildcards.MLtype}_{wildcards.MLmethod}.py"

rule test_model: # Load pre-trained model, pass test data to model, write result to file
    input:
        expand("models/{{drug}}/{{label}}/{K}/{k}/{{MLtype}}_{{MLmethod}}.h5",
               k=config["train_splits"],
               K=config["train_runs"]),
        expand("data/processed/{{drug}}/{{label}}/{{MLtype}}_{{MLmethod}}/{k}/{file}.pkl",
               k=config["train_splits"],
               file=["x_test", "y_test"])
    output:
        "results/{drug}/{label}/{MLtype, (NN)|(SVM)}_{MLmethod, (C|R)}.txt"
    script:
        "src/models/test_{wildcards.MLtype}.py"

rule test_drug:
    input:
        expand("results/{{drug}}/{label}/{MLtype}_{MLmethod}.txt",
               label=['bin', 'clean', 'regular'], MLtype=['SVM', 'NN'],
               MLmethod=['R', 'C'])
    output:
        "results/{drug}.results"
    run:
        for results_file in input:
            with open(results_file, 'r') as f:
                data = f.readlines()[-1]
            with open(output[0], 'a') as f:
                f.write("{0}\n".format(results_file))
                f.write("{0}\n".format(data))

rule test_all_drugs:
    input:
        expand("results/{drug}.results", drug=config["drugs"])
