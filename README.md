# Conflicting Needles-In-A-(Multilingual)-Haystack experiment
## Languages
The experiments were conducted using English and German text, to ensure model support, a high degree of control over manipulated text and an informed interpretation of the outputs.

## Data
The original texts used in the experiments are obtained from the European Commission’s Press Corner (Commission), available under the Creative Com-
mons Attribution 4.0 International (CC BY 4.0) licence.

links.txt contains hyperlinks to the speeches used to create the five baseline haystacks. base_haystacks/ contains the haystacks created from these files and their corresponding official translations from the website. The full set of haystack configurations were created using the 'inject' method from the scripts/mniah.py file and the combination of the scripts with the corresponding question conditions at inference level from scripts/prompt_model.py  

## Scripts
The slurm/find_a_job.sh script was used to run scripts/prompt_model.py on Alvis, which calls scripts/surprisal.py to calculate needle-candidate log probabilities. 
The output of this is a .jsonl file, which scripts/eval.py uses to create .parquet files. The R-notebook analysis_R.Rmd takes these files and summarizes them into tables and plots for the paper. Individual script usage is described briefly in scripts/README.md

## Paper
The corresponding paper will be uploaded to DiVA. (link to follow)
