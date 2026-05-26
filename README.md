# Conflicting Needles-In-A-(Multilingual)-Haystack experiment
## Languages
The experiments were conducted using English and German text, to ensure model support, a high degree of control over manipulated text and an informed interpretation of the outputs.

## Data
The original texts used in the experiments are obtained from the European Commission’s Press Corner (Commission), available under the Creative Com-
mons Attribution 4.0 International (CC BY 4.0) licence.

The `scripts/webscraper.py` file can be used to collect a specified number of texts from the website in the selected languages. To add more languages, uncomment selection from the language_code list in the script. `links.txt` contains the hyperlinks to the speeches used in this experiment to create the five baseline haystacks in both English and German (base_haystacks) using:

```python scripts/mniah.py build_haystacks``` 

The full set of haystack configurations used in the experiments were created by first running: 

```python scripts/mniah.py inject``` 

, which inserts the needles into the distractor texts, and then combining the scripts with the corresponding `questions.json` conditions at inference level by executing `scripts/prompt_model.py`.  

## Inference and analysis
The `slurm/find_a_job.sh` script was used to run `scripts/prompt_model.py` on Alvis. The former collects the models' generated answers and calls `scripts/surprisal.py` to calculate needle-candidate log probabilities. The output of this is `llm_outputs.jsonl` , which `scripts/eval.py` uses to create the `clogit.parquet` file. The R-notebook `analysis_R.Rmd` takes these files and summarizes them into tables and plots for the paper.

## Paper
The corresponding paper will be uploaded to DiVA. (link to follow)
