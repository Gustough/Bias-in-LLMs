# Conflicting Needles-In-A-(Multilingual)-Haystack experiment
## Abstract
Large Language Models (LLMs) have been shown to struggle with resolving conflicting contextual information in retrieval settings. Previous work addressing this issue has leveraged adversarial Needle-in-a-Haystack frameworks using synthetic information to systematically evaluate conflict signalling and resolution across a variety of model sizes and families. By extending this paradigm to conflict settings using two competing needles containing real entities, this thesis investigates how structural encoding, language, and semantic framing influence conflict resolution behaviour in use-case oriented experiments. Conditional logistic regression and likelihood tests are applied across a wide range of haystack configurations to assess the extent to which multilingual LLMs signal conflict and which effects moderate selection likelihood. In line with previous findings, models generally exhibit low rates of conflict signalling, instead tending to select one option over the other. Across German and English texts, structural and semantic effects, model properties (family, size, alignment), and language were found to significantly shape selection likelihood. Larger models displayed higher rates of conflict signalling, while model family and alignment further impacted conflict resolution. The results highlight persisting limitations of LLMs that matter both in controlled experimental settings as well as in realistic application scenarios.

## Languages
The experiments were conducted using English and German text, to ensure model support, a high degree of control over manipulated text and an informed interpretation of the outputs.

## Data
The original texts used in the experiments are obtained from the European Commission’s Press Corner (Commission), available under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence.

The `scripts/webscraper.py` file can be used to collect a specified number of texts from the website in the selected languages. To add more languages, uncomment selection from the language_code list in the script. `links.txt` contains the hyperlinks to the speeches used in this experiment to create the five baseline haystacks in both English and German (base_haystacks) using:

```python scripts/mniah.py build_haystacks``` 

The full set of haystack configurations used in the experiments were created by first running: 

```python scripts/mniah.py inject``` 

, which inserts the needles into the distractor texts, and then combining the scripts with the corresponding `questions.json` conditions at inference level by executing `scripts/prompt_model.py`. For the paper, all inference was run on a GPU cluster, whereas the haystack creation was done locally. When recreatinig as is, hugginface-token specification and cluster-access have to be adjusted accordingly and the models need to be uploaded to the cluster. 

## Inference and Analysis
The `slurm/find_a_job.sh` script was used to run `scripts/prompt_model.py` on Alvis. The former collects the models' generated answers and calls `scripts/surprisal.py` to calculate needle-candidate log probabilities. The output of this is `llm_outputs.jsonl` , which `scripts/eval.py` uses to create the `clogit.parquet` file. The R-notebook `analysis_R.Rmd` takes that file and summarizes the results into tables and plots for the paper.

## Paper
The corresponding paper will be uploaded to DiVA. (link to follow)
