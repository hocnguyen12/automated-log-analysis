# automated-log-analysis
This project aims to streamline the analysis of automated test failures from Robot Framework by combining rule-based classification, log parsing, and AI-assisted suggestions.

## Overview
Robot Framework generates detailed test execution reports, but identifying the root cause of failures and applying fixes can be time-consuming — especially in large test suites. This tool automates the first steps of failure triage by:

- Parsing output.xml files to extract failed test cases, error messages, and step details.

- Automatically labeling failures into fix categories using a rule-based system.

- Training a classifier (optional) on labeled failures to predict fix categories for new errors.

- Suggesting possible corrections or highlighting similar past failures using similarity search.

- Providing a simple Streamlit UI for training, labeling, and testing new logs.

## Running the project
```bash
git clone https://github.com/hocnguyen12/automated-log-analysis
cd automated-log-analysis
streamlit run LogAnalysisUI.py
```

## Instructions
### Train Model
This tab allows the user to train a new model based on their own dataset 

#### Using a JSON dataset
TODO : provide a dedicated script/option to create a JSON dataset from output.xml files -> manually label fails

#### Using output.xml files directly
The user can upload a single or multiple `output.xml` files.

This option requires the `auto_label_fix_category()` function to be written accordingly to the user's needs and the tests specificities.

This option is viable if the informations in the fail logs (error message, documentation, etc.) are relevant to deduce the fail category. (In this case, a classifier may not be necessary/optimal)

#### Hybrid method
The hybrid method consists in using automated labeling for easy cases combined with user input for more obscure cases.


To do so, first modify the `auto_label_fix_category()` to cover all the basic cases -> basically label the tests that can be labeled using only information found in the fail logs.

Other fails will be labeled as `fix_category=unknown`.

upload all your `output.xml` files in the `Train Model` tab.

Click on `Train Model` (This will overwrite the model in the .`/model` directory so save it first if needed).

This will generate a `./dataset/train_fails.json` file containing all the fails and prompt a list of all fails with category labeled as `unknown`.

A list of fails will be displayed and for all of those, the category needs to be chosen by hand. This way, all the fails will be labeled and the model can be retrained by uploading only te `train_fails.json`, that you should rename (for example `dataset_257.json`).

### Analyse fails


## Using a small local LLM
### Ollama
`https://ollama.com/`

- Install : 
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

```bash
pip install ollama
```

- Test model in terminal : 
```bash
ollama run phi
# or for code-specific model
ollama run codellama:7b-instruct
```

- test model with python api (automatic prompt building) :
```bash
python3 LLM/ollamaLM.py
```

## Future Use
While simple failures can be classified using clear rules, many real-world test failures involve multiple interacting issues that aren't visible in a single error message. This project lays the foundation for a more intelligent, context-aware analyzer — one that can grow over time with human feedback and possibly fine-tuned models.
