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
streamlit run LogAnalysisUI.py --theme.base light
```

## Docker Image
Find the version of libraries:
```bash
pip freeze | grep -E 'streamlit|pandas|numpy|scikit-learn|sentence-transformers|faiss|joblib|matplotlib'
```

Create a docker image:
```bash
sudo docker build -t log-analyzer .
```

Running the docker image:
```bash
sudo docker run -p 8501:8501 log-analyzer 
```
```bash
docker run --rm -p 8501:8501 \
  -v "$PWD/LogAnalysisUI.py":/app/LogAnalysisUI.py \
  my-streamlit-app \
  streamlit run /app/LogAnalysisUI.py --server.port=8501 --server.address=0.0.0.0
```
Stop running the docker image:
```bash
sudo docker ps
sudo docker stop <CONTAINER_ID>
```

Delete a docker image:
```bash
sudo docker rmi -f <IMAGE_ID>
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


- To do so, first modify the `auto_label_fix_category()` to cover all the basic cases -> basically label the tests that can be labeled using only information found in the fail logs.

- Other fails will be labeled as `fix_category=unknown`.

- upload all your `output.xml` files in the `Train Model` tab.

- Click on `Train Model` (This will overwrite the model in the .`/model` directory so save it first if needed).

- This will generate a `./dataset/train_fails.json` file containing all the fails and prompt a list of all fails with category labeled as `unknown`. A list of fails will be displayed and for all of those, the category needs to be chosen by hand. This way, all the fails will be labeled and the model can be retrained by uploading only te `train_fails.json`, that you should rename (for example `dataset_257.json`).

### Analyze fails
- The user can upload any Robot Framework by dragging or chosing an `output.xml` file.

- Once the file is uploaded, the user can then choose any fail from this output file and make a prediction.

- The `Show Most Similar Fails` uses an FAISS similarity retrieval index to find the 3 most similar fails out of all the fails the model was trained on.

- The `Give Feedback` button can be used to add this fail to the training dataset. More specifically, a `feedback_fails.json` file is created or updated in the `dataset` directory.

- This feedback dataset is always merged with the given `output.xml` files, when training is done using xml files. This way, the model can improve over time.


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

- The problem is a `classification` problem where the goal is to make `predictions`
- Analyse Fails is the tab where predictions are made for new fails, after we executed a set of tests
- Before performing predictions, we need to build a predictor or a classifier -> There are different types of classifiers (random forest, ...)
- A classifier (as well as any machine learning algorithm) is a function that takes an input (here test fail logs) and gives an output (the root cause of the fail). The goal of this function is to approximate the 'real' function (that is unreachable). The function has a number of parameters that we need to modify so that the function approximates the real function as closely as possible. This is what we call 'fitting' or 'training'.
- There are unknown factors that can lead to a certain output, sometimes the data we have doesn't entirely explain the output/response

- The training tab can be used to upload a dataset OR output.xml files so and calls a script that trains a model
