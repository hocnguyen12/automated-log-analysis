'''
Converting your test failure data into structured JSON is useful when:

    - You want to move beyond text-based similarity (like TF-IDF or embeddings).

    - You want to train machine learning or deep learning models on structured data (classification, regression, transformers, etc.).

    - You want to later fine-tune or pretrain models using structured input (like LLMs or custom models).

    - You want to feed the data into non-NLP pipelines, such as anomaly detection, rule engines, or dashboards.


Goal	                                    |    Approach
--------------------------------------------|-------------------------------------------------------------------------
Train a classifier to predict fix category	|    Use error, steps, keyword usage as features
Train a neural model to suggest fix	        |    Fine-tune a transformer on error + steps → fix suggestion
Analyze trends	                            |    Extract stats: most common keyword before failure, average steps, etc.
Build an API	                            |    Your analyzer could return the structured JSON + fix proposals
'''
import json
from XMLlogsParser import parse_xml
import itertools

def convert_to_json_structured(test):
    '''
    input : output.xml content parsed with XMLlogsParser.parse_xml

    You can then feed this into a:
        - Transformer-based encoder
        - Token classifier
        - Sequence-to-sequence model (to suggest corrections)
    '''
    return {
        "test_name": test["test_name"],
        "error_message": test["error_message"],
        "doc": test["doc"],
        "steps": [
            {
                "keyword": step["keyword"],
                "args": step["args"],
                "status": step["status"],
                "depth": step["depth"],
                "doc": step["doc"],
                "messages": step["messages"]
            }
            for step in test["steps"]
        ]
    }

def save_converted_xml_to_json(xml_file, json_file_name):
    '''
    This function saves the fails to a json file to feed and train the model
    input : path to 'output.xml' file, path to json file that will contain the fails
    output : parsed fail logs
    '''
    fail_logs = parse_xml(xml_file)

    print([t for t in fail_logs])

    json_data = [convert_to_json_structured(t) for t in fail_logs]

    with open(json_file_name, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    return fail_logs

def merge_xml_training_data(xml_list, json_file_name):
    '''
    This function does the same treatment as 'save_converted_xml_to_json' but it takes as an input 
    a list of parsed xml contents instead of just one
    The goal is when we have multiple 'output.xml' we want to use as training for the model, 
    we can have a single '.json' that contains all the fails as a dataset
    '''
    training_data = []
    for xml in xml_list:
        training_data.extend(xml)
    json_data = [convert_to_json_structured(t) for t in training_data]

    with open(json_file_name, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    return training_data



if __name__ == "__main__":
    fail_logs = parse_xml("reports/output.xml")

    json_data = [convert_to_json_structured(t) for t in fail_logs]

    print(f"JSON data : {json_data}")

    with open("log_analysis/structured_failures.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)