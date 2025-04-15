import json

def build_prompt(test_data, keyword_code, variable_file):
    return f"""
    You are an assistant for fixing failed Robot Framework tests.

    ### Test Name:
    {test_data['test_name']}

    ### Error Message:
    {test_data['error']}

    ### Test Steps:
    {json.dumps(test_data['steps'], indent=2)}

    ### Keyword Definitions:
    {keyword_code}

    ### Variables:
    {variable_file}

    ### Suggest the most likely cause of the failure and what change to make in the code or test.
    """