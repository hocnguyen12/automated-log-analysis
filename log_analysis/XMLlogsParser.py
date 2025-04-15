from xml.etree import ElementTree as ET
from pathlib import Path

def extract_keywords_old(keyword_element, depth=0):
    '''
    Recursive function to extract keyword calls and their arguments
    '''
    steps = []
    for kw in keyword_element.findall("kw"):
        name = kw.attrib.get("name", "UNKNOWN")
        args = [arg.text for arg in kw.findall("arg") if arg.text]
        status = kw.find("status").attrib.get("status") if kw.find("status") is not None else "UNKNOWN"
        step = {
            "name": name,
            "args": args,
            "status": status,
            "depth": depth
        }
        steps.append(step)
        # Recursively extract nested keywords
        steps.extend(extract_keywords(kw, depth + 1))
    return steps

def extract_keywords(keyword_element, depth=0):
    '''
    Recursive function to extract keyword calls and their arguments, status, doc, and messages.
    '''
    steps = []
    for kw in keyword_element.findall("kw"):
        name = kw.attrib.get("name", "UNKNOWN")
        args = [arg.text for arg in kw.findall("arg") if arg.text]
        status = kw.find("status").attrib.get("status") if kw.find("status") is not None else "UNKNOWN"

        doc = kw.find("doc")
        doc_text = doc.text.strip() if doc is not None and doc.text else ""

        #msgs = [msg.text.strip() for msg in kw.findall("msg") if msg is not None and msg.text]
        msgs = [
            msg.text.strip()
            for msg in kw.findall("msg")
            if msg is not None and msg.text and msg.attrib.get("level") == "INFO"
        ]

        step = {
            "name": name,
            "args": args,
            "status": status,
            "depth": depth,
            "doc": doc_text,
            "message": msgs
        }
        steps.append(step)
        # Recursively extract nested keywords
        steps.extend(extract_keywords(kw, depth + 1))
    return steps

def parse_xml(xml_path_string):
    xml_path = Path(xml_path_string)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    detailed_failed_tests = []
    for suite in root.iter("suite"):
        for test in suite.findall("test"):
            status = test.find("status")
            if status is not None and status.attrib.get("status") == "FAIL":
                test_name = test.attrib.get("name")
                error_message = status.text.strip() if status.text else "No message"

                doc = test.find("doc")
                doc_text = doc.text.strip() if doc is not None and doc.text else ""

                steps = extract_keywords(test)
                detailed_failed_tests.append({
                    "name": test_name,
                    "error_message": error_message,
                    "doc": doc_text,
                    "steps": steps
                })
    return detailed_failed_tests

def pretty_print_fails(fails):
    '''
    {'name': 'Test for the year 2022', 
    'error_message': '2025 != 2022', 
    'doc': 'Tests if it is still 2022...', 
    'steps': [
        {'name': 'Get Current Date', 
        'args': ['result_format=datetime'], 
        'status': 'PASS', 
        'depth': 0, 
        'doc': 'Returns current local or UTC time with an optional increment.', 
        'messages': ['${date} = 2025-04-14 10:57:37.508516']}, 

        {'name': 'Log', 
        'args': ['${date}'], 
        'status': 'PASS', 
        'depth': 0, 
        'doc': 'Logs the given message with the given level.', 
        'messages': ['2025-04-14 10:57:37.508516']}, 

        {'name': 'Should Be Equal As Strings', 
        'args': ['${date.year}', '2022'], 
        'status': 'FAIL', 
        'depth': 0, 
        'doc': 'Fails if objects are unequal after converting them to strings.', 
    'messages': ["Argument types are:\n<class 'int'>\n<class 'str'>", '2025 != 2022']}]}
    '''
    BOLD = '\033[1m'
    END = '\033[0m'
    for fail in fails:
        print("\n")
        print(BOLD + "Test : " + fail["name"] + END)
        print("Error message : " + fail["error_message"])
        print("Doc : " + fail["doc"])
        print("Steps : ")
        if fail["steps"] == [] :
            print("\tEMPTY")
        else :
            for step in fail["steps"]:
                print('\t' + BOLD + "name : " +  step["name"] + END)
                print(f"\targs : {step['args']}")
                print(f"\tstatus : {step['status']}")
                print(f"\tdepth : {step['depth']}")
                print(f"\tdoc : {step['doc']}")
                print(f"\tmessage : {step['message']}")
                print("\n")

def stringify_test_case(test):
    parts = [f"Test name: {test['name']}", f"Error: {test['error_message']}"]
    for step in test["steps"]:
        keyword = step["name"]
        args = ", ".join(step["args"])
        parts.append(f"Step: {keyword}, Args: {args}")
    return ". ".join(parts)

if __name__ == "__main__":
    fail_logs = parse_xml("reports/output.xml")

    print(fail_logs)

    pretty_print_fails(fail_logs)
