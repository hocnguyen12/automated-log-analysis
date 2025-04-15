from xml.etree import ElementTree as ET
from pathlib import Path


def extract_keywords(keyword_element, depth=0):
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
                steps = extract_keywords(test)
                detailed_failed_tests.append({
                    "name": test_name,
                    "error_message": error_message,
                    "steps": steps
                })
    return detailed_failed_tests

def pretty_print_fails(fails):
    BOLD = '\033[1m'
    END = '\033[0m'
    for fail in fails:
        print("\n")
        print(BOLD + "Test : " + fail["name"] + END)
        print("Message : " + fail["error_message"])
        print("Steps : ")
        if fail["steps"] == [] :
            print("\tEMPTY")
        else :
            for step in fail["steps"]:
                print('\t' + BOLD + "name : " +  step["name"] + END)
                print(f"\targs : {step['args']}")
                print('\t' + "status : " + step["status"])

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
