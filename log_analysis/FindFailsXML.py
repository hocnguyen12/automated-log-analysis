from xml.etree import ElementTree as ET
from pathlib import Path

# Load the XML content from the uploaded file
xml_path = Path("reports/output.xml")
tree = ET.parse(xml_path)
root = tree.getroot()

# Define a recursive function to extract keyword calls and their arguments
def extract_keywords(keyword_element, depth=0):
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

# Extract failed test cases with full step details
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


print(detailed_failed_tests)

pretty_print_fails(detailed_failed_tests)