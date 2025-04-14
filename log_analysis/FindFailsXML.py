from xml.etree import ElementTree as ET
from pathlib import Path

# Load the XML content from the uploaded file
xml_path = Path("reports/output.xml")
tree = ET.parse(xml_path)
root = tree.getroot()

# Extract failed test cases with details (name and error message)
failed_tests = []

for suite in root.iter("suite"):
    for test in suite.findall("test"):
        status = test.find("status")
        if status is not None and status.attrib.get("status") == "FAIL":
            test_name = test.attrib.get("name")
            message = status.text.strip() if status.text else "No message"
            failed_tests.append({"name": test_name, "message": message})

def pretty_print_fails(fails):
    for fail in fails:
        print("- FAILED TEST : " + '\n\t' + fail["name"])
        print("MESSAGE :" + '\n\t' + fail["message"] + '\n')


pretty_print_fails(failed_tests)
