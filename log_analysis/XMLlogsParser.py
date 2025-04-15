from xml.etree import ElementTree as ET
from pathlib import Path

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
            if msg is not None and msg.text and (msg.attrib.get("level") == "INFO" or msg.attrib.get("level") == "WARN")
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
    parts = [f"Test name: {test['name']}", f"Error: {test['error_message']}", f"Doc: {test['doc']}"]
    for step in test["steps"]:
        keyword = step["name"]
        args = ", ".join(step["args"])
        status = ", ".join(step["status"])
        parts.append(f"Step: {keyword}, Args: {args}")

        doc = step["doc"]
        message = step["message"]
        if doc:
            parts.append(f"Doc : {doc}")
        if message:
            parts.append(f"Message : {message}")
    return ". ".join(parts)



if __name__ == "__main__":
    fail_logs = parse_xml("reports/output.xml")

    print(f"Parsed xml : \n{fail_logs}")

    print("\nParsed xml pretty print : \n")
    pretty_print_fails(fail_logs)

    documents = [stringify_test_case(t) for t in fail_logs]
    print("\nStringified test cases : ")
    for doc in documents:
        print(f"String : {doc}\n")
