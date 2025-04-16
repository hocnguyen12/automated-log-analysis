def build_log_text():
    msg = f"Test name: {item['test_name']}\n"
    msg += f"Doc: {item.get('doc', '')}\n"
    msg += f"Error: {item['error']}\n"
    for step in item.get("steps", []):
        msg += f"Step: {step['keyword']}\n"
        msg += f"Args: {' '.join(step['args'])}\n"
        msg += f"Status: {step['status']}\n"
        if step.get("doc"):
            msg += f"Doc: {step['doc']}\n"
        if step.get("messages"):
            msg += f"Messages: {' | '.join(step['messages'])}\n"
    return msg
    

def auto_label_fix_category(data):
    for item in data:
        if "fix_category" not in item or not item["fix_category"]:
            error = item["error"].lower()
            if "missing" in error and "argument" in error:
                item["fix_category"] = "missing_argument"
            elif "not found" in error or "selector" in error:
                item["fix_category"] = "invalid_selector"
            elif "assert" in error or "should be equal" in error:
                item["fix_category"] = "assertion_failed"
            elif "timeout" in error:
                item["fix_category"] = "timeout"
            elif "connection" in error:
                item["fix_category"] = "connection_error"
            else:
                item["fix_category"] = "other"
    return data