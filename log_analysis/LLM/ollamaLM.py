import ollama
from PromptBuilder import build_prompt

test_data = {
    "test_name": "Update Password Test",
    "error": "TypeError: TestObject.__init__() missing 1 required positional argument: 'api_key'",
    "steps": [
        {
            "keyword": "Connect",
            "args": ["http://localhost"],
            "status": "FAIL",
            "messages": []
        }
    ]
}

# Sample keyword and variable contents
keyword_code = """
class CustomLibrary:
    def connect(self, ip):
        self.connection = TestObject(ip)
"""

variable_file = """
${API_KEY}    test-api-key-123
"""

prompt = build_prompt(test_data, keyword_code, variable_file)

response = ollama.chat(
    model='phi',
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print(response['message']['content'])
