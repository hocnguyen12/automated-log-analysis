# automated-log-analysis

Tests taken from [robot framework official documentation](https://robotframework.org/?tab=0&example=Advanced%20Example#getting-started)
## Variables

This examples contains multiple tests with different variable types.
${Scalar} Variables

Scalar variables are single instances of a variable. It is used as is. It does not matter if the variable contains a list, object, dictionary or just a string or integer.
@{List-Like} Variables

Lists or list-like variables, which can be iterated, can be "unpacked". A variable containing a list with a $ sign as prefix is still a single variable. A variable containing a list with a @ sign is unpacked and handled as multiple values. You can compare it with a box of 10 cookies. ${box} is just the box, with all the 10 cookies inside. @{box} are 10 cookies without the box. When you hand over @{box} to a keyword, that keyword gets 10 arguments.

Accessing a single element of the box needs the variable as box. So in our case the prefix shall be $. ${box}[1] is the cookie with the index 1.
&{Dictionary} Variables

The & sign as prefix causes that a variable is handled as dictionary or also a list of key=value pairs. Similar to the list variables, dictionary variables are unpacked with the & sign but handle as single variable with the $ sign as prefix.

Accessing a single value by its key, requires in our case the prefix $ again.

    ${dict}[name] accesses the value behind the key "name".
    ${dict.name} is an alternative and sometime handy method to access elements of a dictionary.

## Behaviour-Driven Testing (Gherkin)

This example contains a test case written in a BDD-style with embedded arguments
Test Object

We are testing a fake Calculator implemented in Calculator.py. It provides the functions

    start_calculator
    calculate_term(term)

Test Suite

This test suite (Calculator_Test_Suite.robot) contains a test written in a BDD-style.
The prefixes Given/When/Then/And are dropped when they are matched to the respective keywords in Calc_keywords.resource.
Arguments like "1 + 1" are passed as embedded arguments to improve the readability. Note how the embedded arguments like "${term}" or "${expected}" (in Calc_keywords.resource) are mentioned in the keyword name and later used as variables inside the keyword. Adding quotes around embedded arguments is optional - but a good practice to improve readability.
The Keyword Set Test Variable is used to make a variables available everywhere within the test (as the variable scope is limited to the keyword by default).
 
## Advanced Example 2

This example contains some more advanced features of Robot Framework.
Test Object

We are testing here a backend api for user management. Users must authenticate before interaction. Depending on the authorizations, different actions can be carried out:

    Administrators can create users, alter user data and fetch details about existing users.
    Normal users can just fetch their own information and only alter their own details.
    Guest users can login but not modify anything.

Test Suite

This test suite (test.robot) contains six test cases such as Access All Users With Admin Rights. Test cases are calling either keywords from the resource file keywords.resource or the Library CustomLibrary.py

keywords.resource contains examples of variables, Return-Values, IF/ELSE and FOR-Loops.

# Log and Reports analysis

### Context :

### Steps
- Parse test results (output.xml) and extract:
    Test name
    Steps/keywords executed (in order)
    Final status
    Error message

- Represent this information in a machine-readable format (like vectors or structured objects).

- Analyze similarities across failed tests (based on both what was tested and how).

- Use AI to suggest fixes, based on:
    Common patterns
    Prior fixes
    Context of failure

### Detailed Steps
#### Step 1: Parse Tests with Full Context

Extract:
- Test name
- List of keyword calls (and arguments)
- Failure message
- Failure point (which keyword failed)

Example data structure:
```
{
  "test_name": "Create User",
  "keywords": [
    "Connect    http://localhost    key123",
    "Login User    admin    admin123",
    "Post New User    John    jdoe"
  ],
  "error": "TypeError: TestObject.__init__() missing 1 required positional argument: 'api_key'"
}
```

I can help you write a parser that does this.
#### Step 2: Represent the Test for Machine Learning

You have several options:
- TF-IDF or sentence embeddings on the error + keyword steps as text
- Structured embedding where each keyword is a token or vector
- Use tree-based models or graph-based models (for keyword call hierarchy)

#### Step 3: Cluster Similar Failures

Using:
- Clustering algorithms like KMeans, DBSCAN, etc.
- Or build a nearest neighbor search to find similar failures when a new one happens

#### Step 4: Suggest Fixes

Approaches:
- Associate each cluster with a correction suggestion
- Use past human-written fixes to train a retrieval model or fine-tune a language model
- Rules or heuristics (e.g., if TestObject.__init__ is failing → "Check API key in constructor")

#### Tools & Libraries You’ll Want
- xml.etree.ElementTree or lxml (to parse XML)
- scikit-learn or sentence-transformers (for clustering and embeddings)
- Optionally: faiss or annoy (for similarity search)
