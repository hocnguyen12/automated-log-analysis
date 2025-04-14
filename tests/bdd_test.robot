*** Settings ***
Resource        ../resources/keywords/bdd_keywords.resource

*** Test Cases ***
Test Calculator With BDD Syntax
    Given The Calculator Is Running
    When The User Enters The Term "1 + 1"
    And The User Triggers The Calculation
    Then The Result Should Be "2"