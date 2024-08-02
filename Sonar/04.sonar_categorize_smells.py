import pandas as pd
from googletrans import Translator

# Initialize the translator
translator = Translator()


# Function to translate text
def translate_text(text, dest_language='th'):
    try:
        translated = translator.translate(text, dest=dest_language)
        return translated.text
    except Exception as e:
        return str(e)


# Define a function to categorize each rule based on the provided keywords
def categorize_rule(description, name):
    bloaters_keywords = ["long method", "large class", "primitive obsession", "long parameter list", "duplicate",
                         "long", "large", "complexity", "size", "multiple variables", "multiple exit points",
                         "lines should not be too long", "too many parameters", "too many statements",
                         "too many fields", "not have too many lines",
                         "should not have too many parameters", "reusable resources", "objects should not be created"
                         "primitive "]

    oo_abusers_keywords = ["alternative", "classes", "different interfaces", "refused bequest", "switch statements",
                           "inheritance", "object-oriented", "encapsulation", "switch", "refused bequest", "bug",
                           "java language", "Method overrides", "main", "stream", "enum",
                           "fields should not be initialized to default values",
                           "class fields should not shadow parent class fields",
                           "should be passed in the correct order",
                           "don't access instance data should be",
                           "arrays should not be created for varargs parameters",
                           "checked exception", "methods returns", "branches",
                           "try-with-resources", "boolean expressions should not be gratuitous",
                           "exception should not be caught",
                           "consumer Builders should be used",
                           "should be set explicitly",
                           "should be preferred", "lambdas should not invoke other lambdas synchronously",
                           "aws region should not be set with a hardcoded String",
                           "boolean checks should not be inverted",
                           "exit methods should not be called", "generic exceptions should never be thrown",
                           "urls should not be hardcoded", "arrays should not be copied using loops"]

    change_preventers_keywords = ["divergent change", "parallel inheritance hierarchies", "shotgun surgery", "change",
                                  "dependency", "modularization", "parameter names", "method name", "class name",
                                  "name", "junit", "test", "test case", "test method", "test class", "test suite",
                                  "custom getter method", "local variable type", "variable type", "variable",
                                  "should be declared with base types", "replace", "assertthat",
                                  "containing characters subject to normalization should use", "diamond operator",
                                  "deprecated annotations", "explanations", "semicolons", "string object",
                                  "package declaration should match source file directory", "should be used instead"]

    dispensables_keywords = ["comments", "duplicate code", "data class", "dead code", "lazy class", "dead",
                             "speculative", "generality", "unused", "dead", "duplicate", "lazy", "data class",
                             "empty arrays", "track uses", "tags", "java 7", "java 8", "java", "java 9", "java 10",
                             "java 11", "java 12", "java 13", "java 14", "null", 'empty', 'track',
                             'multiline blocks', "should be avoided", "redundant", "should be combined", "removed",
                             "remove", "should not be stored", "delete", "deleteonexit", "should not be used",
                             "close curly brace", "statements should be merged", "should be removed",
                             "should be avoided", "merged", "identical implementations", "superfluous",
                             "annotation repetitions", "whitespace", "removal",
                             "statements should be on separate lines"]

    couplers_keywords = ["feature envy", "inappropriate intimacy", "incomplete library class", "message chains",
                         "coupling", "dependencies", "feature envy", "message chain", "inappropriate intimacy",
                         "super class", "sub class", "complex method", "complex", "library", "limited dependence",
                         "limited", "too many dependencies", "too many", "should not be coupled",
                         "should not be dependent", "methods should not have too many return statements",
                         "nested"]

    description_lower = description.lower()
    name_lower = name.lower()

    if any(keyword in (description_lower and name_lower) for keyword in bloaters_keywords):
        return "Bloaters"
    if any(keyword in (description_lower and name_lower) for keyword in oo_abusers_keywords):
        return "Object-Orientation Abusers"
    if any(keyword in (description_lower and name_lower) for keyword in change_preventers_keywords):
        return "Change Preventers"
    if any(keyword in (description_lower and name_lower) for keyword in dispensables_keywords):
        return "Dispensables"
    if any(keyword in (description_lower and name_lower) for keyword in couplers_keywords):
        return "Couplers"

    return "Uncategorized"


if __name__ == "__main__":
    # Load the SonarQube rules data
    java_rules_smell_df = pd.read_pickle("../Sonar/output/sonar_rules_version9.9.6.pkl")
    data = java_rules_smell_df['descriptionSections'].tolist()

    data = []
    for i in java_rules_smell_df['descriptionSections']:
        rule = i[0]['content']
        data.append(rule)

    df_data = pd.DataFrame(data, columns=['content'])

    java_rules_smell_df = pd.concat([java_rules_smell_df, df_data], axis=1)

    # Apply the function to each rule
    java_rules_smell_df['category'] = java_rules_smell_df.apply(
        lambda row: categorize_rule(row['content'], row['name']), axis=1)

    # Display the categorized rules
    categorize_smells = java_rules_smell_df[['key', 'name', 'content', 'category']]
    categorize_smells.to_parquet("../Sonar/output/sonar_rules_categorized.parquet")

    # Filter for rules that are categorized as "Uncategorized"
    uncategorized_rules_df = categorize_smells[categorize_smells['category'] == 'Uncategorized']
    print(uncategorized_rules_df)

    # translate the rules to Thai
    # uncategorized_rules_df['translated'] = uncategorized_rules_df['name'].apply(
    #     lambda x: translate_text(x, dest_language='th'))

    Bloaters = categorize_smells[categorize_smells['category'] == 'Bloaters']
    # print(Bloaters)
    Object_Orientation_Abusers = categorize_smells[categorize_smells['category'] == 'Object-Orientation Abusers']
    # print(Object_Orientation_Abusers)
    Change_Preventers = categorize_smells[categorize_smells['category'] == 'Change Preventers']
    # print(Change_Preventers)
    Dispensables = categorize_smells[categorize_smells['category'] == 'Dispensables']
    # print(Dispensables)
    Couplers = categorize_smells[categorize_smells['category'] == 'Couplers']
    # print(Couplers)
