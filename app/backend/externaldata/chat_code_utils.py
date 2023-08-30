import io
import sys
import builtins
import re

def execute_extracted_code(code_block):
    # Capture standard output and error
    captured_output = io.StringIO()
    sys.stdout = captured_output
    sys.stderr = captured_output

    try:
        exec(code_block, {})
    except Exception as e:
        print("An error occurred:", e)

    # Restore standard output and error
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    return captured_output.getvalue()

def extract_code_from_chat(message):
    # Extract content between triple backticks
    orig_code_matches = re.findall(r'```(.*?)```', message, re.DOTALL)
    code_blocks = []

    for orig_code_match in orig_code_matches:
        code_match = orig_code_match.replace("\\n", "\n")
        code_blocks.append(code_match.strip())

    return code_blocks

def extract_code_from_message(message):
    # Extract content between triple backticks
    orig_code_match = re.search(r'```(.*?)```', message, re.DOTALL)
    if orig_code_match:
        code_match = orig_code_match.group(1)

        # Remove the word "python" if it exists in the code
        code_without_python = re.sub(r'\bpython\b', '', code_match, flags=re.IGNORECASE)

        # Define a function to handle print replacements
        def handle_print(match):
            print_contents = match.group(1)
            print_contents = print_contents.replace("\\n", "\n")
            return f'print({print_contents})'

        # Replace \n outside of print statements
        code_without_newlines = re.sub(r'(?<!print\()(.*?)\\n(.*?)(?!\))', r'\1\n\2', code_without_python)

        # Replace \n inside print statements
        final_code = re.sub(r'print\((.*?)\\n(.*?)\)', handle_print, code_without_newlines)

        return final_code.strip()
    
    return None


def appendimportsandprints(code_from_msg):
    with open('importlib.txt', 'r') as file:
        importfile_content = file.read()

    with open('printoutput.txt','r') as file:
        printfile_content = file.read()
    
    codetoexecute = importfile_content + code_from_msg + printfile_content
    return codetoexecute