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

import re

def extract_code_from_message(message):
    try:
        orig_code_match = re.search(r'```(.*?)```', message, re.DOTALL)
        if orig_code_match:
            code_match = orig_code_match.group(1)

            code_without_python = re.sub(r'\bpython\b', '', code_match, flags=re.IGNORECASE)

            def handle_print(match):
                print_contents = match.group(1)
                print_contents = print_contents.replace("\\n", "\n")
                return f'print({print_contents})'

            code_without_newlines = re.sub(r'(?<!print\()(.*?)\\n(.*?)(?!\))', r'\1\n\2', code_without_python)
            final_code = re.sub(r'print\((.*?)\\n(.*?)\)', handle_print, code_without_newlines)

            with open('modified_code.py', 'w') as output_file:
                output_file.write(final_code)

            return final_code.strip()
        
        return None
    except Exception as e:
        return f"An error occurred in 'extract_code_from_message': {str(e)}"
    
def appendimportsandprints(code_from_msg):
    try:
        with open('importlib.txt', 'r') as import_file:
            importfile_content = import_file.read()
        importfile_content = '''
import yfinance as yf
import numpy as np
from itertools import product
import json
import random
import numpy as np
import pandas as pd
'''

        with open('printoutput.txt', 'r') as print_file:
            printfile_content = print_file.read()
        
        printfile_content = '''
# Print the results
print("Mean Return")
print(meanReturn)
print("Covariance Matrix")
print(covMatrix)

'''

        
        codetoexecute = importfile_content + code_from_msg + printfile_content
        
        with open('codetoexecute.py', 'w') as output_file:
            output_file.write(codetoexecute)
        
        return codetoexecute
    except FileNotFoundError as e:
        return f"Error in 'appendimportsandprints': {e.filename} not found."
    except Exception as e:
        return f"An error occurred in 'appendimportsandprints': {str(e)}"
