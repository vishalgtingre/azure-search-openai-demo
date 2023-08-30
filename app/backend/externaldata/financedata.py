import re
import numpy as np


def extract_and_align_code(text):
    # Extract code wrapped in triple backticks
    code_block = re.search(r'```(.*?)```', text, re.DOTALL)
    
    if not code_block:
        return "No code block found!"
    
    code = code_block.group(1).splitlines()
    min_indent = min([len(re.match(r'^\s*', line).group(0)) for line in code if line.strip()])
    aligned_code = '\n'.join([line[min_indent:] for line in code])
    
    return aligned_code

# Program start
with open('samplecode.txt', 'r') as file:
    file_content = file.read()

print("Original message : ---- " + file_content)

from chat_code_utils import execute_extracted_code , extract_code_from_message , appendimportsandprints

code_from_msg = extract_code_from_message(file_content)

#with open('importlib.txt', 'r') as file:
#    importfile_content = file.read()

#with open('printoutput.txt','r') as file:
#    printfile_content = file.read()

codetoexecute = appendimportsandprints(code_from_msg)

print("Code Extracted from message :   start code ----- " + code_from_msg)
print("Code Extracted from message :   start code ----- " + codetoexecute)

from chat_code_utils import execute_extracted_code
data = execute_extracted_code(codetoexecute)
print(data)

