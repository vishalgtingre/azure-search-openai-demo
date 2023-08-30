import re
import numpy as np
from chat_code_utils import execute_extracted_code , extract_code_from_message , appendimportsandprints


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

#print("Original message : ---- " + file_content)
codemessage = extract_code_from_message(file_content)
chat_content_str = ""
chat_content_str += "1. Extracted the Auto generated Code\n"
codemessage = appendimportsandprints(codemessage)
chat_content_str += "2. Appended Imports and Output Statements\n"
chat_content = execute_extracted_code(codemessage)
chat_content_str += "3. Result from the extracted code -------\n"

# Print the entire chat content including steps
print(chat_content_str + chat_content)
#with open('importlib.txt', 'r') as file:
#    importfile_content = file.read()
#with open('printoutput.txt','r') as file:
#    printfile_content = file.read()
#print("Code Extracted from message :   start code ----- " + codemessage)
#print("Code Extracted from message :   start code ----- " + codemessage)
#from chat_code_utils import execute_extracted_code
#data = execute_extracted_code(codetoexecute)

