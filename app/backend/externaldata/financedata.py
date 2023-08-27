import re
import numpy as np

def extract_code_from_message(message):
    # Extract content between triple backticks
    orig_code_match = re.search(r'```(.*?)```', message, re.DOTALL)
    if orig_code_match:
        code_match = orig_code_match.group(1)
        code_with_newlines = code_match.replace("\\n", "\n")
        return code_with_newlines.strip()
    return None

def append_importlibs(source_file, target_file):
    try:
        with open(source_file, 'r') as source:
            source_contents = source.read()
        
        with open(target_file, 'a') as target:
            target.write(source_contents)
            
        print(f"Contents from {source_file} appended to {target_file} successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

def extract_and_align_code(text):
    # Extract code wrapped in triple backticks
    code_block = re.search(r'```(.*?)```', text, re.DOTALL)
    
    if not code_block:
        return "No code block found!"
    
    code = code_block.group(1).splitlines()
    min_indent = min([len(re.match(r'^\s*', line).group(0)) for line in code if line.strip()])
    aligned_code = '\n'.join([line[min_indent:] for line in code])
    
    return aligned_code

def execute_extracted_code(code):
    # WARNING: Be careful with exec! Only execute code you trust.
    # Create a local namespace for executing the code
    local_ns = {}
    exec(code, {}, local_ns)
    return local_ns.get('data', None)

# Program start
with open('samplecode.txt', 'r') as file:
    file_content = file.read()

print("Original message : ---- " + file_content)

code_from_msg = extract_code_from_message(file_content)

with open('importlib.txt', 'r') as file:
    importfile_content = file.read()

codetoexecute = importfile_content + code_from_msg


print("Code Extracted from message :   start code ----- " + code_from_msg)

print("Code Extracted from message :   start code ----- " + codetoexecute)


data = execute_extracted_code(code_from_msg)
print(data)
