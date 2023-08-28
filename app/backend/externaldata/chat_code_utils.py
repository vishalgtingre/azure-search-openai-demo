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