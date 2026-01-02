import os
import argparse
from prompts import system_prompt
from call_function import (schema_get_files_info,
schema_get_file_content, 
schema_run_python_file, 
schema_write_file,
call_function,)
from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")

if api_key is None:
    raise RuntimeError("no api found")

available_functions = types.Tool(
    function_declarations=[schema_get_files_info,
                           schema_get_file_content,
                           schema_run_python_file,
                           schema_write_file],
)

client = genai.Client(api_key=api_key)
parser = argparse.ArgumentParser(description="AI Agent")
parser.add_argument("user_prompt", type=str, help="User prompt")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
args = parser.parse_args()
messages = [types.Content(role="user", parts=[types.Part(text=args.user_prompt)])]
model_name = "gemini-2.5-flash"
function_results = []


response = client.models.generate_content(
    model=model_name,
    contents=messages,
    config=types.GenerateContentConfig(
    tools=[available_functions], system_instruction=system_prompt
)
)



meta_data = response.usage_metadata

if meta_data is None:
    raise RuntimeError("metadata not found")

prompt_token_count = meta_data.prompt_token_count
candidates_token_count = meta_data.candidates_token_count

if args.verbose:
    print(f"User prompt: {args.user_prompt}")
    print(f"Prompt tokens: {prompt_token_count}")
    print(f"Response tokens: {candidates_token_count}")


if response.function_calls:
    for function_call in response.function_calls:
        function_call_result = call_function(function_call, verbose=args.verbose)
        
        if function_call_result.parts is None:
            raise Exception("Error: Function_call is emtpy")
        first_part = function_call_result.parts[0]
        function_response = first_part.function_response
        
        if function_response is None:
            raise Exception("Error: Function_response is empty.")
        if function_response.response is None:
            raise Exception("Error: Response is empty.")
        function_results.append(first_part)
        
        if args.verbose:
            print(f"-> {function_response.response}")

else:
       print(response.text)
    
