import os
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")

if api_key is None:
    raise RuntimeError("no api found")

client = genai.Client(api_key=api_key)
parser = argparse.ArgumentParser(description="AI Agent")
parser.add_argument("user_prompt", type=str, help="User prompt")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
args = parser.parse_args()
messages = [types.Content(role="user", parts=[types.Part(text=args.user_prompt)])]




response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents= messages
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
    print(response.text)

else:
    print(response.text)
    
