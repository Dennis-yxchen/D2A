import sentence_transformers
import sys
from concordia.language_model import utils
import json
import os

ROOT = r"Your path to the folder contain concordia and examples"
episode_length = 3
disable_language_model = True
st_model = sentence_transformers.SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2')
embedder = lambda x: st_model.encode(x, show_progress_bar=False)

tested_agents = ['ReAct']

Use_Previous_profile = True
previous_profile_file = None
previous_profile = None

if Use_Previous_profile:
  previous_profile_file = os.path.join(r'examples\D2A\result_folder\outdoor_result\Your folder name', 'Your previous profile name.json')
  try:
    with open(previous_profile_file, 'r') as f:
      previous_profile = json.load(f)
  except:
    raise ValueError('The previous profile file is not found.')
else:
  previous_profile = None

api_type = 'together_ai'
# model_name='google/gemma-2-9b-it'
model_name = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
api_key='Your_API_Key'
device = 'cpu'
model = utils.language_model_setup(
    api_type=api_type,
    model_name=model_name,
    api_key=api_key,
    disable_language_model=disable_language_model,
)

wanted_desires = [
  'hunger',
  'thirst',
  'comfort',
  'sleepiness',
  'joyfulness',
  'spiritual satisfaction',
  'social connectivity',
  'sense of control',
  'recognition',
  'sense of superiority'
  ]

hidden_desires = ['thirst']

current_file_path = os.path.dirname(os.path.abspath(__file__))
result_folder_name = 'result_folder'
current_folder_path = os.path.join(current_file_path, result_folder_name)
if not os.path.exists(current_folder_path):
  os.makedirs(current_folder_path)

subsub_folder = os.path.join(current_folder_path, 'outdoor_result')
if not os.path.exists(subsub_folder):
  os.makedirs(subsub_folder)
