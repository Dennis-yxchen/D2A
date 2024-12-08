import collections
import concurrent.futures
import datetime
import random
import matplotlib.pyplot as plt
import sys
ROOT = r"Your path to the folder contain concordia and examples"
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)
from IPython import display
import sentence_transformers
from collections.abc import Callable, Sequence
from concordia.language_model import language_model
from concordia import components as generic_components
from concordia.agents import deprecated_agent as basic_agent
from concordia.components.agent import to_be_deprecated as components
from concordia.agents import deprecated_agent
from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import game_master as gm_components
from concordia.environment import game_master
from concordia.language_model.language_model import LanguageModel
from concordia.metrics import goal_achievement
from concordia.metrics import common_sense_morality
from concordia.metrics import opinion_of_others
from concordia.utils import measurements as measurements_lib
from concordia.language_model import gpt_model
from concordia.utils import html as html_lib
from concordia.utils import plotting
from NPC_agent.generic_support_agent import build_support_agent
from concordia.language_model import utils
import json
import os
from D2A_agent.ValueAgent import build_value_agent
from Baseline_agent.Baseline_ReAct import build_ReAct_agent
from Baseline_agent.Baseline_LLMob import build_LLMob_agent
from Baseline_agent.Baseline_BabyAGI import build_BabyAGI_agent
## setting start here
from concordia.typing.entity_component import EntityWithComponents
from value_components.init_value_info_social import construct_all_profile_dict
from value_components import value_comp
episode_length = 3
disable_language_model = True
test_agents = ['ReAct', 'LLMob', 'value', 'BabyAGI']
st_model = sentence_transformers.SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2')
embedder = lambda x: st_model.encode(x, show_progress_bar=False)
NUM_PLAYERS = 3

EXP_START_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

Use_Previous_profile = False
if Use_Previous_profile:
  previous_profile_file = os.path.join(r'examples\D2A\result_folder\outdoor_result\2024-12-08_23-12-34', '2024-12-08_23-12-43_ReAct.json')
  try:
    with open(previous_profile_file, 'r') as f:
      previous_profile = json.load(f)
  except:
    raise ValueError('The previous profile file is not found.')
else:
  previous_profile = None

current_file_path = os.path.dirname(os.path.abspath(__file__))
result_folder_name = 'result_folder'
current_folder_path = os.path.join(current_file_path, result_folder_name)
if not os.path.exists(current_folder_path):
  os.makedirs(current_folder_path)

subsub_folder = os.path.join(current_folder_path, 'outdoor_result')
if not os.path.exists(subsub_folder):
  os.makedirs(subsub_folder)

stored_target_folder = os.path.join(subsub_folder, EXP_START_TIME)
if not os.path.exists(stored_target_folder):
  os.makedirs(stored_target_folder)


api_type = 'Type of API'
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


importance_model = importance_function.AgentImportanceModel(model)
importance_model_gm = importance_function.ConstantImportanceModel()

SETUP_TIME = datetime.datetime(hour=8, year=2024, month=10, day=1)
START_TIME = datetime.datetime(hour=9, year=2024, month=10, day=2)


shared_memories = [
    'Central Park hosts the "Enjoy Your Life" party event.',
    'Food trucks offer a variety of cuisines at Central Park.', # hunger
    'Buffet stations provide diverse options at Central Park.',
    'Snack bars serve quick bites at Central Park.',
    'Beverage stands offer drinks at Central Park.', # thirst
    'Hydration stations offer water and electrolyte drinks.',
    'Juice bars serve freshly squeezed juices at Central Park.',
    'Chill-out zones with bean bags are available at Central Park.', # sleepy & passion
    'Quiet tents provide a resting space at Central Park.',
    'Coffee stalls offer freshly brewed coffee at Central Park.',
    'Seating areas with comfortable chairs are at Central Park.',
    'Shaded spots with tents are available at Central Park.',
    'Networking lounges facilitate social connections at Central Park.', # social connection
    'Interactive games are available for attendees at Central Park.',
    'Photo booths with fun props are set up at Central Park.',
    'A live music stage features bands and DJs at Central Park.', # joy
    'A dance floor with sound systems is set up at Central Park.',
    'Comedy shows entertain attendees at Central Park.',
    'An art corner supplies materials for painting or crafts.',
    'Workshops offer short sessions on cooking or photography.',
    'Discussion panels on various topics are held at Central Park.', # spiritual
    'Meditation zones offer guided sessions at Central Park.',
    'Nature walks provide reflection paths at Central Park.',
]

shared_context = model.sample_text(
    'Summarize the following passage in a concise and insightful fashion:\n'
    + '\n'.join(shared_memories)
    + '\n'
    + 'Summary:'
)


class FormativeMemoryFactoryWithoutBackground(formative_memories.FormativeMemoryFactory):
    def __init__(self, * ,
                 model:  language_model.LanguageModel,
                 shared_memories: Sequence[str] = (),
                 delimiter_symbol: str = '***',
                 blank_memory_factory_call: Callable[[], associative_memory.AssociativeMemory],
                 current_date: datetime.datetime | None = None):
        super().__init__(model=model,
                         shared_memories=shared_memories,
                         blank_memory_factory_call=blank_memory_factory_call,
                         delimiter_symbol=delimiter_symbol,
                         current_date=current_date)


    def make_memories(self, agent_config: formative_memories.AgentConfig) -> associative_memory.AssociativeMemory:
      mem = self._blank_memory_factory_call()
      for item in self._shared_memories:
        mem.add(item)

      context = agent_config.context
      if agent_config.goal:
        context += '\n' + agent_config.goal

      if context:
        context_items = context.split('\n')
        for item in context_items:
          if item:
            mem.add(item)

      if agent_config.specific_memories:
        specific_memories = agent_config.specific_memories.split('\n')
        for item in specific_memories:
          if item:
            mem.add(item)

      # add the specific desires
      if agent_config.extras.get("desires", False):
        desires = agent_config.extras["desires"].split('\n')
        for item in desires:
          if item:
            mem.add(item)
      return mem

## general setting end here

## agent setting start here

def make_random_big_five()->str:
  return str({
      'extraversion': random.randint(1, 10),
      'neuroticism': random.randint(1, 10),
      'openness': random.randint(1, 10),
      'conscientiousness': random.randint(1, 10),
      'agreeableness': random.randint(1, 10),
  })

def get_extras(name, is_main_character):
    if is_main_character:
        return {
            'specific_memories': [f'{name} is an attendee of the party.'],
            'main_character': is_main_character,
        }
    else:
        return {
            'specific_memories': [f'{name} is the staff.'],
            'main_character': is_main_character,
        }

wanted_desires = ['hunger',
                  'thirst',
                  'comfort',
                  'sleepiness',
                  'joyfulness',
                  'spiritual satisfaction',
                  'social connectivity',
                  'sense of control',
                  'recognition',
                  'sense of superiority']
# wanted_desires = [
#     'hunger',
#     'thirst',
#     'comfort',
#     'health',
#     'sleepiness',
#     'joyfulness',
#     'cleanliness',
#     'safeness',
#     'passion',
#     'spiritual satisfaction',
#     'social connectivity',
# ]

hidden_desires = ['thirst']
## agent setting end here



## sth that will not change start here

if previous_profile:
  visual_desires_dict, hidden_desires_dict, selected_desire_dict, all_desire_traits_dict, visual_desire_string = construct_all_profile_dict(
   wanted_desires = wanted_desires,
   hidden_desires = hidden_desires,
   predefined_desires = previous_profile
)
else:
  visual_desires_dict, hidden_desires_dict, selected_desire_dict, all_desire_traits_dict, visual_desire_string = construct_all_profile_dict(
   wanted_desires = wanted_desires,
   hidden_desires = hidden_desires
)

# def get_extras_for_specific_agent(name, is_main_character, desires):
#     if is_main_character and current_agent.lower() == 'value':
#         return {
#             'specific_memories': [f'{name} is an attendee of the party.'],
#             'main_character': is_main_character,
#             'desires': desires,
#         }
#     elif is_main_character:
#        return {
#             'specific_memories': [f'{name} is an attendee of the party.'],
#             'main_character': is_main_character,
#         }
#     else:
#         return {
#             'specific_memories': [f'{name} is the staff.'],
#             'main_character': is_main_character,
#         }

# player_configs = [
#     formative_memories.AgentConfig(
#         name='Alice',
#         gender='female',
#         goal='Alice wants to enjoy the party in the comfortable way, and also satisfy her desires.' if current_agent.lower() != 'value' else 'Alice wants to enjoy the party in the comfortable way',
#         context=shared_context+' Alice is a very socially active attendee.',
#         traits = make_random_big_five(),
#         extras=get_extras_for_specific_agent(name = 'Alice',
#                                              is_main_character=True,
#                                              desires=visual_desire_string.format(agent_name='Alice')),
#             ),
#     formative_memories.AgentConfig(
#         name='Bob',
#         gender='male',
#         goal="Bob hopes to meet the guests' needs.",
#         context=shared_context + " Bob is a very enthusiastic and outgoing staff member, and he stays in the networking lounge.",
#         traits = make_random_big_five(),
#         extras=get_extras('Bob', False)
#             ),
#     formative_memories.AgentConfig(
#         name='Charlie',
#         gender='male',
#         goal="Charlie wants to enjoy the party and makes more friends.",
#         context=shared_context + " Charlie is an attendee of the party, and he loves taking photos and networking with other people.",
#         traits = make_random_big_five(),
#         extras=get_extras('Charlie', False)
#       ),
# ]


if Use_Previous_profile:
  numerical_desire = previous_profile['initial_value']
else:
  numerical_desire = {
  desire_name : int(random.randint(0, 10))
    for desire_name in wanted_desires
    }

## sth that will not change end here

measurements = measurements_lib.Measurements()


def _get_current_agent(agent_name, config, mem, clock):
    if agent_name == 'LLMob':
      # directly use the desire profile (same as value agent)
      agent = build_LLMob_agent(config = config,
                                  context_dict=all_desire_traits_dict,
                                  predefined_setting=numerical_desire,
                                  selected_desire=wanted_desires,
                                  model = model,
                                  background_knowledge='\n'.join([shared_context]),
                                  profile=visual_desire_string,
                                  memory=mem,
                                  clock = clock,
                                  update_time_interval=None)
    elif agent_name == 'ReAct':
      # do not use any desire profile
      agent = build_ReAct_agent(config = config,
                                  context_dict=all_desire_traits_dict,
                                  predefined_setting=numerical_desire,
                                  selected_desire=wanted_desires,
                                  model = model,
                                  background_knowledge='\n'.join([shared_context]),
                                  memory=mem,
                                  clock = clock,
                                  update_time_interval=None)
    elif agent_name == 'value':
      # directly use the desire profile
      agent = build_value_agent(config = config,
                                  context_dict=all_desire_traits_dict,
                                  selected_desire=wanted_desires,
                                  predefined_setting=numerical_desire,
                                  background_knowledge='\n'.join([shared_context]),
                                  model = model,
                                  profile = visual_desire_string,
                                  memory=mem,
                                  clock = clock,
                                  update_time_interval=None)
    elif agent_name == 'BabyAGI':
       agent = build_BabyAGI_agent(config = config,
                                  context_dict=all_desire_traits_dict,
                                  selected_desire=wanted_desires,
                                  predefined_setting=numerical_desire,
                                  background_knowledge='\n'.join([shared_context]),
                                  model = model,
                                  memory=mem,
                                  profile=visual_desire_string,
                                  clock = clock,
                                  update_time_interval=None)
    else:
      raise ValueError('This function is meant for a supporting character '
                      'but it was called on a main character.')
    return agent

def build_memory(agent_config, blank_memory_factory):
    if agent_config.extras.get('main_character', False):
        formative_memory_factory = FormativeMemoryFactoryWithoutBackground(
            model=model,
            shared_memories=shared_memories,
            blank_memory_factory_call=blank_memory_factory.make_blank_memory,
        )
    else:
        formative_memory_factory = formative_memories.FormativeMemoryFactory(
            model=model,
            shared_memories=shared_memories,
            blank_memory_factory_call=blank_memory_factory.make_blank_memory,
        )
    return formative_memory_factory.make_memories(agent_config)



def build_players_list(blank_memory_factory: blank_memories.MemoryFactory,
                       clock: game_clock.MultiIntervalClock, current_agent: str):

  def get_extras_for_specific_agent(name, is_main_character, desires):
    if is_main_character and current_agent.lower() == 'value':
        return {
            'specific_memories': [f'{name} is an attendee of the party.'],
            'main_character': is_main_character,
            'desires': desires,
        }
    elif is_main_character:
        return {
            'specific_memories': [f'{name} is an attendee of the party.'],
            'main_character': is_main_character,
        }
    else:
        return {
            'specific_memories': [f'{name} is the staff.'],
            'main_character': is_main_character,
        }

  player_configs = [
      formative_memories.AgentConfig(
          name='Alice',
          gender='female',
          goal='Alice wants to enjoy the party in the comfortable way, and also satisfy her desires.' if current_agent.lower() != 'value' else 'Alice wants to enjoy the party in the comfortable way',
          context=shared_context+' Alice is a very socially active attendee.',
          traits = make_random_big_five(),
          extras=get_extras_for_specific_agent(name = 'Alice',
                                              is_main_character=True,
                                              desires=visual_desire_string.format(agent_name='Alice')),
              ),
      formative_memories.AgentConfig(
          name='Bob',
          gender='male',
          goal="Bob hopes to meet the guests' needs.",
          context=shared_context + " Bob is a very enthusiastic and outgoing staff member, and he stays in the networking lounge.",
          traits = make_random_big_five(),
          extras=get_extras('Bob', False)
              ),
      formative_memories.AgentConfig(
          name='Charlie',
          gender='male',
          goal="Charlie wants to enjoy the party and makes more friends.",
          context=shared_context + " Charlie is an attendee of the party, and he loves taking photos and networking with other people.",
          traits = make_random_big_five(),
          extras=get_extras('Charlie', False)
        ),
    ]
  player_configs = player_configs[:NUM_PLAYERS]
  player_names = [player.name for player in player_configs][:NUM_PLAYERS]

  players = []
  memories = {}

  main_character = []
  supported_characters = []

  main_character = [player for player in player_configs if player.extras.get('main_character', False)]
  supported_characters = [player for player in player_configs if not player.extras.get('main_character', False)]

  for config in main_character:
      mem = build_memory(config, blank_memory_factory)
      agent  = _get_current_agent(current_agent, config, mem, clock)
      players.append(agent)
      memories[agent.name] = mem
      for extra_memory in config.extras['specific_memories']:
        mem.add(f'{extra_memory}', tags=['initial_player_specific_memory'])

  for config in supported_characters:
      mem = build_memory(config, blank_memory_factory)
      agent = build_support_agent(config = config, model = model, memory=mem, clock = clock, update_time_interval=None)
      players.append(agent)
      memories[agent.name] = mem
      for extra_memory in config.extras.get('specific_memories', []):
        mem.add(f'{extra_memory}', tags=['initial_specific_memory'])

  # citizen_names = [player.name for player in players]
  player_names = [player.name for player in players]
  return players, memories, player_names, main_character, supported_characters, player_configs


def build_game_master(main_character, players, player_names, memories, clock, player_configs, blank_memory_factory):

  game_master_memory = associative_memory.AssociativeMemory(
      embedder, importance_model_gm.importance, clock=clock.now)

  for config in main_character:
      for extra_memory in config.extras['specific_memories']:
        game_master_memory.add(f'{extra_memory}', tags=['initial_player_specific_memory'])

  facts_on_village = generic_components.constant.ConstantComponent(
      ' '.join(shared_memories), 'General knowledge of Party event.'
  )
  player_status = gm_components.player_status.PlayerStatus(
      clock.now, model, game_master_memory, player_names
  )

  relevant_events = gm_components.relevant_events.RelevantEvents(
      clock.now, model, game_master_memory
  )
  time_display = gm_components.time_display.TimeDisplay(clock)

  direct_effect_externality = gm_components.direct_effect.DirectEffect(
      players,
      model=model,
      memory=game_master_memory,
      clock_now=clock.now,
      verbose=False,
      components=[player_status],
  )

  convo_externality = None
#   convo_externality = gm_components.conversation.Conversation(
#     players,
#     model,
#     clock=clock,
#     memory=game_master_memory,
#     burner_memory_factory=blank_memory_factory,
#     components=[player_status],
#     cap_nonplayer_characters=0,
#     shared_context=shared_context,
#     max_conversation_length = 2,
#     verbose=False,
# )

  env = game_master.GameMaster(
      model=model,
      memory=game_master_memory,
      clock=clock,
      players=players,
      components=[
          facts_on_village,
          player_status,
          direct_effect_externality,
          relevant_events,
          time_display,
          # convo_externality,
      ],
      randomise_initiative=True,
      player_observes_event=False,
      verbose=True,
  )
  clock.set(START_TIME)

  for index, player in enumerate(players):
    gender = player_configs[index].gender
    how_to_call = 'she' if gender == 'female' else 'he'
    if player.name == 'Alice' or player.name == 'Charlie':
      player.observe(
          f'{player.name} is at the party, {how_to_call} has just arrived.'
      )
      game_master_memory.add(f'{player.name} is at the party.')
    else:
      player.observe(
          f'{player.name} is a staff member at the party, {how_to_call} is in the networking lounge.'
      )
      game_master_memory.add(f'{player.name} is at the party. People can find {player.name} in the networking lounge.')
  return env, game_master_memory, relevant_events, player_status, direct_effect_externality, convo_externality

def start_simulation(current_test_agent: str):
  clock = game_clock.MultiIntervalClock(
  start=SETUP_TIME,
  step_sizes=[datetime.timedelta(minutes=20), datetime.timedelta(seconds=10)]
  )

  blank_memory_factory = blank_memories.MemoryFactory(
    model=model,
    embedder=embedder,
    importance=importance_model.importance,
    clock_now=clock.now,
    )

  players, memories, player_names, main_character, supported_characters, player_configs = build_players_list(blank_memory_factory,
                                                                                             clock,
                                                                                             current_test_agent)
  env, game_master_memory, relevant_events, player_status, direct_effect_externality, convo_externality = build_game_master(main_character,
                                                                         players,
                                                                         player_names,
                                                                         memories,
                                                                         clock,
                                                                         player_configs,
                                                                         blank_memory_factory)
  for _ in range(episode_length):
    env.step()

  all_gm_memories = env._memory.retrieve_recent(k=10000, add_time=True)

  detailed_story = '\n'.join(all_gm_memories)
  print('len(detailed_story): ', len(detailed_story))
  # print(detailed_story)

  episode_summary = model.sample_text(
      f'Sequence of events:\n{detailed_story}'+
      '\nNarratively summarize the above temporally ordered ' +
      'sequence of events. Write it as a news report. Summary:\n',
      max_tokens=3500, terminators=())
  print(episode_summary)

  player_logs = []
  player_log_names = []
  for player in players:
    name = player.name
    detailed_story = '\n'.join(memories[player.name].retrieve_recent(
        k=1000, add_time=True))
    summary = ''
    summary = model.sample_text(
        f'Sequence of events that happened to {name}:\n{detailed_story}'
        '\nWrite a short story that summarises these events.\n'
        ,
        max_tokens=3500, terminators=())

    all_player_mem = memories[player.name].retrieve_recent(k=1000, add_time=True)
    all_player_mem = ['Summary:', summary, 'Memories:'] + all_player_mem
    player_html = html_lib.PythonObjectToHTMLConverter(all_player_mem).convert()
    player_logs.append(player_html)
    player_log_names.append(f'{name}')
  if convo_externality:
    history_sources = [env, direct_effect_externality, relevant_events, player_status, convo_externality]
  else:
    history_sources = [env, direct_effect_externality, relevant_events, player_status]
  histories_html = [html_lib.PythonObjectToHTMLConverter(history.get_history()).convert() for history in history_sources]
  histories_names = [history.name for history in history_sources]

  gm_mem_html = html_lib.PythonObjectToHTMLConverter(all_gm_memories).convert()

  tabbed_html = html_lib.combine_html_pages(
      histories_html + [gm_mem_html] + player_logs,
      histories_names + ['GM'] + player_log_names,
      summary=episode_summary,
      title='Riverbend elections experiment',
  )

  tabbed_html = html_lib.finalise_html(tabbed_html)

  # @title Save the output to a file
  current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


  output_file = f'{current_time}_{current_test_agent}.html'  # @param {type: 'string'}
  output_file = os.path.join(stored_target_folder, output_file)

  try:
    with open(output_file, 'w') as f:
      f.write(tabbed_html)
  except:
    try:
      with open(f'{output_file}_1.html', 'w', encoding='utf-8') as f:
        f.write(tabbed_html)
    except:
        tabbed_html = tabbed_html.encode('utf-8', 'replace').decode('utf-8')
        with open(f'{output_file}_2.html', 'w', encoding='utf-8') as f:
          f.write(tabbed_html)

  def track_each_action_delta_change(action_sequences:list[str], value_tracker: value_comp.ValueTracker):
    individual_delta_tracker = value_tracker.get_individual_delta_tracker()
    previous_one_delta = None
    change_in_desire = dict()
    for index, action in enumerate(action_sequences):
      if index == 0: # skip the first action
        previous_one_delta = individual_delta_tracker.get(index, None) # get the first delta
        continue
      current_one_delta = individual_delta_tracker.get(index, None) # get the current delta
      if previous_one_delta and current_one_delta: # if both are not None
        # if delta smaller than previous one, record it
        delta_change = [k for k in current_one_delta.keys() if current_one_delta[k] < previous_one_delta[k]]
        print(f"Action: {action}, Delta Change: {delta_change}")
      change_in_desire[index] = delta_change




  def summarise_value(player: EntityWithComponents, other_players: list = None):
      value_tracker = player.get_component('ValueTracker', type_=value_comp.ValueTracker)
      action_seq = value_tracker.get_action_sequence()
      json_result = dict()
      json_result['save_timestamp'] = current_time
      json_result['start_time'] = str(EXP_START_TIME)
      json_result['action_sequence'] = [
        {'timestamp': str(each_act_dict['timestamp']),
          'action': each_act_dict['action'].strip()}
          for each_act_dict in action_seq]
      json_result['step'] = episode_length

      whole_delta_tracker = value_tracker.get_whole_delta_tracker()
      print(f"whole_delta_tracker: {whole_delta_tracker}")
      json_result['whole_delta'] = {int(k): float(v) for k, v in whole_delta_tracker.items()}

      individual_delta_tracker = value_tracker.get_individual_delta_tracker()
      print(f"individual_delta_tracker: {individual_delta_tracker}")
      json_result['individual_delta'] = {
          int(k_step): {delta: float(value) for delta, value in delta_value_pair.items()}
          for k_step, delta_value_pair in individual_delta_tracker.items()
      }

      individual_desire_tracker = value_tracker.get_individual_desire_tracker()
      print(f"individual_desire_tracker: {individual_desire_tracker}")
      json_result['individual_desire'] = {
          int(k_step): {desire: int(value) for desire, value in desire_value_pair.items()}
          for k_step, desire_value_pair in individual_desire_tracker.items()
      }

      individual_qualitative_desire_tracker = value_tracker.get_individual_qualitative_desire_tracker()
      print(f"individual_qualitative_desire_tracker: {individual_qualitative_desire_tracker}")
      json_result['individual_qualitative_desire'] = {int(k): v for k,v in individual_qualitative_desire_tracker.items()}

      expected_values = value_tracker.get_expected_value_dict()
      print(f"expected_values: {expected_values}")
      json_result['expected_values'] = {desire_name: float(exp_value) for desire_name, exp_value in expected_values.items()}

      profile = visual_desire_string
      json_result['profile'] = profile

      initial_value = {k: float(v) for k, v in numerical_desire.items()}
      json_result['initial_value'] = initial_value

      sampled_background = shared_context
      json_result['sampled_background'] = sampled_background

      alice = [i for i in player_configs if i.name == 'Alice'][0]
      alice_dict = alice.to_dict()
      json_result['Alice_setting'] = alice_dict

      json_result['wanted_desires'] = wanted_desires
      json_result['hidden_desires'] = hidden_desires
      json_result['visual_desires_dict'] = visual_desires_dict
      json_result['hidden_desires_dict'] = hidden_desires_dict
      json_result['selected_desire_dict'] = selected_desire_dict
      json_result['all_desire_traits_dict'] = all_desire_traits_dict
      json_result['visual_desire_string'] = visual_desire_string
      json_result['model_name'] = model_name

      if other_players:
          other_players_dict = {player.name: player.to_dict() for player in other_players}
          json_result['other_players'] = other_players_dict

      if Use_Previous_profile:
        json_result['previous_profile_file'] = previous_profile_file


      return json_result


  value_result = f'{current_time}_{current_test_agent}.json'
  value_result_file = os.path.join(stored_target_folder, value_result)

  alice = [i for i in players if i.name == 'Alice'][0]
  other_players = [i for i in player_configs if i.name != 'Alice']

  try:
    with open(value_result_file, 'w') as f:
      json.dump(summarise_value(alice, other_players), f, indent=4)
  except:
    with open('filename.json', 'w', encoding='utf-8') as f:
      json.dump(summarise_value(alice, other_players), f, indent=4)


if __name__ == '__main__':
  # for agent in ['LLMob', 'ReAct', 'value', 'BabyAGI']:
  for agent in test_agents:
    start_simulation(agent)
    print(f"Finish simulation for {agent}")
