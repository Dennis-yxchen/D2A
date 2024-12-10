"""A factory implementing the three key questions agent as an entity."""

from collections.abc import Callable
import datetime
import json
from concordia.typing import entity as entity_lib
import types
from typing import Mapping
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity_component
from concordia.utils import measurements as measurements_lib
import numpy as np
from collections.abc import Sequence
from concordia.typing import entity_component
from concordia.typing import logging
from concordia.components.agent import observation
from concordia.document import interactive_document
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
IMPORT_AGENT_BASE_DIR = 'examples.D2A.value_components'
init_value_info_social = importlib.import_module(
    f'{IMPORT_AGENT_BASE_DIR}.init_value_info_social')
value_comp = importlib.import_module(f'{IMPORT_AGENT_BASE_DIR}.value_comp')

from .BabyAGI_ActComp import BabyAGIActComponent
import NullObservation

class BackgroundKnowledge(agent_components.constant.Constant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


def get_profile_summary(model, profile):
  prompt = interactive_document.InteractiveDocument(model)
  profile_summary = prompt.open_question(
      f"""
      Based on a brief description of a person's character and personal traits (a few complete statements with degree adverbs, such as: "Alice is somewhat gluttonous. Alice is very materialistic."), write a paragraph in the first-person perspective describing the person's living habits in their room. Refer to the following aspects:

      1.Health Habits:
      Regular routine
      Balanced diet
      Regular exercise
      Adequate sleep
      Maintain good personal hygiene
      Regular medical check-ups
      Moderate drinking
      Non-smoking

      2.Bad Habits:
      Staying up late
      Overeating
      Sedentary lifestyle
      Excessive drinking
      Smoking
      Substance abuse
      Lack of exercise
      Poor sitting or standing posture

      3.Work Habits:
      Punctuality
      Efficiency
      Organized
      Focused
      Teamwork
      Self-motivated
      Continuous learning
      Good communication skills

      4.Social Habits:
      Friendly
      Sociable
      Helpful
      Respectful to others
      Good listener
      Honest
      Polite
      Maintains personal space

      5.Personal Habits:
      Frugal
      Tidy
      Organized
      Enjoys solitude
      Prefers quiet environments
      Adventurous
      Likes trying new things
      Creative

      6.Leisure Habits:
      Reading
      Traveling
      Watching movies
      Listening to music
      Cooking
      Gardening
      Handicrafts
      Sports

      7.Sleep Habits:
      Early to bed and early to rise
      Needs naps
      Reads before sleeping
      Uses an alarm clock
      High requirements for sleep environment (e.g., quiet, dark)

      8.Eating Habits:
      Vegetarian
      Enjoys fast food
      Likes trying different foods
      Eats at regular times and portions
      Prefers home cooking
      Likes dining out

      Please based on these aspects and in combination with the person's character and personal traits, infer and creatively describe their living habits in their room from a first-person perspective.
      The person's <profile>: {profile}, do not include the words in <profile> in the description.
      """,
      max_tokens=1200,
      terminators=(),
      )
  return profile_summary


def get_likely_to_do_summary(model, profile, environment):
  prompt = interactive_document.InteractiveDocument(model)
  profile_summary = prompt.open_question(
      f"""
      Given the environment:
      {environment}
      Please based on the environment and in combination with the person's character and personal traits, infer and creatively describe their habits.
      The person's <profile>: {profile}, do not include the words in <profile> in the description.
      """,
      max_tokens=1200,
      terminators=(),
      )
  return profile_summary


def build_BabyAGI_agent(*,
      config: formative_memories.AgentConfig,
      context_dict: Mapping[str, str], # contain value related context
      predefined_setting: Mapping[str, Mapping[str, str]], # contain value related setting
      selected_desire: Sequence[str], # contain value related desire
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      background_knowledge:str,
      profile: str,
      clock: game_clock.MultiIntervalClock,
      update_time_interval: datetime.timedelta,
      additional_components: Mapping[
          entity_component.ComponentName,
          entity_component.ContextComponent,
      ] = types.MappingProxyType({}),
  ) -> entity_agent_with_logging.EntityAgentWithLogging:

    del update_time_interval
    if not config.extras.get('main_character', False):
        raise ValueError('This function is meant for a main character '
                        'but it was called on a supporting character.')
    agent_name = config.name
    raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)

    measurements = measurements_lib.Measurements()
    instructions = agent_components.instructions.Instructions(
      agent_name=agent_name,
      logging_channel=measurements.get_channel('Instructions').on_next,
    )

    time_display = agent_components.report_function.ReportFunction(
      function=clock.current_time_interval_str,
      pre_act_key='\nCurrent time',
      logging_channel=measurements.get_channel('TimeDisplay').on_next,
    )

    background_knowledge_comp = BackgroundKnowledge(
        state=background_knowledge,
        pre_act_key='\nbackground knowledge',
        logging_channel=measurements.get_channel('Background Knowledge').on_next,
    )

    observation_label = '\nCurrent observation'
    observation = agent_components.observation.Observation(
        clock_now=clock.now,
        timeframe=clock.get_step_size(),
        pre_act_key=observation_label,
        logging_channel=measurements.get_channel('Observation').on_next,
    )


    ## Value Components
    general_pre_act_label = f"\n{agent_name}" + "'s current feeling of {desire_name} is"
    ### init the information to be used in the value component
    detailed_values_dict, expected_values = init_value_info_social.preprocess_value_information(context_dict, predefined_setting, selected_desires=selected_desire)
    print(f"detailed_values_dict: {detailed_values_dict}")
    print(f"expected_values: {expected_values}")
    print(f"len of detailed_values_dict: {len(detailed_values_dict)}")
    print(f"len of expected_values: {len(expected_values)}")
    all_desire_components = init_value_info_social.get_all_desire_components_without_PreAct(model, general_pre_act_label, observation, clock, measurements, detailed_values_dict, expected_values,wanted_desires = selected_desire)


    target_tracking_desire_component = dict()
    for desire_name, desire_component in all_desire_components.items():
        target_tracking_desire_component[_get_class_name(desire_component)] = desire_component
    value_tracker = value_comp.ValueTracker(
        pre_act_key='',
        desire_components=target_tracking_desire_component,
        logging_channel=measurements.get_channel('ValueTracker').on_next,
        init_value = predefined_setting,
        expected_value_dict=expected_values,
        clock_now = clock.now,
    )

    null_observation = NullObservation.NULLObservation(
        clock_now=clock.now,
        timeframe=None,
        memory_component_name=agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
        logging_channel=measurements.get_channel('NullObservation').on_next,
        pre_act_key='',
    )

    entity_components = (
        # Components that provide pre_act context.
        instructions,
        time_display,
        background_knowledge_comp,
        observation,

        # Components that do not provide pre_act context.
        value_tracker,
        null_observation
    )

    entity_components += tuple(all_desire_components.values())

    components_of_agent = {_get_class_name(component): component
                            for component in entity_components}
    components_of_agent[
        agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME] = (
            agent_components.memory_component.MemoryComponent(raw_memory))
    component_order = list(components_of_agent.keys())

    daily_routine = get_likely_to_do_summary(model, profile, environment = background_knowledge)
    act_component = BabyAGIActComponent(
        model=model,
        clock=clock,
        profile = daily_routine,
        background_component_name = _get_class_name(background_knowledge_comp),
        observation_component_name = _get_class_name(observation),
        memory_component_name = agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
        component_order=component_order,
        logging_channel=measurements.get_channel('ActComponent').on_next,
    )

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=agent_name,
        act_component=act_component,
        context_components=components_of_agent,
        component_logging=measurements,
    )

    return agent
