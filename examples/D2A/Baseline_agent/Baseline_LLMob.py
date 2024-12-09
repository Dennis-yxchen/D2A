"""A factory implementing the three key questions agent as an entity."""

import types
from collections.abc import Callable, Mapping
import datetime
import json
from concordia.typing import entity as entity_lib

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
from concordia.components.agent.constant import DEFAULT_PRE_ACT_KEY
DEFAULT_PRE_ACT_KEY = 'Act'
from concordia.components.agent import observation
from concordia.document import interactive_document
import importlib
IMPORT_AGENT_BASE_DIR = 'examples.D2A.value_components'
init_value_info_social = importlib.import_module(
    f'{IMPORT_AGENT_BASE_DIR}.init_value_info_social')
value_comp = importlib.import_module(f'{IMPORT_AGENT_BASE_DIR}.value_comp')
from concordia.components.agent import memory_component

# from init_value_info import preprocess_value_information, get_all_desire_components
# from init_value_info_social import preprocess_value_information, get_all_desire_components_without_PreAct
from .LLMob_ActComp import LLMobActComponent
import NullObservation

class ConstantProfile(agent_components.constant.Constant):
  def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class BackgroundKnowledge(agent_components.constant.Constant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

DEFAULT_PRE_ACT_KEY = 'Plan'
_ASSOCIATIVE_RETRIEVAL = legacy_associative_memory.RetrieveAssociative()


class LLMobPlan(agent_components.action_spec_ignored.ActionSpecIgnored):
  """Component representing the agent's plan."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      observation_component_name: str,
      profile_component_name: str,
      background_knowledge_component_name: str,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      clock_now: Callable[[], datetime.datetime] | None = None,
      goal_component_name: str | None = None,
      num_memories_to_retrieve: int = 10,
      horizon: str = 'the rest of the day',
      pre_act_key: str = DEFAULT_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initialize a component to represent the agent's plan.

    Args:
      model: a language model
      observation_component_name: The name of the observation component from
        which to retrieve obervations.
      memory_component_name: The name of the memory component from which to
        retrieve memories
      components: components to build the context of planning. This is a mapping
        of the component name to a label to use in the prompt.
      clock_now: time callback to use for the state.
      goal_component_name: index into `components` to use to represent the goal
        of planning
      num_memories_to_retrieve: how many memories to retrieve as conditioning
        for the planning chain of thought
      horizon: string describing how long the plan should last
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: channel to use for debug logging.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._observation_component_name = observation_component_name
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._clock_now = clock_now
    self._goal_component_name = goal_component_name
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._horizon = horizon

    self._current_plan = ''

    self._logging_channel = logging_channel

    self._profile_component_name = profile_component_name
    self._background_knowledge_component_name = background_knowledge_component_name
    self._profile_string = ''
    self._background_knowledge_string = ''
    self._likely_to_do = ''

  def _get_profile(self) -> str:
    if self._profile_string:
      return self._profile_string
    self._profile_string = self.get_named_component_pre_act_value(
        self._profile_component_name
        )
    return self._profile_string

  def _get_background_knowledge(self) -> str:
    if self._background_knowledge_string:
      return self._background_knowledge_string
    self._background_knowledge_string = self.get_named_component_pre_act_value(
        self._background_knowledge_component_name
        )
    return self._background_knowledge_string

  def _get_likely_to_do(self) -> str:
    if self._likely_to_do:
      return self._likely_to_do
    prompt = interactive_document.InteractiveDocument(self._model)
    agent_name = self.get_entity().name
    profile = self._get_profile()
    background = self._get_background_knowledge()
    likely_to_do_question = f"""
      Context: {agent_name} lives in the given environment and {agent_name} is a person with the following profile:
      {profile}
      Environment: {background}
      Instructions: Reflecting on the context given, I would like you to suggest some actions that you would likely take in this environment.
      Your description should be coherent, utilizing conclusive language to form a well-structured paragraph.
      """

    prefix = f"{agent_name} is likely to: \n"

    self._likely_to_do = prompt.open_question(
        likely_to_do_question,
        answer_prefix=prefix,
        max_tokens=1200,
        terminators=(),
    )
    return self._likely_to_do

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name

    # get the observation from environment
    observation_component = self.get_entity().get_component(
        self._observation_component_name,
        type_=observation.Observation)
    latest_observations = observation_component.get_pre_act_value()

    # get goal
    goal = None
    if self._goal_component_name:
      goal_component = self.get_entity().get_component(
          self._goal_component_name,
          type_=agent_components.action_spec_ignored.ActionSpecIgnored)
      goal = goal_component.get_pre_act_value()

    # likely to do
    likely_to_do = self._get_likely_to_do()
    # background knowledge
    background_knowledge = self._get_background_knowledge()

    # ask the model to generate the motivation behind the action
    motivation_question = (
       f"Context: Act as {agent_name} in the given environment and describe the motivation for {agent_name}'s activities.\n"
       f"Environment: \n{background_knowledge}\n"
       f"Activities: Today, {agent_name} has the following activities: \n{latest_observations}\n"
       f"Goal: \n{goal}\n"
       f"Instructions: Describe in one sentence {agent_name}'s future motivation today after these activities. There are some activities that {agent_name} is likely to do: {likely_to_do}."
       f"Highlight any personal interests and needs."
    )

    prefix = f"{agent_name}'s motivation is: \n"
    prompt = interactive_document.InteractiveDocument(self._model)
    motivation = prompt.open_question(
        motivation_question,
        max_tokens=1200,
        answer_prefix=prefix,
        terminators=(),
    )

    in_context_example = (
        ' Please format the plan like in this example: [21:00 - 22:00] watch TV'
    )
    motivation_prompt_string = prompt.view().text()
    prompt = prompt.new()
    prompt.statement(f'Motivation:\n{motivation}')
    prompt.statement(f'Likely to do:\n{likely_to_do}')
    if goal is not None:
      prompt.statement(f'Goal: {goal}')

    prompt.statement(f'Current plan: {self._current_plan}')
    prompt.statement(f'Current situation: {latest_observations}')
    time_now = self._clock_now().strftime('[%d %b %Y %H:%M:%S]')
    prompt.statement(f'The current time is: {time_now}\n')
    should_replan = prompt.yes_no_question(
        f'Given the above, should {agent_name} change their current '
        'plan? Please answer in the format of (a) or (b).'
    )
    if should_replan or not self._current_plan:
      # Replan on the first turn and whenever the LLM suggests the agent should.
      goal_mention = '.'
      if self._goal_component_name:
        goal_mention = ', keep in mind the goal.'
      self._current_plan = prompt.open_question(
          f"Write {agent_name}'s plan for {self._horizon}."
          ' Provide a detailed schedule'
          + goal_mention
          + in_context_example,
          max_tokens=1200,
          terminators=(),
      )

    result = self._current_plan

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
        'motivation prompt': motivation_prompt_string,
        'Chain of thought of plan': prompt.view().text().splitlines(),
    })

    return result

  def get_state(self) -> entity_component.ComponentState:
    """Converts the component to JSON data."""
    with self._lock:
      return {
          'current_plan': self._current_plan,
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the component state from JSON data."""
    with self._lock:
      self._current_plan = state['current_plan']

##
def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__

def build_LLMob_agent(
    *,
    config: formative_memories.AgentConfig,
    selected_desire: Sequence[str], # contain the name of the desire
    context_dict: Mapping[str, str], # contain value related context
    predefined_setting: Mapping[str, Mapping[str, str]], # contain value related setting
    model: language_model.LanguageModel,
    background_knowledge: str,
    profile:str ,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta | None = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an agent.

    Args:
        config: The agent config to use.
        model: The language model to use.
        memory: The agent's memory object.
        clock: The clock to use.
        update_time_interval: Agent calls update every time this interval passes.

    Returns:
        An agent.
    """

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


    observation_label = '\nObservation'
    observation = agent_components.observation.Observation(
        clock_now=clock.now,
        timeframe=clock.get_step_size(),
        pre_act_key=observation_label,
        logging_channel=measurements.get_channel('Observation').on_next,
    )

    profile_label = '\nProfile'
    profile_comp = ConstantProfile(
        state=profile.format(agent_name=agent_name),
        pre_act_key=profile_label,
        logging_channel=measurements.get_channel(profile_label).on_next,
    )

    # background knowledge
    background_knowledge = BackgroundKnowledge(
        state=background_knowledge,
        pre_act_key='\nbackground knowledge',
        logging_channel=measurements.get_channel('Background Knowledge').on_next,
    )

    time_display = agent_components.report_function.ReportFunction(
        function=clock.current_time_interval_str,
        pre_act_key='\nCurrent time',
        logging_channel=measurements.get_channel('TimeDisplay').on_next,
    )



    if config.goal:
        goal_label = '\nGoal'
        overarching_goal = agent_components.constant.Constant(
            state=config.goal,
            pre_act_key=goal_label,
            logging_channel=measurements.get_channel(goal_label).on_next)
    else:
        goal_label = None
        overarching_goal = None

    LLMobPlan_component = LLMobPlan(
        model=model,
        observation_component_name=_get_class_name(observation),
        profile_component_name=_get_class_name(profile_comp),
        background_knowledge_component_name=_get_class_name(background_knowledge),
        memory_component_name=agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
        components={},
        clock_now=clock.now,
        goal_component_name= goal_label,
        num_memories_to_retrieve=10,
        horizon='the rest of the day',
        pre_act_key=DEFAULT_PRE_ACT_KEY,
        logging_channel=measurements.get_channel('LLMobPlan').on_next,
    )
    
    ## Value Components
    general_pre_act_label = f"\n{agent_name}" + "'s current feeling of {desire_name} is"
    ### init the information to be used in the value component
    detailed_values_dict, expected_values = init_value_info_social.preprocess_value_information(context_dict, predefined_setting, selected_desire)
    print(f"detailed_values_dict: {detailed_values_dict}")
    print(f"expected_values: {expected_values}")
    print(f"len of detailed_values_dict: {len(detailed_values_dict)}")
    print(f"len of expected_values: {len(expected_values)}")
    all_desire_components = init_value_info_social.get_all_desire_components_without_PreAct(model, general_pre_act_label, observation, clock, measurements, detailed_values_dict, expected_values, wanted_desires = selected_desire)
    # all_desire_components = get_all_desire_components(model, general_pre_act_label, observation_without_pre_act, clock, measurements, detailed_values_dict, expected_values)


    target_tracking_desire_component = dict()
    for desire_name, desire_component in all_desire_components.items():
        target_tracking_desire_component[_get_class_name(desire_component)] = desire_component
    value_tracker = value_comp.ValueTracker(
        pre_act_key='',
        desire_components=target_tracking_desire_component,
        logging_channel=measurements.get_channel('ValueTracker').on_next,
        init_value = predefined_setting,
        expected_value_dict=expected_values,
        clock_now=clock.now,
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
        profile_comp,
        time_display,
        background_knowledge,

        # Components that do not provide pre_act context.
        observation,
        value_tracker,
        LLMobPlan_component,
        null_observation
    )

    entity_components += tuple(all_desire_components.values())

    components_of_agent = {_get_class_name(component): component
                            for component in entity_components}
    components_of_agent[
        agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME] = (
            agent_components.memory_component.MemoryComponent(raw_memory))
    component_order = list(components_of_agent.keys())
    if overarching_goal is not None:
        components_of_agent[goal_label] = overarching_goal
        # Place goal after the instructions.
        component_order.insert(1, goal_label)


    # LLMobActComponent component
    act_component = LLMobActComponent(
        model=model,
        clock=clock,
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
