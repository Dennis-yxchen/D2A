"""A factory implementing the three key questions agent as an entity."""

from collections.abc import Callable
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
from typing import Mapping, Sequence
import importlib
IMPORT_AGENT_BASE_DIR = 'examples.D2A.value_components'
init_value_info_social = importlib.import_module(
    f'{IMPORT_AGENT_BASE_DIR}.init_value_info_social')
value_comp = importlib.import_module(f'{IMPORT_AGENT_BASE_DIR}.value_comp')

from .ReAct_ActComp import ReActComponent
import NullObservation


class ObservationWithoutPreAct(agent_components.action_spec_ignored.ActionSpecIgnored):
    """A simple component to receive observations."""

    def __init__(
        self,
        clock_now: Callable[[], datetime.datetime],
        timeframe: datetime.timedelta,
        memory_component_name: str = (
            agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME),
        pre_act_key: str = agent_components.observation.DEFAULT_OBSERVATION_PRE_ACT_KEY,
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        """Initializes the observation component.

        Args:

        clock_now: Function that returns the current time.
        timeframe: Delta from current moment to display observations from, e.g. 1h
            would display all observations made in the last hour.
        memory_component_name: Name of the memory component to add observations to
            in `pre_observe` and to retrieve observations from in `pre_act`.
        pre_act_key: Prefix to add to the output of the component when called
            in `pre_act`.
        logging_channel: The channel to use for debug
        """

        self._component = agent_components.observation.Observation(
            clock_now=clock_now,
            timeframe=timeframe,
            memory_component_name=memory_component_name,
            pre_act_key=pre_act_key,
            logging_channel=logging_channel,
        )

    def pre_observe(self, observation: str) -> str:
        return self._component.pre_observe(observation)

    def pre_act(
        self,
        unused_action_spec: entity_lib.ActionSpec,
    ) -> str:
        del unused_action_spec
        self.get_pre_act_value()
        return ''

    def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
        return self._component.set_entity(entity)

    def _make_pre_act_value(self) -> str:
        return self._component.get_pre_act_value()

    def get_pre_act_value(self) -> str:
        pre_act_value = self._make_pre_act_value()
        return pre_act_value

    def update(self) -> None:
        self._component.update()

    def get_state(self) -> entity_component.ComponentState:
        return self._component.get_state()

    def set_state(self, state: entity_component.ComponentState) -> None:
        self._component.set_state(state)



class BackgroundKnowledge(agent_components.constant.Constant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__

def build_ReAct_agent(
    *,
    config: formative_memories.AgentConfig,
    context_dict: Mapping[str, str], # contain value related context
    selected_desire: Sequence[str], # contain value related desire
    predefined_setting: Mapping[str, Mapping[str, str]], # contain value related setting
    model: language_model.LanguageModel,
    background_knowledge: str,
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

    observation_without_pre_act = ObservationWithoutPreAct(
        clock_now=clock.now,
        timeframe=datetime.timedelta(hours=1),
        memory_component_name=agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
        logging_channel=measurements.get_channel('Observation').on_next,
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

    ## Value Components
    general_pre_act_label = f"\n{agent_name}" + "'s current feeling of {desire_name} is"
    ### init the information to be used in the value component
    detailed_values_dict, expected_values = init_value_info_social.preprocess_value_information(context_dict, predefined_setting, selected_desires=selected_desire)
    print(f"detailed_values_dict: {detailed_values_dict}")
    print(f"expected_values: {expected_values}")
    print(f"len of detailed_values_dict: {len(detailed_values_dict)}")
    print(f"len of expected_values: {len(expected_values)}")
    all_desire_components = init_value_info_social.get_all_desire_components_without_PreAct(model,
                                                                     general_pre_act_label,
                                                                     observation_without_pre_act,
                                                                     clock, measurements,
                                                                     detailed_values_dict,
                                                                     expected_values,
                                                                     wanted_desires=selected_desire)



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
        time_display,
        background_knowledge,

        # Components that do not provide pre_act context.
        observation_without_pre_act,
        null_observation,
        value_tracker,
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


    # Reason-Act component
    act_component = ReActComponent(
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
