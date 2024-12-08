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
