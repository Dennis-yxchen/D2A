import datetime
import functools
import random

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib

from concordia.components.agent import question_of_recent_memories
from concordia.components.agent import observation
from typing import Sequence
from typing_extensions import override

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import logging
from concordia.utils import concurrency
from collections.abc import Callable, Mapping
import types
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import logging
from concordia.utils import helper_functions
_ASSOCIATIVE_RETRIEVAL = legacy_associative_memory.RetrieveAssociative()

class NULLObservation(observation.Observation):
    """A simple component to receive observations. All memory will be stored in log"""
    """ version: 11072100"""

    def __init__(
        self,
        clock_now: Callable[[], datetime.datetime],
        timeframe: datetime.timedelta,
        memory_component_name: str = (
            memory_component.DEFAULT_MEMORY_COMPONENT_NAME),
        pre_act_key: str = 'NULLObservation',
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
        logging_channel: The channel to use for debug logging.
        """
        super().__init__(clock_now=clock_now, timeframe=timeframe, memory_component_name=memory_component_name, pre_act_key=pre_act_key, logging_channel=logging_channel)

    def pre_observe(
        self,
        observation: str,
    ) -> str:
        return ''

    def _make_pre_act_value(self) -> str:
        """Returns the latest observations to preact."""
        memory = self.get_entity().get_component(
            self._memory_component_name,
            type_=memory_component.MemoryComponent)
        # removes memories that are not observations
        mems = memory.retrieve(
            scoring_fn=_ASSOCIATIVE_RETRIEVAL,
            limit=-1,
            query=''
            )
        # Remove memories that are not observations.
        mems = [mem.text for mem in mems]
        result = '\n'.join(mems) + '\n'
        self._logging_channel(
            {'Key': self.get_pre_act_key(), 'Value': result.splitlines()})

        return ''


    def pre_act(
        self,
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        del action_spec
        self._make_pre_act_value()
        return ''
