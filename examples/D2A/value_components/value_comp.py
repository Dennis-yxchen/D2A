import random

from collections.abc import Mapping
import datetime
import types
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
from concordia.components.agent import memory_component
from collections.abc import Callable, Sequence
from concordia.typing import logging
from concordia.document import interactive_document
from concordia.typing.entity import ActionSpec
from .hardcoded_value_state import hardcode_state
DEFAULT_VALUE_SCALE = tuple(range(11))
def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__

class desire(agent_components.action_spec_ignored.ActionSpecIgnored):
    def __init__(self,
                 *,
                 model: language_model.LanguageModel,
                 pre_act_key: str,
                 observation_component_name: str,
                 add_to_memory: bool = False,
                 memory_component_name: str = (
                    memory_component.DEFAULT_MEMORY_COMPONENT_NAME
                ),
                 init_value:int,
                 value_name: str,
                 description: str,
                 decrease_step: int,
                 decrease_interval: int,
                 time_step: datetime.timedelta,
                 reverse: bool = False,
                 extra_instructions: str = '',
                 MAX_ITER = 2,
                 clock_now: Callable[[], datetime.datetime] | None = None,
                 logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
                ) -> None:
        super().__init__(pre_act_key)
        self._model = model
        self._add_to_memory = add_to_memory
        self._memory_component_name = memory_component_name
        self._observation_component_name = observation_component_name
        self._description = description
        self._decrease_step = decrease_step
        self._decrease_interval = decrease_interval
        self._time_step = time_step
        self._reverse = reverse
        self._extra_instructions = extra_instructions
        self._clock_now = clock_now
        self._logging_channel = logging_channel
        self._value_scale = [str(i) for i in sorted(DEFAULT_VALUE_SCALE)]
        self._value_change_cache = []
        self._action_cache = []
        self._value = int(init_value)
        self._value_name = value_name
        self._MAX_ITER = MAX_ITER

        self._decrease_interval_minutes = datetime.timedelta(hours=decrease_interval)
        # print(f"decrease_interval_minutes: {self._decrease_interval_minutes}")
        # print(f"time_step: {self._time_step}")
        # print(f"decrease_step: {decrease_step}")
        self.decrease_probability = decrease_step / (self._decrease_interval_minutes / self._time_step)

    # for update the value of the desire

    def _update_value_prompt(self,agent_name:str, action: str, observation: str,reflection_prompt_history:str, prompt:interactive_document.InteractiveDocument) -> str:
        question = (
                f"The current magnitude value of {self._value_name} is {round(self._value)}.\n"
                f"The agent {agent_name}'s action is: {action}.\n"
                f"And the consequence is: \n{observation}.\n"
                f"{self._description}"
                f"How would the magnitude value of {self._value_name} change according to the consequence of the action? \n"
                )

        if reflection_prompt_history != "":
                current_reflection = (f"There are some unreasonable examples:\n {reflection_prompt_history}\n")
                question += current_reflection
        zero, *_, ten = self._value_scale

        question += (
            f"Please select the final magnitude value after the event on the scale of {zero} to {ten}, "
            "if the consequence of the action will not affect the state value "
            "(e.g. The action is irrelevant with this value dimension or the action was failed to conduct), "
            "then maintain the previous magnitude value.\n"
            "Please just answer in the format of (a) (b) (c) (d) and so on, Rating: \n"
            # f"""Output format:
            # <Reason>
            # The final answer is: \n(Your choice in letter)\n"""
            # f"""Output example:
            # Since Alice felt more relaxed and centered after her actions......
            # The final answer is: (c)\n"""
            # f"**Make sure you answer in the format of a letter corresponding to your choice:**"
            )

        current_value = prompt.multiple_choice_question(question,answers=self._value_scale)
        return current_value, prompt.view().text()

    def _check_reasonable(self, agent_name, previous_value: int, current_value: int, action: str, observation: str, prompt:interactive_document.InteractiveDocument) -> bool:
        reasonable_question = (
                f"The current magnitude value of {self._value_name} is {round(self._value)}."
                f"The agent {agent_name}'s action is: {action}."
                f"And the consequence is: {observation}."
                + self._description ,
                f"The reward model has changed the magnitude value of {self._value_name} from {previous_value} to {current_value}. "
                f"Is the change of the magnitude value of {self._value_name} reasonable? "
                f"You should check whether the consequence can lead to a change in the magnitude value of {self._value_name} (e.g., looking for an item but not using it yet)."
                f"Please answer in the format of the letter with brackets : (a) Yes. (b) No."
            )

        reasonable = prompt.open_question(reasonable_question)
        if 'Yes' in reasonable or '(a)' in reasonable:
            reasonable = True
        else:
            reasonable = False

        return reasonable, prompt.view().text()

    def _think_why_not_reasonable(self, agent_name, previous_value: int, current_value: int, action: str, observation: str, prompt:interactive_document.InteractiveDocument) -> str:
        reflection_prompt = (
        f"The current magnitude value of {self._value_name} is {round(self._value)}."
        f"The agent {agent_name}'s action is: {action}."
        f"And the consequence is: {observation}."
        + self._description ,
        f"The reward model has changed the magnitude value of {self._value_name} from {previous_value} to {current_value}."
        f"And the change is not reasonable."
        f"You should consider whether the consequence can lead to the change of the magnitude value of {self._value_name} (e.g. looking for an item but not using it yet)."
        f"Please explain why the change of the magnitude value of {self._value_name} is not reasonable."
        )
        prefix = (
            f"After '{action}', {self._value_name} updated from {previous_value} to {current_value} is not reasonable because: "
        )
        reason = prompt.open_question(
            reflection_prompt,
            max_tokens=1200,
            answer_prefix=prefix,
            terminators=("\n\n\n", ),
            )
        reason = (f"{prefix}\n"
                  f"{reason}")
        return reason, prompt.view().text()

    def _update_value_from_action_and_observation(self, action_attempt: str, observation_value: str) -> dict:
        # previous action + current observation -:> current desire
        agent_name = self.get_entity().name

        previous_value = round(self._value) # current real value of the desire before update
        current_value = None # estimated value of the desire after update

        previous_result = ''
        reasonable = False
        reflection_prompt_history = ""

        current_step = 0

        reflective_log = dict()

        while reasonable == False:
            prompt = interactive_document.InteractiveDocument(self._model)
            current_value, prompt_string = self._update_value_prompt(agent_name, action_attempt, observation_value,reflection_prompt_history, prompt)
            reflective_log[current_step] = {
                "previous_value": previous_value,
                "current_value": current_value,
                'prompt': prompt.view().text(),
                'question': prompt_string,
            }
            prompt = prompt.new()
            # reasonable_question = (
            #     f"The current magnitude value of {self._value_name} is {self._value}."
            #     f"The agent {self._agent_name}'s action is: {action}."
            #     f"And the consequence is: {event_statement}."
            #     + self._description ,
            #     f"The reward model has changed the magnitude value of {self._value_name} from {previous_value} to {current_value}. "
            #     f"Is the change of the magnitude value of {self._value_name} reasonable? "
            #     f"You should check whether the consequence can lead to a change in the magnitude value of {self._value_name} (e.g., looking for an item but not using it yet)."
            #     f"Please answer in the format of the letter with brackets : (a) Yes. (b) No."
            #     )
            reasonable, reflection_prompt_prompt = self._check_reasonable(agent_name, previous_value, current_value, action_attempt, observation_value, prompt)

            reflective_log[current_step]['reasonable'] = {
                'Question': reflection_prompt_prompt,
                'Answer': reasonable
            }
            prompt = prompt.new()
            if reasonable == False:
                reason, reason_prompt = self._think_why_not_reasonable(agent_name, previous_value, current_value, action_attempt, observation_value, prompt)
                reflective_log[current_step]['why not reasonable'] = {
                'Question': reason_prompt,
                'Answer': reason
                }

                unreasonable_example = reason
                reflection_prompt_history += f"\n{unreasonable_example}\n"
                prompt = prompt.new()

                if current_step >= self._MAX_ITER:
                    _value_cache = previous_value
                    self._value = int(current_value)
                    break
                current_step += 1

            else:
                _value_cache = previous_value
                self._value = int(current_value)

        update_log = {
            'reflective_log': reflective_log,
            'action_attempt': action_attempt,
            'observation': observation_value,
            'value before update': _value_cache,
            'value after update': int(self._value)
            }
        return update_log

    # end here


    # for converting the numeric desire to qualitative desire
    def _convert_numeric_desire_to_qualitative(self) -> tuple[str, str]:
        agent_name = self.get_entity().name
        question = (
            f"How would one describe {agent_name}'s "
            f'{self._value_name} state given the current value {round(self._value)}? '
            f'{self._description} \n'
            f"Please answer in descriptive words. Do not include the numerical value in your answer."
            f'{self._extra_instructions}'
        )

        if self._clock_now is not None:
            question = f'Current time: {self._clock_now()}.\n{question}'

        prompt = interactive_document.InteractiveDocument(self._model)
        current_quatitative_value = prompt.open_question(
            question,
            max_tokens=1200,
            answer_prefix=f'{agent_name} is ',
            terminators=("\n\n\n",),
            )

        return current_quatitative_value, prompt.view().text()

    def _convert_numeric_desire_to_qualitative_by_hard_coding(self) -> str:
        current_value = round(self._value)
        qualitative_value = hardcode_state[self.get_desire_name()][current_value]
        return qualitative_value, "By hard coding"

    def _make_pre_act_value(self) -> str:

        updated_log = dict()
        # only for the first time, skip the following steps
        # step 1: get the previous action
        if len(self._action_cache) != 0:
            action_attempt = self._action_cache[-1]
            # step 2: get the current observation
            observation_value = self.get_named_component_pre_act_value(
                self._observation_component_name
            )

            # step 3: update the value of the desire
            updated_log = self._update_value_from_action_and_observation(action_attempt, observation_value)

        # print("after update the value of the desire")
        # end here
        # print(self._value_name, '\n',updated_log)

        # before make the pre act value, we need to fluctuate the value of the desire
        fluctuate_dict = self._fluctuate_value()




        # print("after fluctuate the value of the desire")

        # convert the numeric desire to qualitative desire
        # qualitative_desire, convert_numeric_prompt = self._convert_numeric_desire_to_qualitative()
        qualitative_desire, convert_numeric_prompt = self._convert_numeric_desire_to_qualitative_by_hard_coding()
        converted_log = {
            'convert_numeric_prompt': convert_numeric_prompt,
            'qualitative_desire': qualitative_desire
        }

        # print("after convert the numeric desire to qualitative desire")

        total_log = {
            'fluctuate_dict': fluctuate_dict,
            'update_log': updated_log,
            'converted_log': converted_log
        }
        self._logging_channel(total_log)

        return qualitative_desire

    def _fluctuate_value(self) -> dict:
        random_number = random.uniform(0,1)
        fluctuate_dict = dict()
        fluctuate_dict['value before fluctuation'] = round(self._value)
        if random_number < self.decrease_probability:
            if self._reverse:
                self._value = min(10,round(self._value)+1)
            else:
                self._value = max(0,round(self._value)-1)

        fluctuate_dict['random_number'] = random_number
        fluctuate_dict['decrease_probability'] = self.decrease_probability
        fluctuate_dict['Is decreased?'] = random_number < self.decrease_probability
        fluctuate_dict['value after fluctuation'] = round(self._value)

        return fluctuate_dict

    def get_current_numerical_value(self) -> int:
        self.get_pre_act_value() # update the value of the desire
        return int(self._value)

    def get_current_qualitative_value(self) -> str:
        return self.get_pre_act_value()

    def get_desire_name(self) -> str:
        return self._value_name

    def post_act(self, action_attempt: str) -> str:
        self._action_cache.append(action_attempt)
        return ''

class Hunger(desire):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class Thirst(desire):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class Comfort(desire):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class Health(desire):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class Sleepiness(desire):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class Joyfulness(desire):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class Cleanliness(desire):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class Safety(desire):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class Passion(desire):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class SpiritualSatisfaction(desire):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class SocialConnectivity(desire):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

## new added desire

class SenseOfControl(desire):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class Recognition(desire):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class SenseOfSuperiority(desire):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

# class SenseOfWonder(desire):
#     def __init__(self, *args, **kwargs):
#         super().__init__(**kwargs)

## end here


class desireWithoutPreAct(agent_components.action_spec_ignored.ActionSpecIgnored):
    def __init__(self,  *args, **kwargs):
        self._component = desire(*args, **kwargs)

    def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
        self._component.set_entity(entity)

    def _make_pre_act_value(self) -> str:
        return self._component.get_pre_act_value()

    def get_pre_act_value(self) -> str:
        return self._make_pre_act_value()

    def pre_act(
        self,
        unused_action_spec: entity_lib.ActionSpec,
    ) -> str:
        del unused_action_spec
        self.get_pre_act_value()
        return ''

    def update(self) -> None:
        self._component.update()

    def get_state(self) -> entity_component.ComponentState:
        return self._component.get_state()

    def set_state(self, state: entity_component.ComponentState) -> None:
        self._component.set_state(state)

    def get_desire_name(self) -> str:
        return self._component.get_desire_name()

    def get_current_numerical_value(self) -> int:
        return self._component.get_current_numerical_value()

    def get_current_qualitative_value(self) -> str:
        return self._component.get_current_qualitative_value()

    def post_act(self, action_attempt: str) -> str:
        return self._component.post_act(action_attempt)

class HungerWithoutPreAct(desireWithoutPreAct):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class ThirstWithoutPreAct(desireWithoutPreAct):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class ComfortWithoutPreAct(desireWithoutPreAct):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class HealthWithoutPreAct(desireWithoutPreAct):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class SleepinessWithoutPreAct(desireWithoutPreAct):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class JoyfulnessWithoutPreAct(desireWithoutPreAct):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class CleanlinessWithoutPreAct(desireWithoutPreAct):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class SafetyWithoutPreAct(desireWithoutPreAct):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class PassionWithoutPreAct(desireWithoutPreAct):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class SpiritualSatisfactionWithoutPreAct(desireWithoutPreAct):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class SocialConnectivityWithoutPreAct(desireWithoutPreAct):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

## new added desire

class SenseOfControlWithoutPreAct(desireWithoutPreAct):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class RecognitionWithoutPreAct(desireWithoutPreAct):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

class SenseOfSuperiorityWithoutPreAct(desireWithoutPreAct):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

# class SenseOfWonderWithoutPreAct(desireWithoutPreAct):
#     def __init__(self, *args, **kwargs):
#         super().__init__(**kwargs)


## end here

class ValueTracker(agent_components.action_spec_ignored.ActionSpecIgnored):
    def __init__(self, *,
                 pre_act_key: str,
                 desire_components: Mapping[str, desire],
                 init_value: Mapping[str, int],
                 expected_value_dict: Mapping[str, int],
                 clock_now: Callable[[], datetime.datetime] | None = None,
                 logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
                 ) -> None:
        super().__init__(pre_act_key)
        self._desire_components = dict(desire_components)
        self._logging_channel = logging_channel
        self._expected_value_dict = expected_value_dict
        self._step_counter = 0 # track the step, always the next step
        self._whole_delta_tracker = dict()
        self._individual_desire_tracker = dict()
        self._individual_delta_tracker = dict()
        self._individual_qualitative_desire_tracker = dict()
        # self._track_value()
        self._track_initial_value(init_value, expected_value_dict)
        self._action_cache = []
        self._clock_now = clock_now

    def _track_initial_value(self, init_value, expected_value):
        current_numerical_desire_tracker = dict()
        current_qualitative_desire_tracker = dict()
        current_delta_tracker = dict()
        for desire_component_name, desire_component in self._desire_components.items():
            current_value_name = desire_component.get_desire_name().lower()
            current_numerical_value = init_value[current_value_name]
            current_qualitative_value = None
            # print(f"current_value_name: {current_value_name}")
            # print(f"expected_value: {expected_value}")
            current_expected_value = expected_value[current_value_name]

            if current_value_name in ['hunger', 'thirst', 'sleepiness']:
                component_delta = max(0, current_numerical_value - current_expected_value)
            else:
                component_delta = max(0, current_expected_value - current_numerical_value)

            current_numerical_desire_tracker[current_value_name] = current_numerical_value
            current_qualitative_desire_tracker[current_value_name] = current_qualitative_value
            current_delta_tracker[current_value_name] = component_delta

        self._individual_desire_tracker[self._step_counter] = current_numerical_desire_tracker
        self._individual_delta_tracker[self._step_counter] = current_delta_tracker
        self._whole_delta_tracker[self._step_counter] = sum(current_delta_tracker.values())
        self._individual_qualitative_desire_tracker[self._step_counter] = current_qualitative_desire_tracker
        self._step_counter += 1


    def _track_value(self):
        current_numerical_desire_tracker = dict() # track the current numerical value of the desire
        current_qualitative_desire_tracker = dict() # track the current qualitative value of the desire
        current_delta_tracker = dict() # track the delta of the desire
        for desire_component_name, desire_component in self._desire_components.items():
            current_numerical_value = desire_component.get_current_numerical_value()
            current_qualitative_value = desire_component.get_current_qualitative_value()
            current_value_name = desire_component.get_desire_name().lower()
            expected_value = self._expected_value_dict[current_value_name]

            if current_value_name in ['hunger', 'thirst', 'sleepiness']:
                component_delta = max(0, current_numerical_value - expected_value)
            else:
                component_delta = max(0, expected_value - current_numerical_value)

            current_numerical_desire_tracker[current_value_name] = current_numerical_value
            current_qualitative_desire_tracker[current_value_name] = current_qualitative_value
            current_delta_tracker[current_value_name] = component_delta

        self._individual_desire_tracker[self._step_counter] = current_numerical_desire_tracker
        self._individual_delta_tracker[self._step_counter] = current_delta_tracker
        self._whole_delta_tracker[self._step_counter] = sum(current_delta_tracker.values())
        self._individual_qualitative_desire_tracker[self._step_counter] = current_qualitative_desire_tracker
        self._step_counter += 1


    def _make_pre_act_value(self) -> str:
        index = self._step_counter
        self._track_value()
        self._logging_channel({
            'index': index,
            'individual_desire_tracker': self._individual_desire_tracker[index],
            'individual_delta_tracker': self._individual_delta_tracker[index],
            'whole_delta_tracker': self._whole_delta_tracker[index],
            'individual_qualitative_desire_tracker': self._individual_qualitative_desire_tracker[index]
        })
        return "The value of the desire has been updated."

    def get_action_sequence(self) -> Sequence[dict]:
        return self._action_cache

    def pre_act(
        self,
        unused_action_spec: entity_lib.ActionSpec,
    ) -> str:
        del unused_action_spec
        self.get_pre_act_value()
        return ''

    def post_act(self, action_attempt: str) -> str:
        self._action_cache.append({'timestamp': self._clock_now(), 'action':action_attempt})
        return ''

    def get_whole_delta_tracker(self) -> dict:
        return self._whole_delta_tracker

    def get_individual_delta_tracker(self) -> dict:
        return self._individual_delta_tracker

    def get_individual_desire_tracker(self) -> dict:
        return self._individual_desire_tracker

    def get_individual_qualitative_desire_tracker(self) -> dict:
        return self._individual_qualitative_desire_tracker

    def get_expected_value_dict(self) -> dict:
        return self._expected_value_dict
