from collections.abc import Sequence

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import clock as game_clock
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import logging
from concordia.utils import helper_functions
from typing_extensions import override

DEFAULT_PRE_ACT_KEY = 'Act'


class LLMobActComponent(entity_component.ActingComponent):
  def __init__(
      self,
      model: language_model.LanguageModel,
      clock: game_clock.GameClock,
      component_order: Sequence[str] | None = None,
      pre_act_key: str = DEFAULT_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        self._model = model
        self._clock = clock
        if component_order is None:
            self._component_order = None
        else:
            self._component_order = tuple(component_order)
        if self._component_order is not None:
            if len(set(self._component_order)) != len(self._component_order):
                raise ValueError(
                    'The component order contains duplicate components: '
                    + ', '.join(self._component_order)
                )

        self._pre_act_key = pre_act_key
        self._logging_channel = logging_channel

  def _context_for_action(
        self,
        contexts: entity_component.ComponentContextMapping,
    ) -> str:
        if self._component_order is None:
            # context = [context for context in contexts.values() if context]
            LLMob_plan_context = contexts['LLMobPlan']
            other_contexts = [context for context in contexts.values() if context and context != LLMob_plan_context]

        else:
            order = self._component_order + tuple(sorted(
                set(contexts.keys()) - set(self._component_order)))
            # context = [contexts[name] for name in order if contexts[name]]
            LLMob_plan_context = contexts['LLMobPlan']
            other_contexts = [contexts[name] for name in order if contexts[name] and name != 'LLMobPlan']

        other_contexts_string = '\n'.join(other_contexts)
        return other_contexts_string + '\n' + f"{LLMob_plan_context}"
  @override
  def get_action_attempt(
      self,
      contexts: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    prompt = interactive_document.InteractiveDocument(self._model)
    context = self._context_for_action(contexts)
    prompt.statement(context + '\n')

    call_to_action = action_spec.call_to_action.format(
        name=self.get_entity().name,
        timedelta=helper_functions.timedelta_to_readable_str(
            self._clock.get_step_size()
        ),
    )
    if action_spec.output_type == entity_lib.OutputType.FREE:
      output = self.get_entity().name + ' '
      output += prompt.open_question(
          call_to_action,
          max_tokens=2200,
          answer_prefix=output,
          # This terminator protects against the model providing extra context
          # after the end of a directly spoken response, since it normally
          # puts a space after a quotation mark only in these cases.
          terminators=('" ', '\n'),
          question_label='Exercise',
      )
      self._log(output, prompt)
      return output
    elif action_spec.output_type == entity_lib.OutputType.CHOICE:
      idx = prompt.multiple_choice_question(
          question=call_to_action, answers=action_spec.options
      )
      output = action_spec.options[idx]
      self._log(output, prompt)
      return output
    elif action_spec.output_type == entity_lib.OutputType.FLOAT:
      prefix = self.get_entity().name + ' '
      sampled_text = prompt.open_question(
          call_to_action,
          max_tokens=2200,
          answer_prefix=prefix,
      )
      self._log(sampled_text, prompt)
      try:
        return str(float(sampled_text))
      except ValueError:
        return '0.0'
    else:
      raise NotImplementedError(
          f'Unsupported output type: {action_spec.output_type}. '
          'Supported output types are: FREE, CHOICE, and FLOAT.'
      )

  def _log(self,
          result: str,
          prompt: interactive_document.InteractiveDocument):
    self._logging_channel({
        'Key': self._pre_act_key,
        'Value': result,
        'Prompt': prompt.view().text().splitlines(),
    })

  def get_state(self) -> entity_component.ComponentState:
    """Converts the component to a dictionary."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass
