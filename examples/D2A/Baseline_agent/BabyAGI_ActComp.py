
from collections import deque
from collections.abc import Sequence
import json

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import clock as game_clock
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import logging
from concordia.utils import helper_functions
from typing_extensions import override
from concordia.components import agent as agent_components
from concordia.memory_bank import legacy_associative_memory
_ASSOCIATIVE_RETRIEVAL = legacy_associative_memory.RetrieveAssociative()

DEFAULT_PRE_ACT_KEY = 'Act'
def reformat_prioritized_actions(response: str, this_action_id: int):
  new_actions = response.split("\n")
  print(f"new_actions: {new_actions}")
  prioritized_action_list = []
  for action_string in new_actions:
    if not action_string.strip(): # skip empty lines
      continue
    action_parts = action_string.strip().split(".", 1)
    print(f"action_parts: {action_parts}")
    if len(action_parts) == 2:
          action_id = action_parts[0].strip()
          action_name = action_parts[1].strip()
          prioritized_action_list.append({"action_id": this_action_id, "action_name": action_name})
          this_action_id += 1

  return prioritized_action_list, this_action_id

class BabyAGIActComponent(entity_component.ActingComponent):
    def __init__(
      self,
      model: language_model.LanguageModel,
      clock: game_clock.GameClock,
      profile:str,
      background_component_name: str,
      observation_component_name: str,
      memory_component_name: str,
      component_order: Sequence[str] | None = None,
      pre_act_key: str = DEFAULT_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        self._model = model
        self._clock = clock
        self._profile = profile
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
        self._background_component_name = background_component_name
        self._observation_component_name = observation_component_name
        self._memory_component_name = memory_component_name

        ## only for BabyAGI
        self.action_list = deque()
        self.action_id_counter = 1
        self.current_action_id = 1
        self.previous_action = []


    def _context_for_action(
        self,
        contexts: entity_component.ComponentContextMapping,
    ) -> str:
        # if self._component_order is None:
        #     return '\n'.join(
        #         context for context in contexts.values() if context
        # )
        # else:
        #     order = self._component_order + tuple(sorted(
        #         set(contexts.keys()) - set(self._component_order)))
        #     return '\n'.join(
        #         contexts[name] for name in order if contexts[name]
        #     )
        if self._component_order is None:
            other_comp = [context for name, context in contexts.items()
                          if name != self._background_component_name
                          and name != self._observation_component_name]
            return '\n'.join(
                context for context in other_comp if context
            )
        else:
            order = self._component_order + tuple(sorted(
                set(contexts.keys()) - set(self._component_order)))
            other_comp = [
                contexts[name] for name in order
                if name != self._background_component_name
                and name != self._observation_component_name
                and contexts[name]
                ]
            return '\n'.join(
                other_comp
            )
    def add_action(self, action:dict):
      self.action_list.append(action)

    def action_creation(self, observation:str, previous_one_action:str):
      current_id = self.current_action_id
      if current_id == 1:
        return None

      memory = self.get_entity().get_component(
         self._memory_component_name,
          type_=agent_components.memory_component.MemoryComponent
          )

      relavent_scorer = _ASSOCIATIVE_RETRIEVAL

      mems = memory.retrieve(
         query = observation,
         scoring_fn=relavent_scorer,
         limit=5,
         )

      mem_text = '\n'.join([mem.text for mem in mems if '[action]' in mem.text])

      enrichment_log = dict()

      prompt = interactive_document.InteractiveDocument(self._model)
      enriched_observation = prompt.open_question(f"""
      Action: {previous_one_action}
      Observation after action: {observation}
      Instruction: Based on the action and the observation, explain why or why not the action was successful in several sentences.
      """, terminators=())
      enrichment_log["enriched_observation"] = enriched_observation
      enrichment_log["enriched_observation_prompt"] = prompt.view().text()
      enrichment_log["previous_one_action"] = previous_one_action
      enrichment_log["observation"] = observation

      memory.add(
        text=f"[action] Action: {previous_one_action}\nAnalysis: {enriched_observation}",
        metadata={'tags': ['action']},
        )

      incomplete_actions = '\n'.join([action["action_name"] for action in self.action_list])
      profile = self._profile
      background = self.get_entity().get_component(self._background_component_name,
                                                   type_=agent_components.constant.Constant)
      background_info = background.get_pre_act_value()

      action_creation_log = dict()
      creation_prompt = rf"""
        Environment: {background_info}
        Profile: {profile}
        Incompleted action: {incomplete_actions}
        Current action: {previous_one_action}
        Result of current action: {observation}
        Related context: {mem_text}
        Instruction: According to your characteristics and the result of the current action, create new actions to be completed that do not overlap with incomplete actions.
        Please provide the output in the following JSON format with '\n' as the separator:
        {{"action": "one action that you would likely take"}}
        {{"action": "another action you would likely take"}}

        Ensure the output is strictly in JSON format without any additional text or explanation.
        """

      prompt = prompt.new()
      new_actions = prompt.open_question(creation_prompt, terminators=())
      new_actions = new_actions.split("\n")

      action_creation_log["creation_prompt"] = creation_prompt
      action_creation_log["new_actions"] = new_actions
      action_creation_log["prompt"] = prompt.view().text()

      return new_actions, enrichment_log, action_creation_log






    def create_next_action(self, observation: str, previous_one_action: str):
      new_actions, enrichment_log, action_creation_log  = self.action_creation(observation, previous_one_action)
      if new_actions == None:
        return
      for action in new_actions:
        if not action.strip():
          continue

        try:
          json_action = json.loads(action)
        except json.JSONDecodeError:
          continue

        new_action = {
          "action_id": self.action_id_counter,
          "action_name": json_action["action"]
        }
        self.add_action(new_action)
        self.action_id_counter += 1

      return enrichment_log, action_creation_log

    @override
    def get_action_attempt(
        self,
        contexts: entity_component.ComponentContextMapping,
        action_spec: entity_lib.ActionSpec,
        ) -> str:
      """Get an action attempt from the model."""
      context = self._context_for_action(contexts)
      agent_name = self.get_entity().name

      prompt = interactive_document.InteractiveDocument(self._model)
      # current_action_id is the action_id of the current action
      # action_id_counter is the last action_id that was added
      self.action_id_counter = self.current_action_id
      background = contexts[self._background_component_name]
      time = self._clock.now()
      observation = contexts[self._observation_component_name]
      profile = self._profile
      init_action_log = dict()
      if self.action_id_counter == 1 or len(self.action_list) == 0:
          initial_action_context = f"""
            Environment: {background}
            Context: {agent_name} lives in the given environment.
            Current time: {time}
            Profile: {profile}
            Notice that {agent_name} can only interact with the items that provided by the environment.
            Instructions: Reflecting on the context and profile given, I would like you to suggest some actions that {agent_name} would likely take in this environment.
            Please provide the output in the following JSON format with '\n' as the separator:
            {{"action": "one action that {agent_name} would likely take"}}
            {{"action": "another action {agent_name} would likely take"}}

            Ensure the output is strictly in JSON format without any additional text or explanation.
            """
          response = prompt.open_question(initial_action_context, terminators = ())
          new_actions = response.split("\n")
          for new_action in new_actions:
            if not new_action.strip():
              continue
            try:
              new_action = json.loads(new_action)
            except json.JSONDecodeError:
              print(f"Error in decoding JSON: {new_action}")
              print(f"prompt: {prompt.view().text()}")
              continue
            self.add_action({"action_id": self.action_id_counter, "action_name": new_action["action"]})
            self.action_id_counter += 1

          init_action_log["action_id_after_init_before_prioritize"] = self.action_id_counter
          init_action_log["current_action_id"] = self.current_action_id
          init_action_log["actions"] = self.action_list
          init_action_log["prompt to initial actions"] = prompt.view().text()


      enrichment_log = dict()
      action_creation_log = dict()
      if observation and self.previous_action:
         previous_one_action = self.previous_action[-1]
         enrichment_log, action_creation_log = self.create_next_action(observation, previous_one_action)


      prioritize_log = dict()
      action_names = '\n'.join([t["action_name"] for t in self.action_list])
      prioritize_prompt = rf"""
        Environment: {background}
        Current time: {time}
        Profile: {profile}
        Incompleted actions:
        {action_names}

        Instruction: According to your characteristics, please prioritize the following actions based on your characteristics and the environment. Do not remove any actions.
        Output format:
        #. First action
        #. Second action
        Output example:
        1. go to kitchen and make a cup of coffee
        Start the action list with number {self.current_action_id}. Do not explain the reasons for prioritizing the actions.
        """
      prompt = prompt.new()
      prioritize_response = prompt.open_question(prioritize_prompt, terminators = ())

      prioritize_log["prioritize_prompt"] = prioritize_prompt
      prioritize_log["prioritize_response"] = prioritize_response
      prioritize_log["prioritize_prompt"] = prompt.view().text()
      prioritize_log["current_action_id"] = self.current_action_id
      prioritize_log["action_id_before_prioritize"] = self.action_id_counter
      prioritize_log["before_prioritize_action_names"] = action_names

      prioritize_list, self.action_id_counter = reformat_prioritized_actions(prioritize_response, self.current_action_id)
      self.action_list = deque(prioritize_list)
      prioritize_log["after_prioritize_action_names"] = str(self.action_list)
      prioritize_log["action_id_after_prioritize"] = self.action_id_counter

      next_action_log = dict()
      if len(self.action_list) != 0:
        next_action = self.action_list.popleft()
        output = next_action["action_name"]
      else:
        next_action = "Alice is thinking about what to do next."
        output = "Alice is thinking about what to do next."


      self.previous_action.append(output)
      self.current_action_id += 1
      next_action_log["next_action"] = next_action
      next_action_log["after_get_action_latest_id"] = self.current_action_id

      total_log = dict()
      total_log["init_action_log"] = init_action_log
      total_log["enrichment_log"] = enrichment_log
      total_log["action_creation_log"] = action_creation_log
      total_log["prioritize_log"] = prioritize_log
      total_log["next_action_log"] = next_action_log
      self._log(total_log, prompt)
      return output

    def _log(self,
            result: str,
            prompt: interactive_document.InteractiveDocument):
        self._logging_channel({
            'Key': self._pre_act_key,
            'Value': result,
            'last prompt': prompt.view().text().splitlines(),
        })

    def get_state(self) -> entity_component.ComponentState:
        """Converts the component to a dictionary."""
        return {}

    def set_state(self, state: entity_component.ComponentState) -> None:
        pass
