# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from fitbert import FitBert
from transformers import (
    BlenderbotSmallTokenizer, 
    BlenderbotSmallForConditionalGeneration,
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    ElectraForMaskedLM,
    ElectraTokenizer,
    pipeline
)
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Hello World!")

        return []

class ActionOnFallBack(Action):

    def name(self) -> Text:
        return "utter_fallback_chat"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Here's a fallback! Implement Blenderbot here")

        return []