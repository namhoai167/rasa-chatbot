# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

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

# This is a simple example for a custom action which utters "Hello World!"
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
        return "action_fallback_chat"
        
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Load BlenderBot
        # Maybe put in init function or __init__.py
        BB_PATH = './blenderbot_small-90M'
        BBModel = BlenderbotSmallForConditionalGeneration.from_pretrained(BB_PATH)
        BBTokenizer = BlenderbotSmallTokenizer.from_pretrained(BB_PATH)

        latest_user_message = tracker.latest_message['text']

        inputs = BBTokenizer([latest_user_message], return_tensors='pt')
        reply_ids = BBModel.generate(**inputs)
        bot_reply = BBTokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        dispatcher.utter_message(text=str(bot_reply))

        return []

class ActionSolveMulipleChoiceSentenceCompletrion(Action):

    def name(self) -> Text:
        return "action_solve_multiple_choice_sentence_completion"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        ELECTRA_PATH = './electra-small-generator'
        ELECTRAmodel = ElectraForMaskedLM.from_pretrained(ELECTRA_PATH)
        ELECTRAtokenizer = ElectraTokenizer.from_pretrained(ELECTRA_PATH)
        fb = FitBert(model=ELECTRAmodel, tokenizer=ELECTRAtokenizer)
        sentence_value = tracker.get_slot("sentence")
        sentence_value = sentence_value.replace('_', '***mask***')
        answers = [
            tracker.get_slot("answer_a"),
            tracker.get_slot("answer_b"), 
            tracker.get_slot("answer_c"), 
            tracker.get_slot("answer_d")
        ]
        bot_choice = fb.rank(sentence_value, options=answers)[0]
        dispatcher.utter_message(text=f"My guess is: \"{bot_choice}\"")

        return []