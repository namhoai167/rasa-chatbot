# This file contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

from typing import Any, Text, Dict, List, Optional
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.types import DomainDict
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, AllSlotsReset
from rasa_sdk.interfaces import ActionExecutionRejection

from fitbert import FitBert
from transformers import (
    BlenderbotSmallTokenizer,
    BlenderbotSmallForConditionalGeneration,
    ElectraForMaskedLM,
    ElectraTokenizer
)
from gingerit.gingerit import GingerIt
import language_tool_python
import re
import requests as rq
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import tokenize
nltk.download('wordnet')
nltk.download('punkt')

language_tool = language_tool_python.LanguageTool('en-US')
gg = GingerIt()
ELECTRAmodel = ElectraForMaskedLM.from_pretrained('./electra-small-generator')
ELECTRAtokenizer = ElectraTokenizer.from_pretrained('./electra-small-generator')

def cut_paragraph(text, maximum_len=300):
    sentences = tokenize.sent_tokenize(text)
    len_for_each_sentence = [len(sentence) for sentence in sentences]
    splited_paragraph = []
    while len(sentences) > 0:
        current_len = 0
        for i in range(len(sentences)):
            current_len += len_for_each_sentence[i] + 1
            if current_len > maximum_len:
                splited_paragraph.append(
                    " ".join([sentence for idx, sentence in enumerate(sentences) if idx < i]))
                del sentences[:i]
                del len_for_each_sentence[:i]
                break
            if i == len(sentences) - 1:
                splited_paragraph.append(
                    " ".join([sentence for sentence in sentences]))
                return splited_paragraph


def split_into_sentences(text, string=True, joiner='\n'):
    sentences = tokenize.sent_tokenize(text)
    if string:
        return joiner.join(sentences)
    return sentences


def correct_gingerit(text, gg):
    splited_text = cut_paragraph(text)
    return split_into_sentences(" ".join([gg.parse(t)['result'] for t in splited_text]))


def correct_language_tool(text, language_tool):
    return(split_into_sentences(language_tool.correct(text)))


def get(l, idx=0, ret_when_error=''):
    try:
        return l[idx]
    except:
        return ret_when_error


def remove_duplicates(error_list):
    error_list = list(error_list)
    error_list = np.array(error_list)
    new_array = np.empty(shape=(0, error_list.shape[1]))
    for error in error_list:
        if error[1] not in new_array[:, 1]:
            new_array = np.vstack([new_array, error])
        else:
            if int(error[2]) == int(new_array[np.where(new_array[:, 1] == error[1])][0, 2]):
                new_array[np.where(new_array[:, 1] == error[1])[
                    0][0], 0] += " / " + error[0]
            elif int(error[2]) > int(new_array[np.where(new_array[:, 1] == error[1])][0, 2]):
                new_array[np.where(new_array[:, 1] == error[1])[0][0]:] = error
    return new_array.tolist()


def print_errors(text, language_tool, gg):
    text = split_into_sentences(text)
    sub_texts = cut_paragraph(text)

    # This will be list of sets, each item of a set is a tuple which is (correct_word, start_idx, end_idx)
    unified_errors = []

    # Make unified_language_tool_errors
    language_tool_errors = [language_tool.check(t) for t in sub_texts]
    unified_language_tool_errors = []
    for subtext_language_tool_errors in language_tool_errors:
        unified_language_tool_errors.append(set(
            (get(error.replacements), error.offset, error.offset+error.errorLength-1) for error in subtext_language_tool_errors))
    # print(unified_language_tool_errors)
    # print('\n')

    # Make unified_gg_errors
    unified_gg_errors = []
    gg_errors = [gg.parse(t)['corrections'] for t in sub_texts]
    for subtext_gg_errors in gg_errors:
        unified_gg_errors.append(set(
            (error['correct'], error['start'], error['start'] + len(error['text']) - 1) for error in subtext_gg_errors))
    # print(unified_gg_errors)
    # print('\n')
    # Merge two lists of sets (union)
    for language_tool_set, gg_set in zip(unified_language_tool_errors, unified_gg_errors):
        unified_errors.append(language_tool_set | gg_set)
    # print(unified_errors)
    # print('\n')
    # Handle duplicate or colided errors
    unified_errors = [remove_duplicates(error_list)
                      for error_list in unified_errors]
    # print(unified_errors)
    # print('\n')
    for idx, subtext_errors in enumerate(unified_errors):
        for error in sorted(subtext_errors, key=lambda x: int(x[1]), reverse=True):
            start_idx = int(error[1])
            end_idx = int(error[2])
            if error[0] == '':
                sub_texts[idx] = sub_texts[idx][:start_idx] + "{" + sub_texts[idx][start_idx:end_idx +
                                                                                   1] + "}" + """ (This may have a semantic error)""" + sub_texts[idx][end_idx+1:]
            else:
                sub_texts[idx] = sub_texts[idx][:start_idx] + "{" + sub_texts[idx][start_idx:end_idx +
                                                                                   1] + "}" + """ (Did you mean: {0})""".format(error[0]) + sub_texts[idx][end_idx+1:]
    return split_into_sentences(" ".join(sub_texts))


class ActionOnFallBack(Action):

    def name(self) -> Text:
        return "action_fallback_chat"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Load BlenderBot
        # Maybe we should put this in init function or __init__.py later on
        BB_PATH = './blenderbot_small-90M'
        BBModel = BlenderbotSmallForConditionalGeneration.from_pretrained(
            BB_PATH)
        BBTokenizer = BlenderbotSmallTokenizer.from_pretrained(BB_PATH)
        latest_user_message = tracker.latest_message['text']
        correction = correct_language_tool(latest_user_message, language_tool)
        correction = correct_gingerit(correction, gg)
        inputs = BBTokenizer([latest_user_message], return_tensors='pt')
        reply_ids = BBModel.generate(**inputs)
        bot_reply = BBTokenizer.batch_decode(
            reply_ids, skip_special_tokens=True)[0]
        bot_reply = " ".join([sent.capitalize() for sent in split_into_sentences(
            str(bot_reply), string=False)])
        dispatcher.utter_message(text=bot_reply)
        if latest_user_message.casefold() != correction.casefold():
            dispatcher.utter_message(
                "Small note: I found some mistakes in your message! Did you mean \"{0}\"?".format(correction))
        return [AllSlotsReset()]


# https://rasa.com/docs/rasa/forms#dynamic-form-behavior
# https://github.com/RasaHQ/rasa-form-examples/blob/main/06-custom-name-experience/actions/actions.py
class ValidateTOEICpart5Form(FormValidationAction):
    def name(self) -> Text:
        return "validate_TOEIC_part5_form"
    
    async def required_slots(
        self,
        slots_mapped_in_domain: List[Text],
        dispatcher: "CollectingDispatcher",
        tracker: "Tracker",
        domain: "DomainDict",
    ) -> Optional[List[Text]]:

        # Trigger when sentence too short. If the incomple sentence too short, maybe it just a word
        # Or there are no _ or -- to MASK in sentence
        sentence = tracker.get_slot("incomplete_sentence")
        if sentence is not None:
            if len(sentence) < 20 or not re.search(r'[_]+|[-]{2,}', sentence):
                dispatcher.utter_message(template="utter_ask_incomplete_sentence_type_correctly", sentence=sentence)
                return ["incomplete_sentence_type_correctly"] + slots_mapped_in_domain

        # Trigger when only one choices
        choices = tracker.get_slot("choices")
        if choices is not None:
            if isinstance(choices, str):
                choices = re.split(r'\([A-Da-d]\)|[A-Da-d]\s+?\.|\n+|,', choices)
            elif isinstance(choices, list) and len(choices) == 1:
                choices = re.split(r'\([A-Da-d]\)|[A-Da-d]\s+?\.|\n+|,', choices[0])
            choices = [x for x in choices if x.strip()]
            if len(choices) < 2:
                dispatcher.utter_message(template="utter_ask_choices_type_correctly", choicez=choices)
                return ["choices_type_correctly"] + slots_mapped_in_domain
        return slots_mapped_in_domain
    
    async def extract_incomplete_sentence_type_correctly(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> Dict[Text, Any]:
        intent = tracker.get_intent_of_latest_message()
        if intent not in ["affirm", "deny", "cancel"]:
            return {"incomplete_sentence_type_correctly": None}
        return {"incomplete_sentence_type_correctly": intent}

    async def extract_choices_type_correctly(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> Dict[Text, Any]:
        intent = tracker.get_intent_of_latest_message()
        if intent not in ["affirm", "deny", "cancel"]:
            return {"choices_type_correctly": None}
        return {"choices_type_correctly": intent}

    def validate_choices_type_correctly(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `choices_type_correctly` value."""
        user_intent = tracker.get_slot("choices_type_correctly")
        if user_intent == "affirm":
            return {"choices": tracker.get_slot("choices"), "choices_type_correctly": "affirm"}
        elif user_intent == "deny":
            return {"choices": None, "choices_type_correctly": "deny"}
        elif user_intent == "cancel":
            return ActionExecutionRejection()
        return {"choices": None, "choices_type_correctly": None}

    def validate_incomplete_sentence_type_correctly(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `incomplete_sentence_type_correctly` value."""
        user_intent = tracker.get_slot("incomplete_sentence_type_correctly")
        if user_intent == "affirm":
            return {"incomplete_sentence": tracker.get_slot("incomplete_sentence"), "incomplete_sentence_type_correctly": "affirm"}
        elif user_intent == "deny":
            return {"incomplete_sentence": None, "incomplete_sentence_type_correctly": "deny"}
        elif user_intent == "cancel":
            return ActionExecutionRejection()
        return {"incomplete_sentence": None, "incomplete_sentence_type_correctly": None}

    def validate_incomplete_sentence(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        print(slot_value)
        if len(slot_value) < 20 or not re.search(r'[_]+|[-]{2,}', slot_value):
            dispatcher.utter_message(template="utter_ask_incomplete_sentence_type_correctly", sentence=slot_value)
            return {"incomplete_sentence": None}
        else:
            return {"incomplete_sentence": slot_value}

    def validate_choices(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        print(slot_value)
        if isinstance(slot_value, str):
            choices = re.split(r'\([A-Da-d]\)|[A-Da-d]\s+?\.|\n+|,', slot_value)
        elif isinstance(slot_value, list) and len(slot_value) == 1:
            choices = re.split(r'\([A-Da-d]\)|[A-Da-d]\s+?\.|\n+|,', slot_value[0])
        choices = [x for x in choices if x.strip()]
        if len(choices) < 2:
            dispatcher.utter_message(template="utter_ask_choices_type_correctly", choicez=choices)
            return {"choices": None}
        else:
            return {"choices": slot_value}
                     

class ActionSolveMultipleChoiceSentenceCompletion(Action):

    def name(self) -> Text:
        return "action_solve_TOEIC_part5"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        fb = FitBert(model=ELECTRAmodel, tokenizer=ELECTRAtokenizer)
        sentence = str(tracker.get_slot("incomplete_sentence"))
        sentence = re.sub(r'[_]+|[-]{2,}', '***mask***', sentence)

        choices = tracker.get_slot("choices")
        if isinstance(choices, str):
            choices = re.split(r'\([A-Da-d]\)|[A-Da-d]\s+?\.|\n+|,', choices)
        elif isinstance(choices, list) and len(choices) == 1:
            choices = re.split(r'\([A-Da-d]\)|[A-Da-d]\s+?\.|\n+|,', choices[0])
        choices = [x for x in choices if x.strip()]
        bot_answer_choice = fb.rank(sentence, options=choices)[0]
        dispatcher.utter_message(
            template="utter_bot_choice", 
            bot_choice=bot_answer_choice
        )

        return [AllSlotsReset()]


class ActionRequestToTracau(Action):
    def name(self) -> Text:
        return "action_request_to_tracau"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            entities = [entity['value'] for entity in tracker.latest_message['entities']
                        if entity['entity'] == 'phrase_to_tracau']
            query = max(entities, key=lambda x: len(x))
            query = query.strip('"').strip()
        except:
            dispatcher.utter_message(
                "Sorry, I cannot find out what phrase you are trying to refer :(")
            dispatcher.utter_message(
                "Please put the phrase after a colon, or wrap the phrase in double quotation mark \" \" or in any kind of brackets so I can pick it up.")
            return [AllSlotsReset()]
        url = "https://api.tracau.vn/WBBcwnwQpV89/s/" + query + "/en"
        try:
            respond = rq.get(url).json()
            html = respond['tratu'][0]['fields']['fulltext']
        except:
            lemmatizer = WordNetLemmatizer()
            query = lemmatizer.lemmatize(query)
            url = "https://api.tracau.vn/WBBcwnwQpV89/s/" + query + "/en"
            respond = rq.get(url).json()
            try:
                html = respond['tratu'][0]['fields']['fulltext']
            except:
                list_sentences = [sent['fields']
                                  for sent in respond['sentences']]
                list_return = []
                for sentence in list_sentences:
                    en = BeautifulSoup(
                        sentence['en'], features="lxml").getText()
                    en = " ".join(en.split())
                    vi = sentence['vi']
                    vi = " ".join(vi.split())
                    list_return.append('- '+en+'\n'+vi)
                dispatcher.utter_message("Hmm... Although I can't find any definition of the phrase \"{0}\" but I managed to find some examples for references\n \n".format(
                    query))
                for item in list_return:
                    dispatcher.utter_message(item)
                return [AllSlotsReset()]
        parsed_html = BeautifulSoup(html, features="lxml")
        if parsed_html.find("article", {'data-tab-name': "Ngữ pháp"}):
            l = [x for x in parsed_html.find("article", {
                                             'data-tab-name': "Ngữ pháp"}).find("div", {'class': "dict--content"}).children]
        elif parsed_html.find("article", {'data-tab-name': "Thành ngữ"}):
            l = [x for x in parsed_html.find("article", {
                                             'data-tab-name': "Thành ngữ"}).find("div", {'class': "dict--content"}).children]
        elif parsed_html.find("article", {'data-tab-name': "Anh - Anh"}):
            l = [x for x in parsed_html.find("article", {
                                             'data-tab-name': "Anh - Anh"}).find("div", {'class': "dict--content"}).children]
        else:
            dispatcher.utter_message(
                text="Sorry, I can't find the phrase {0} in the database".format(query))
            return [AllSlotsReset()]
        l2 = []
        for element in l:
            if element.name == 'dtrn':
                definition_text = element.find(
                    'dtrn', text=True, recursive=False)
                l2.append(definition_text)
                for child in element.children:
                    if child.string:
                        l2.append(child.string)
            else:
                l2.append(element.getText())

        dispatcher.utter_message(text='\n'.join(
            [string for string in l2 if string]))
        return [AllSlotsReset()]


class ActionRequestToTracauFromSlot(Action):
    def name(self) -> Text:
        return "action_request_to_tracau_from_slot"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        query = tracker.get_slot("phrase_to_tracau").strip(
            '"').strip()
        url = "https://api.tracau.vn/WBBcwnwQpV89/s/" + query + "/en"
        try:
            respond = rq.get(url).json()
            html = respond['tratu'][0]['fields']['fulltext']
        except:
            lemmatizer = WordNetLemmatizer()
            query = lemmatizer.lemmatize(query)
            url = "https://api.tracau.vn/WBBcwnwQpV89/s/" + query + "/en"
            respond = rq.get(url).json()
            try:
                html = respond['tratu'][0]['fields']['fulltext']
            except:
                list_sentences = [sent['fields']
                                  for sent in respond['sentences']]
                list_return = []
                for sentence in list_sentences:
                    en = BeautifulSoup(
                        sentence['en'], features="lxml").getText()
                    en = " ".join(en.split())
                    vi = sentence['vi']
                    vi = " ".join(vi.split())
                    list_return.append('- '+en+'\n'+vi)
                dispatcher.utter_message("Hmm... Although I can't find any definition of the phrase \"{0}\" but I managed to find some examples for references\n \n".format(
                    query))
                for item in list_return:
                    dispatcher.utter_message(item)
                return [SlotSet("phrase_to_tracau", None)]
        parsed_html = BeautifulSoup(html, features="lxml")
        if parsed_html.find("article", {'data-tab-name': "Ngữ pháp"}):
            l = [x for x in parsed_html.find("article", {
                                             'data-tab-name': "Ngữ pháp"}).find("div", {'class': "dict--content"}).children]
        elif parsed_html.find("article", {'data-tab-name': "Thành ngữ"}):
            l = [x for x in parsed_html.find("article", {
                                             'data-tab-name': "Thành ngữ"}).find("div", {'class': "dict--content"}).children]
        elif parsed_html.find("article", {'data-tab-name': "Anh - Anh"}):
            l = [x for x in parsed_html.find("article", {
                                             'data-tab-name': "Anh - Anh"}).find("div", {'class': "dict--content"}).children]
        else:
            dispatcher.utter_message(
                text="Sorry, I can't find the phrase {0} in the database".format(query))
            return [SlotSet("phrase_to_tracau", None)]
        l2 = []
        for element in l:
            if element.name == 'dtrn':
                definition_text = element.find(
                    'dtrn', text=True, recursive=False)
                l2.append(definition_text)
                for child in element.children:
                    if child.string:
                        l2.append(child.string)
            else:
                l2.append(element.getText())

        dispatcher.utter_message(text='\n'.join(
            [string for string in l2 if string]))
        return [AllSlotsReset()]


class ActionWritingCheck(Action):
    def name(self) -> Text:
        return "action_writing_check"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Remember to fix /python3.8/site-packages/rasa/core/channels/console.py line DEFAULT_STREAM_READING_TIMEOUT_IN_SECONDS = 10 to 30
        # After install gingerit, go to /python3.8/site-packages/gingerit/gingerit.py and add <"end": end,> to line 51
        try:
            entities = [entity['value']
                        for entity in tracker.latest_message['entities'] if entity['entity'] == 'text']
            text = max(entities, key=lambda x: len(x))
            text = text.strip('"').strip()
        except:
            dispatcher.utter_message(
                "Please wrap the text in double quotation mark \" \"")
            return [AllSlotsReset()]
        try:
            dispatcher.utter_message('Here are some mistakes that I found:')
            errors = print_errors(text, language_tool, gg)
            dispatcher.utter_message(errors)
            dispatcher.utter_message("Here is the text after correction:")
            corrected_text = correct_language_tool(
                correct_gingerit(text, gg), language_tool)
            dispatcher.utter_message(corrected_text)
            dispatcher.utter_message(
                "Please keep in mind that these corrections may be wrong and should only be used as a reference")
        except:
            dispatcher.utter_message(
                "Hooray! I did not find any errors in your provided text.")
        return [AllSlotsReset()]


class ActionWritingCheckFromSlot(Action):
    def name(self) -> Text:
        return "action_writing_check_from_slot"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Remember to fix /python3.8/site-packages/rasa/core/channels/console.py line DEFAULT_STREAM_READING_TIMEOUT_IN_SECONDS = 10 to 30
        # After install gingerit, go to /python3.8/site-packages/gingerit/gingerit.py and add <"end": end,> to line 51
        text = tracker.get_slot("text_for_correction").strip(
            '"').strip()
        try:
            response = rq.get(text)
            dispatcher.utter_message(
                "Sorry, I don't support images yet! But let's wait for Pham Ngoc Man to implement an OCR function then I'll be ready!")
            #dispatcher.utter_message("Your url is: {0}".format(text))
        except:
            try:
                dispatcher.utter_message(
                    'Here are some mistakes that I found:')
                errors = print_errors(text, language_tool, gg)
                dispatcher.utter_message(errors)
                dispatcher.utter_message("Here is the text after correction:")
                corrected_text = correct_language_tool(
                    correct_gingerit(text, gg), language_tool)
                dispatcher.utter_message(corrected_text)
                dispatcher.utter_message(
                    "Please keep in mind that these corrections may be wrong and should only be used as a reference")
            except:
                dispatcher.utter_message(
                    "Hooray! I did not find any errors in your provided text.")
        return [AllSlotsReset()]
