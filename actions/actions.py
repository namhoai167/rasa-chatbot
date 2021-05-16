# This file contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

from fitbert import FitBert
from transformers import (
    BlenderbotSmallTokenizer,
    BlenderbotSmallForConditionalGeneration,
    ElectraForMaskedLM,
    ElectraTokenizer,
    pipeline
)
import requests as rq
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import tokenize
nltk.download('wordnet')
nltk.download('punkt')
import language_tool_python
from gingerit.gingerit import GingerIt

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
        inputs = BBTokenizer([latest_user_message], return_tensors='pt')
        reply_ids = BBModel.generate(**inputs)
        bot_reply = BBTokenizer.batch_decode(
            reply_ids, skip_special_tokens=True)[0]
        dispatcher.utter_message(text=str(bot_reply))
        return []


class ActionSolveMultipleChoiceSentenceCompletion(Action):

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

        return [
            SlotSet("sentence", None),
            SlotSet("answer_a", None),
            SlotSet("answer_b", None),
            SlotSet("answer_c", None),
            SlotSet("answer_d", None)
        ]


class ActionRequestToTracau(Action):
    def name(self) -> Text:
        return "action_request_to_tracau"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        query = tracker.latest_message['entities'][-1]['value'].strip(
            '"').strip()
        # dispatcher.utter_message(text=query)
        url = "https://api.tracau.vn/WBBcwnwQpV89/s/" + query + "/en"
        try:
            respond = rq.get(url).json()
            html = respond['tratu'][0]['fields']['fulltext']
        except:
            lemmatizer = WordNetLemmatizer()
            query = lemmatizer.lemmatize(query)
            url = "https://api.tracau.vn/WBBcwnwQpV89/s/" + query + "/en"
            respond = rq.get(url).json()
            html = respond['tratu'][0]['fields']['fulltext']
        parsed_html = BeautifulSoup(html)
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
            return []
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
        return []


def cut_paragraph(text, maximum_len=300):
    sentences = tokenize.sent_tokenize(text)
    len_for_each_sentence = [len(sentence) for sentence in sentences]
    splited_paragraph = []
    while len(sentences) > 0:
        current_len = 0
        for i in range(len(sentences)):
            current_len += len_for_each_sentence[i] + 1
            if current_len > maximum_len:
                splited_paragraph.append(" ".join([sentence for idx, sentence in enumerate(sentences) if idx < i]))
                del sentences[:i]
                del len_for_each_sentence[:i]
                break
            if i == len(sentences) - 1:
                splited_paragraph.append(" ".join([sentence for sentence in sentences]))
                return splited_paragraph
            
            
def split_into_sentences(text, string=True):
    sentences = tokenize.sent_tokenize(text)
    if string:
        return "\n\n".join(sentences)
    return sentences


def correct_gingerit(text, gg):
    splited_text = cut_paragraph(text)
    return split_into_sentences(" ".join([gg.parse(t)['result'] for t in splited_text]))


def correct_language_tool(text, language_tool):
    return(split_into_sentences(language_tool.correct(text)))



class ActionWritingCheck(Action):
    def name(self) -> Text:
        return "action_writing_check"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Remember to fix /python3.8/site-packages/rasa/core/channels/console.py line DEFAULT_STREAM_READING_TIMEOUT_IN_SECONDS = 10 to 30
        language_tool = language_tool_python.LanguageTool('en-US')
        gg = GingerIt()
        text = tracker.latest_message['entities'][0]['value'].strip(
            '"').strip()
        message = f"Here are two versions of your text after correction:\n\n"
        message = message + "------------\n\n"
        message = message + correct_language_tool(text, language_tool) + "\n\n"
        message = message + "------------\n\n"
        message = message + correct_gingerit(text, gg) + "\n\n"
        message = message + "------------\n\n"
        message = message + "Please keep in mind that these corrections may be wrong and should only be used as a preference"
        dispatcher.utter_message(message)
        return []