# This file contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

from gingerit.gingerit import GingerIt
import language_tool_python
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from rasa_sdk.events import AllSlotsReset

from fitbert import FitBert
from transformers import (
    BlenderbotSmallTokenizer,
    BlenderbotSmallForConditionalGeneration,
    ElectraForMaskedLM,
    ElectraTokenizer,
    pipeline
)
import requests as rq
import re
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import tokenize
nltk.download('wordnet')
nltk.download('punkt')
global language_tool
global gg
language_tool = language_tool_python.LanguageTool('en-US')
gg = GingerIt()


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


def split_into_sentences(text, string=True):
    sentences = tokenize.sent_tokenize(text)
    if string:
        return "\n".join(sentences)
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
            (error['correct'], error['start'], error['end']) for error in subtext_gg_errors))
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
        if latest_user_message.casefold() != correction.casefold():
            bot_reply = bot_reply + \
                "\n(Small note: I found some mistakes in your message! Did you mean \"{0}\"?)".format(
                    correction)
        dispatcher.utter_message(text=bot_reply)
        return [AllSlotsReset()]


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
        entities = tracker.latest_message['entities']
        sentence_value = next(
            (item['value'] for item in entities if item['entity'] == 'sentence'), '').strip()
        if sentence_value == '':
            dispatcher.utter_message(
                "Please enter with the right syntax, for example: Solve this TOEIC reading question: <sentence>(A)answer A(B)answer B(C)answer C(D)answer D")
            return [AllSlotsReset()]
        sentence_value = re.sub(r'_+', '***mask***', sentence_value)
        answers = [
            next((item['value']
                 for item in entities if item['entity'] == 'answer_a'), '').strip(),
            next((item['value']
                 for item in entities if item['entity'] == 'answer_b'), '').strip(),
            next((item['value']
                 for item in entities if item['entity'] == 'answer_c'), '').strip(),
            next((item['value']
                 for item in entities if item['entity'] == 'answer_d'), '').strip()
        ]
        bot_choice = fb.rank(sentence_value, options=answers)[0]
        dispatcher.utter_message(text=f"My guess is: \"{bot_choice}\"")

        return [AllSlotsReset()]


class ActionRequestToTracau(Action):
    def name(self) -> Text:
        return "action_request_to_tracau"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            query = tracker.latest_message['entities'][-1]['value'].strip(
                '"').strip()
        except:
            dispatcher.utter_message(
                "Please wrap the phrase in double quotation mark \" \"")
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
                    list_return.append(en+'\n'+vi)
                message = "Hmm... Although I can't find any definition of the phrase \"{0}\" but I managed to find some examples for references:\n \n".format(
                    query)
                mesage = message + \
                    '\n---------------------\n'.join(list_return)
                dispatcher.utter_message(mesage)
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
                    list_return.append(en+'\n'+vi)
                message = "Hmm... Although I can't find any definition of the phrase \"{0}\" but I managed to find some examples for references\n \n".format(
                    query)
                mesage = message + \
                    '\n---------------------\n'.join(list_return)
                dispatcher.utter_message(mesage)
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
            text = tracker.latest_message['entities'][0]['value'].strip(
                '"').strip()
        except:
            dispatcher.utter_message(
                "Please wrap the text in double quotation mark \" \"")
            return [AllSlotsReset()]
        message = f'Here are some mistakes that I found:\n\n'
        message = message + "------------\n\n"
        message = message + print_errors(text, language_tool, gg) + '\n\n'
        message = message + "------------\n\n"
        message = message + f"Here is the text after correction:\n\n"
        message = message + "------------\n\n"
        message = message + \
            correct_gingerit(correct_language_tool(
                text, language_tool), gg) + "\n\n"
        message = message + "------------\n\n"
        message = message + "Please keep in mind that these corrections may be wrong and should only be used as a reference"
        dispatcher.utter_message(message)
        return [AllSlotsReset()]


class ActionWritingCheckFromSlot(Action):
    def name(self) -> Text:
        return "action_writing_check_from_slot"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Remember to fix /python3.8/site-packages/rasa/core/channels/console.py line DEFAULT_STREAM_READING_TIMEOUT_IN_SECONDS = 10 to 30
        # After install gingerit, go to /python3.8/site-packages/gingerit/gingerit.py and add <"end": end,> to line 51
        text = tracker.get_slot("text").strip(
            '"').strip()
        message = f'Here are some mistakes that I found:\n'
        message = message + "------------\n\n"
        message = message + print_errors(text, language_tool, gg) + '\n'
        message = message + "------------\n \n"
        message = message + f"Here is the text after correction:\n"
        message = message + "------------\n"
        message = message + \
            correct_gingerit(correct_language_tool(
                text, language_tool), gg) + "\n"
        message = message + "------------\n"
        message = message + "Please keep in mind that these corrections may be wrong and should only be used as a reference"
        dispatcher.utter_message(message)
        return [AllSlotsReset()]
