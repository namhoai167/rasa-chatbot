# This file contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

import cv2
from skimage import io
import pytesseract
import matplotlib.pyplot as plt
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
import re
import requests as rq
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


def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    #gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(newImage, (9, 9), 0)
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(
        dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(newImage, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    #print (len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    #cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle
# Rotate the image around its center


def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(
        newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage


def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle - 90.0)


def ocr(img_url) -> str:
    image = io.imread(img_url)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh, image = cv2.threshold(image, 210, 230, cv2.THRESH_BINARY)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    image = deskew(image)
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    text = str(((pytesseract.image_to_string(image))))
    text = re.sub(r"[\n]{2,}", ". ", text)
    text = re.sub(r"\n(?=[A-Z])", ". ", text)
    text = re.sub(r"\n(?=[^A-Z])", " ", text)
    return text


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
        correction = correct_gingerit(latest_user_message, gg)
        correction = correct_language_tool(correction, language_tool)
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
        sentence = next(
            (item['value'] for item in entities if item['entity'] == 'sentence'), '').strip()
        if sentence == '':
            dispatcher.utter_message(
                "Please enter with the right syntax, for example: Solve this TOEIC reading question: <sentence>(A)answer A(B)answer B(C)answer C(D)answer D")
            return [AllSlotsReset()]
        sentence = re.sub(r'_+', '***mask***', sentence)
        choices = [
            next((item['value']
                 for item in entities if item['entity'] == 'answer_a'), '').strip(),
            next((item['value']
                 for item in entities if item['entity'] == 'answer_b'), '').strip(),
            next((item['value']
                 for item in entities if item['entity'] == 'answer_c'), '').strip(),
            next((item['value']
                 for item in entities if item['entity'] == 'answer_d'), '').strip()
        ]
        bot_answer_choice = fb.rank(sentence, options=choices)[0]
        dispatcher.utter_message(text=f"My guess is: \"{bot_answer_choice}\"")

        return [AllSlotsReset()]


class ActionSolveMultipleChoiceSentenceCompletionWithListOfChoices(Action):

    def name(self) -> Text:
        return "action_solve_multiple_choice_sentence_completion_with_list_of_choices"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        ELECTRA_PATH = './electra-small-generator'
        ELECTRAmodel = ElectraForMaskedLM.from_pretrained(ELECTRA_PATH)
        ELECTRAtokenizer = ElectraTokenizer.from_pretrained(ELECTRA_PATH)
        fb = FitBert(model=ELECTRAmodel, tokenizer=ELECTRAtokenizer)

        sentence = tracker.get_slot("sentence")
        choices = tracker.get_slot("choices")

        # Use regex to replace mask for fitbert sentence
        sentence = re.sub(r'[-_]+', '***mask***', sentence)

        # Reformat choices list for fitbert
        if choices.count(',') >= 3:
            choices = choices.rsplit(',')
            choices = [s.strip() for s in choices]
        else:
            choices = re.split(
                r'\s*?\([A-Da-d]\)|[A-Da-d]\.|[A-Da-d]\s+?\.*', choices)
            choices.pop(0)
            choices = [s.strip() for s in choices]

        bot_answer_choice = fb.rank(sentence, options=choices)[0]
        dispatcher.utter_message(text=f"My guess is: \"{bot_answer_choice}\"")

        return [
            SlotSet("sentence", None),
            SlotSet("choices", None)
        ]


'''
class ActionSolveMultipleChoiceSentenceCompletionWithListOfSentenceAndchoices(Action):

    def name(self) -> Text:
        return "action_solve_multiple_choice_sentence_completion_with_list_of_sentence_and_choices"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        ELECTRA_PATH = './electra-small-generator'
        ELECTRAmodel = ElectraForMaskedLM.from_pretrained(ELECTRA_PATH)
        ELECTRAtokenizer = ElectraTokenizer.from_pretrained(ELECTRA_PATH)
        fb = FitBert(model=ELECTRAmodel, tokenizer=ELECTRAtokenizer)

        sentence_and_choices = tracker.get_slot("sentence_and_choices")

        # Extract sentence from string with regex
        sentence = re.match(
            r'.*[-_]+.*?(|\.*?)((?=\s*\w+(,\s*\w+){1,3}$)|(?=\s*?[Aa]\.)|(?=\s*?\([Aa]\)))', sentence_and_choices).group().strip()

        # Extract choices
        choices = sentence_and_choices.replace(sentence, '').strip()

        # Reformat sentence and choices for fitbert
        # Use regex to replace mask
        sentence = re.sub(r'[-_]+', '***mask***', sentence)

        if choices.count(',') >= 3:
            choices = choices.rsplit(',', 3)
            choices = [s.strip() for s in choices]
        else:
            choices = re.split(
                r'\([A-Da-d]\)|\.[A-Da-d]|[A-Da-d]\.|[A-Da-d]\s', choices)
            choices.pop()
            choices = [s.strip() for s in choices]

        bot_answer_choice = fb.rank(sentence, options=choices)[0]
        dispatcher.utter_message(text=f"My guess is: \"{bot_answer_choice}\"")

        return [
            SlotSet("sentence_and_choices", None)
        ]
'''


'''class ActionRequestToTracau(Action):
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
'''


class ActionRequestToTracau(Action):
    def name(self) -> Text:
        return "action_request_to_tracau"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            query = tracker.get_slot("phrase_to_tracau").strip(
                '"').strip().strip('(').strip(')').strip('[').strip(']').strip("'").strip('{').strip('}')
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
            elif parsed_html.find("article", {'data-tab-name': "Anh - Việt"}):
                l = [x for x in parsed_html.find("article", {
                    'data-tab-name': "Anh - Việt"}).find("div", {'class': "dict--content"}).children]
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
                    try:
                        l2.append(element.getText())
                    except:
                        l2.append(element)

            dispatcher.utter_message(text='\n'.join(
                [string for string in l2 if string]))
        except:
            dispatcher.utter_message("Sorry, there's an error occured :(")
        return [AllSlotsReset()]


'''class ActionWritingCheck(Action):
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
'''


class ActionWritingCheck(Action):
    def name(self) -> Text:
        return "action_writing_check"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Remember to fix /python3.8/site-packages/rasa/core/channels/console.py line DEFAULT_STREAM_READING_TIMEOUT_IN_SECONDS = 10 to 30
        try:
            text = tracker.get_slot("text_for_correction").strip()
            #dispatcher.utter_message('Your image url: ' + text)
            try:
                scanned_text = ocr(text)
                #dispatcher.utter_message('Scanned text: ' + scanned_text)
                try:
                    dispatcher.utter_message(
                        'Here are some mistakes that I found:')
                    errors = print_errors(scanned_text, language_tool, gg)
                    dispatcher.utter_message(errors)
                    dispatcher.utter_message(
                        "Here is the text after correction:")
                    corrected_text = correct_language_tool(
                        correct_gingerit(scanned_text, gg), language_tool)
                    dispatcher.utter_message(corrected_text)
                    dispatcher.utter_message(
                        "Please keep in mind that these corrections may be wrong and should only be used as a reference")
                except:
                    dispatcher.utter_message(
                        "Hooray! I did not find any errors in your provided text.")
            except:
                try:
                    errors = print_errors(text, language_tool, gg)
                    dispatcher.utter_message(
                        'Here are some mistakes that I found:')
                    dispatcher.utter_message(errors)
                    corrected_text = correct_language_tool(
                        correct_gingerit(text, gg), language_tool)
                    dispatcher.utter_message(
                        "Here is the text after correction:")
                    dispatcher.utter_message(corrected_text)
                    dispatcher.utter_message(
                        "Please keep in mind that these corrections may be wrong and should only be used as a reference")
                except:
                    dispatcher.utter_message(
                        "Hooray! I did not find any errors in your provided text.")
        except:
            dispatcher.utter_message("Sorry, there's an error occured :(")
        return [AllSlotsReset()]
