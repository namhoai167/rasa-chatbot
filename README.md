# rasa-chatbot

How to setup and run locally on Ubuntu 20.04, Python 3.8.5 pre-installed
1. Install requirements\
`pip3 install -r requirements.txt`

1. Install Java, Tesseract, PyTorch, language pipeline of spacy
```
sudo apt-get install default-jre
sudo apt-get install tesseract-ocr
pip3 install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
python3 -m spacy download en_core_web_md
```
3. Go to `lib/python3.8/site-packages/rasa/core/channels/console.py` and change `DEFAULT_STREAM_READING_TIMEOUT_IN_SECONDS` to 50

4. Train model\
`rasa train`

5. Start 2 terminals
   1. One for running action server\
   `rasa run actions`
   1. One for chatting\
   `rasa shell`

[Chatbot function demo](https://www.youtube.com/playlist?list=PLpzLTT344JfnyGnPigiINg62yDJlFYvXn)