# rasa-chatbot

How to install and run on local
1/ Install requirements
pip install -r requirements.txt

2/ Install tesseract-ocr
sudo apt-get install tesseract-ocr

3/ Go to /home/user/.local/lib/python3.8/site-packages/rasa/core/channels/console.py and change DEFAULT_STREAM_READING_TIMEOUT_IN_SECONDS to 50

4/ Train model
rasa train

5/ Start chatting!
rasa shell
