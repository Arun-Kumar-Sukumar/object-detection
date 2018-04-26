# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:41:00 2018

@author: 692908
"""

from __future__ import print_function

# libraries for summarization algo
from gensim.summarization import summarize
from gensim.summarization import keywords
import docx
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# libraires to covert pdf to text #
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

# libraires to speech recognition
import speech_recognition as sr
import webbrowser as wb

from langdetect import detect
from googletrans import Translator
import pyttsx3

from playsound import playsound
engine = pyttsx3.init()
rate = 350
engine.setProperty('rate', rate)


# function convert pdf to text
def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'ascii'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text


def start():
    print("1 - Summarize a Microsoft Word document \n" +
          "2 - Summarize a PDF\n" +
          "3 - Detect a language \n" +
          "4 - Translate a Speech \n" +
          "5 - Search on the web \n" +
          "6 - Object detection")
    engine.say('What do you want to do today? ' +
               'I can summarize a Microsoft word or pdf file, ' +
               'detect a langage, translate a speech or search ' +
               'on the web the topic of your choice. Please let me know ' +
               'what to do for you')
    engine.runAndWait()   
    

def end():    
    endchoice = "Do you want to do an other thing ? yes/no "
    heard = False
    while not heard:
        r = sr.Recognizer()
        with sr.Microphone() as source: 
            r.adjust_for_ambient_noise(source)
            r.energy_threshold += 280
            print("Please say your CHOICE  : ")
            playsound('audio.wav')
            audio = r.listen(source)
            try:
                endchoice = r.recognize_google(audio)
                heard = True
            except Exception as e:
                engine.say('I do not understand your answer please try again')
                engine.runAndWait()
    print(endchoice) 
    
    if endchoice == 'yes':
        start()
        main()
    elif endchoice == 'no':
        engine.say('Thanks a lot. Please feel free to ask me if you need anything else')
        engine.runAndWait()
    else:
        engine.say('I do not understand your answer, please try again')
        engine.runAndWait()
        end()    

def choosetopwords():
    print("Would you like to see the top 3 keywords of the text? (yes/no)")
    
    heard = False
    while not heard:
        r = sr.Recognizer()
        with sr.Microphone() as source: 
            r.adjust_for_ambient_noise(source)
            r.energy_threshold += 280
            print("Please say your CHOICE  : ")
            playsound('audio.wav')
            audio = r.listen(source)
            try:
                keywordschoice = r.recognize_google(audio)
                heard = True
            except Exception as e:
                engine.say('I do not understand your answer please try again')
                engine.runAndWait()
    print(keywordschoice) 
    
    if keywordschoice == 'yes':
            print('\nKeywords: \n')
            print(keywords(docText, words=2))
            engine.say('Here are the top 3 keywords, do you want to do something else ?')
            engine.runAndWait()
            end()
        
    elif keywordschoice =='no':
            engine.say('Do you want to summarize an other text?')
            engine.runAndWait()
            end()
    else: 
            engine.say('I do not understand your answer please try again')
            engine.runAndWait()
            choosetopwords()


def topwords():    
    finishchoice = input("Tell me when you finished reading (type ok)")
    if finishchoice == 'ok':
        engine.say('Would you like to see the top 3 keywords of the text?')
        engine.runAndWait()
        
        choosetopwords()
        
    else:
        engine.say('I do not understand your answer please try again')
        engine.runAndWait()
        topwords()


global docText


def main():
    
    r = sr.Recognizer()
    with sr.Microphone() as source:
     r.adjust_for_ambient_noise(source)
     r.energy_threshold += 280
     print("Please say your CHOICE  : ") 
     playsound('audio.wav')
     audio = r.listen(source)
    try:
      choice=r.recognize_google(audio)
    except Exception as e:
      engine.say('I do not understand your answer please try again')
      engine.runAndWait()
      main()
  
    heard = False
    while not heard:
        r = sr.Recognizer()
        with sr.Microphone() as source: 
            r.adjust_for_ambient_noise(source)
            r.energy_threshold += 280
            engine.say("I heard that you said:  " +choice + ". Is it right ?")
            engine.runAndWait()
            print("I heard that you said:  "+choice)
            print("Is it right ?")
            print("Please say your CHOICE  : ")
            playsound('audio.wav')
            audio = r.listen(source)
            try:
                response = r.recognize_google(audio)
                heard = True
            except Exception as e:
                engine.say('I do not understand your answer please try again')
                engine.runAndWait()
    print(response) 
    
    if "yes" in response:
        if "Microsoft" in choice :
            engine.say('Please enter the name of the word file')
            engine.runAndWait()
            connected = False
            while not connected:
                try:
                    filepath = input("Please type the name of the word file : ")
                    filepath = filepath + ".docx"
                    document = docx.Document(filepath)
                    connected = True
                except:
                    engine.say('You made a mistake, I did not found this file, please try again')
                    engine.runAndWait()

            global docText
            docText = '\n\n'.join([paragraph.text for paragraph in document.paragraphs])
            print('\nSummary: \n')
            print(summarize(docText, ratio=0.2))
            engine.say('my summary will come in a second : type ok when you finished reading')
            engine.runAndWait()
            topwords()
                        
        elif "PDF" in choice:
            engine.say('Please enter the name of the PDF file')
            engine.runAndWait()
            connected = False
            while not connected:
                try:
                    filepath = input("Please enter the name of the PDF file : ")
                    filepath = filepath + ".pdf"
                    text = convert_pdf_to_txt(filepath)
                    connected = True
                except:
                    engine.say('You made a mistake, I did not found this file, please try again')
                    engine.runAndWait()


            docText = text.replace("\n", " ")
            print('\nSummary: \n')
            print(summarize(docText, ratio=0.2))
            engine.say('my summary will come in a second : tell me when you finished reading')
            engine.runAndWait()
            topwords()
        elif "detect" in choice:
            heard = False
            while not heard:
                r = sr.Recognizer()
                with sr.Microphone() as source: 
                    r.adjust_for_ambient_noise(source)
                    r.energy_threshold += 280
                    engine.say('Tell me what you want to detect please')
                    engine.runAndWait()
                    print("Tell me what you want to detect please : ")
                    playsound('audio.wav')
                    audio = r.listen(source)
                try:
                    text= r.recognize_google(audio)
                    print("You said "+text)
                    language_id = detect(text)
                    print("laguage  is : "+language_id)
                    engine.say("the language is "+language_id)
                    engine.runAndWait()
                    heard = True
                except Exception as e:
                    engine.say('I do not understand your answer please try again')
                    engine.runAndWait()
            end()            
                               
        elif "speech" in choice:
            heard = False
            while not heard:
                r = sr.Recognizer()
                with sr.Microphone() as source: 
                    r.adjust_for_ambient_noise(source)
                    r.energy_threshold += 280
                    engine.say('Say what you want transalte :')
                    engine.runAndWait()
                    print("Say what you want to translate :")
                    playsound('audio.wav')
                    audio = r.listen(source)
                try:
                    text= r.recognize_google(audio)
                    print("You said " + text)
                    translator = Translator()
                    g=translator.translate(text, dest='fr')
                    print("the translation in french of what you said is : "+g.text)
                    engine.say("the translation in french of what you said is : "+g.text)
                    engine.runAndWait()
                    heard = True
                except Exception as e:
                    engine.say('I do not understand your answer please try again')
                    engine.runAndWait()
            end()   
        elif "search" in choice:
            heard = False
            while not heard:
                r = sr.Recognizer()
                url="https://en.wikipedia.org/wiki/"
                with sr.Microphone() as source: 
                    r.adjust_for_ambient_noise(source)
                    r.energy_threshold += 280
                    engine.say('Say what you want to search on the web :')
                    engine.runAndWait()
                    print("Say what you want to search on the web :")
                    playsound('audio.wav')
                    audio = r.listen(source)
                try:
                    text= r.recognize_google(audio)
                    print("You said "+text)
                    wb.open_new(url+text)
                    heard = True
                except Exception as e:
                    engine.say('I do not understand your answer please try again')
                    engine.runAndWait()
            engine.say('Type ok when you finished reading the web page')
            engine.runAndWait()
            finishchoice = input("Tell me when you finished reading the webpage (type ok)")
            if finishchoice == 'ok':
                engine.say('Do you want to do something else ?')
                engine.runAndWait()
                end()       
            else:
                engine.say('Even if you did not typed ok, i know you want to leave.Do you want to do something else ?')
                engine.runAndWait()
                end()

        else:
            engine.say('The choiche you made doesnt exist. PLease try again')
            engine.runAndWait()
            main()
        
    elif "no" in response:
         engine.say('Sorry then. I made a mistake. Please try again')
         engine.runAndWait()
         main()
         
    else:
         engine.say('I asked for yes or no. PLease try again')
         engine.runAndWait()
         main()
 
engine.say('Welcome to the Cognizant AI Platform. My name is Lara')
engine.runAndWait()
start()
main()
        
        
        
        


