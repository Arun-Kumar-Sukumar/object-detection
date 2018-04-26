# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:41:00 2018

@author: 692908, 668900
"""


from __future__ import print_function

# text summarisation
from gensim.summarization import summarize
from gensim.summarization import keywords
import docx
import warnings

# covert pdf to text
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

# speech recognition
import speech_recognition as sr
import webbrowser as wb
from langdetect import detect
from googletrans import Translator
import pyttsx3
from playsound import playsound

# object detection
import numpy as np
import os
import sys
import cv2
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
from utils import ops as utils_ops

# setting
engine = pyttsx3.init()
rate = 200
engine.setProperty('rate', rate)
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# go upward to folder tree
sys.path.append("..")

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


# What model to download
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Load the Frozen Tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Label Map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper Function
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# DETECTION
# For testing with image, add image to TEST_IMAGE_PATHS.

# PATH_TO_TEST_IMAGES_DIR = 'test_images'
# TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 4)]

# Size, in inches, of the output images.
# IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                  tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                  'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


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
    pagenos = set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                  check_extractable=True):
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
          "6 - Object recognition")
    engine.say('Tell me what do you want to do ? ')
    engine.runAndWait()


def end():
    print("Do you want to do an other thing ? yes/no")
    heard = False
    while not heard:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            r.energy_threshold += 280
            print("Please say your CHOICE after the tone : ")
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
            print("Please say your CHOICE after the tone   : ")
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

    elif keywordschoice == 'no':
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
        print("Please say your CHOICE after the tone : ")
        playsound('audio.wav')
        audio = r.listen(source)
    try:
        choice = r.recognize_google(audio)
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
            engine.say("I heard that you said:  " + choice + ". Is it right ?")
            engine.runAndWait()
            print("I heard that you said:  " + choice)
            print("Is it right ?")
            print("Please say your CHOICE after the tone   : ")
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

        if "Microsoft" in choice:
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
                    text = r.recognize_google(audio)
                    print("You said " + text)
                    language_id = detect(text)
                    print("language  is : " + language_id)
                    engine.say("the language is " + language_id)
                    engine.runAndWait()
                    heard = True
                except Exception as e:
                    engine.say('I do not understand your answer please try again')
                    engine.runAndWait()
            engine.say('Do you want to do something else ?')
            engine.runAndWait()
            end()

        elif "speech" in choice:

            heard = False
            while not heard:
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source)
                    r.energy_threshold += 280
                    engine.say('Say what you want translate :')
                    engine.runAndWait()
                    print("Say what you want to translate after the tone :")
                    playsound('audio.wav')
                    audio = r.listen(source)
                try:
                    text = r.recognize_google(audio)
                    print("You said " + text)
                    translator = Translator()
                    g = translator.translate(text, dest='fr')
                    print("the translation in french of what you said is : " + g.text)
                    engine.say("the translation in french of what you said is : " + g.text)
                    engine.runAndWait()
                    heard = True
                except Exception as e:
                    engine.say('I do not understand your answer please try again')
                    engine.runAndWait()
            engine.say('Do you want to do something else ?')
            engine.runAndWait()
            end()

        elif "search" in choice:

            heard = False
            while not heard:
                r = sr.Recognizer()
                url = "https://en.wikipedia.org/wiki/"
                url2 = "http://google.com/search?q= "
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source)
                    r.energy_threshold += 280
                    engine.say('Say what you want to search on the web :')
                    engine.runAndWait()
                    print("Say what you want to search on the web after the tone  :")
                    playsound('audio.wav')
                    audio = r.listen(source)
                try:
                    text = r.recognize_google(audio)
                    print("You said " + text)
                    wb.open_new(url + text)
                    wb.open_new(url2 + text)
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

        elif "object" in choice:
            engine.say('Please show object to the camera!')
            engine.runAndWait()

            # ============================================ #
            print("Reading from the webcam...")

            stream = cv2.VideoCapture(0)

            grabbed = True

            while (grabbed):
                # image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                # image_np = load_image_into_numpy_array(image)

                grabbed, image_np = stream.read()

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                # image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                output_dict = run_inference_for_single_image(image_np, detection_graph)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=8)

                cv2.imshow('image', cv2.resize(image_np, (600, 400)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    cv2.destroyAllWindows()
                    stream.release()

                id = output_dict['detection_classes'][0]

                print("Object is the {}".format(category_index[id]['name']))
                print("Score {}%".format(round(output_dict['detection_scores'][0] * 100)))

                engine.say("Object is the {}".format(category_index[id]['name']))
                engine.say("Score {}%".format(round(output_dict['detection_scores'][0] * 100)))
        else:
            engine.say('The choice you made doesnt exist. PLease try again')
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


engine.say('Welcome to the Cognizant AI Platform. My name is Lara...' +
           'I can summarize a Microsoft word or pdf file, ' +
           'detect a language, translate a speech ' +
           'search on the web the topic of your choice ' +
           'or recognize object via webcam.'
           )

engine.runAndWait()
start()
main()






