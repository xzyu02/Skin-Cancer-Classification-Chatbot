import nltk
import numpy as np
import string # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


f=open('skin cancer.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()# converts to lowercase

sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences
word_tokens = nltk.word_tokenize(raw)# converts to list of words

lemmer = nltk.stem.WordNetLemmatizer()
#
def LemTokens(tokens):
   return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
   return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey","ai","next")

GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me","You're welcome, this is my job","You'd better talk with the doctor and you need further treatment"]

def greeting(sentence):
   for word in sentence.split():
       for i in range(len(GREETING_INPUTS)):
            if word.lower() == GREETING_INPUTS[i]:
                return GREETING_RESPONSES[i]

def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

# cnn part
import numpy as np
import pandas as pd
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError

# load the model
loaded_model = models.load_model('728cnn.h5')

def get_class(str):
    try:
        test_image = Image.open(str) 
        Image.open
    except BaseException:
        return 'false'
    else:
        test_image = test_image.resize((28, 28))
        test_image = image.img_to_array(test_image)
        test_image = test_image.reshape(-1, 28, 28, 3)
        test_image = test_image/255
        # predict the result
        result = loaded_model.predict(test_image)
        # cancer classes
        classes = {4: ('nv', 'melanocytic nevi'),
            6: ('mel', 'melanoma'),
            2: ('bkl', 'benign keratosis-like lesions'), 
            1: ('bcc' , 'basal cell carcinoma'),
            5: ('vasc', 'pyogenic granulomas and hemorrhage'),
            0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
            3: ('df', 'dermatofibroma')}
        return classes.get(np.argmax(result))[1]

def is_path(str):
    pics = ['bmp','png','jpg','jpeg','tiff','gif', 'pcx', 'tga', 'exif', 'fpx', 'svg','psd','cdr','pc','dxf','ufo','eps','ai','raw']
    if (str.find('.') == -1):
        return -1
    elif(str[str.rfind('.')+1::] in pics):
        return str
    else:
        return -1
    #/Users/yuxizheng/xizheng/proj_past_7007/Week_5/test_pics_with_label/ISIC_0034299_bcc_1.jpg

def chat(user_response):
    rob_response = "DOCTOR STRANGE: Hi! I am a chatbot to tell you the diagnosis, please show me your skin picture."
    # check input is path
    path = is_path(user_response)
    if (path == -1):
        user_response = user_response.lower()
    # process user response
    if (user_response != 'bye'):
        if (user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            rob_response = "DOCTOR STRANGE: You are welcome, this is my job"
        elif (path != -1):
            r = get_class(path)
            rob_response = "DOCTOR STRANGE: Please wait few second, your picture is processing."
            if (r == 'false'):
                rob_response ="DOCTOR STRANGE: Sorry, cannot find the picture through your input path. Please try again."
            else:
                rob_response ="DOCTOR STRANGE: The diagnosis shows that you are having " + r + '\n' + "DOCTOR STRANGE: " + response(r)
                sent_tokens.remove(r)
        else:
            if (greeting(user_response) != None):
                rob_response ="DOCTOR STRANGE: " + greeting(user_response)
            else:
                rob_response ="DOCTOR STRANGE: " +response(user_response)

                sent_tokens.remove(user_response)
    else:
        rob_response ="DOCTOR STRANGE: Bye! take care."
    return rob_response
