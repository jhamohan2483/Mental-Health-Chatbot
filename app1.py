from flask import Flask, flash, request,render_template, redirect,session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import os
from datetime import date, datetime
import nltk
nltk.download('popular')
nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('model.h5')
import json
import random
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
import re  # For regular expression matching

# translator pipeline for english to swahili translations

eng_swa_tokenizer = AutoTokenizer.from_pretrained("Rogendo/en-sw")
eng_swa_model = AutoModelForSeq2SeqLM.from_pretrained("Rogendo/en-sw")

eng_swa_translator = pipeline(
    "text2text-generation",
    model = eng_swa_model,
    tokenizer = eng_swa_tokenizer,
)

def translate_text_eng_swa(text):
    translated_text = eng_swa_translator(text, max_length=128, num_beams=5)[0]['generated_text']
    return translated_text

# translator pipeline for swahili to english translations

swa_eng_tokenizer = AutoTokenizer.from_pretrained("Rogendo/sw-en")
swa_eng_model = AutoModelForSeq2SeqLM.from_pretrained("Rogendo/sw-en")

swa_eng_translator = pipeline(
    "text2text-generation",
    model = swa_eng_model,
    tokenizer = swa_eng_tokenizer,
)

def translate_text_swa_eng(text):
  translated_text = swa_eng_translator(text,max_length=128, num_beams=5)[0]['generated_text']
  return translated_text


def get_lang_detector(nlp, name):
    return LanguageDetector()

nlp = spacy.load("en_core_web_sm")

Language.factory("language_detector", func=get_lang_detector)

nlp.add_pipe('language_detector', last=True)

intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
def getResponse(ints, intents_json):
    if ints: 
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result
    else:
        return "Sorry, I didn't understand that."

def chatbot_response(msg):
    doc = nlp(msg)
    detected_language = doc._.language['language']
    print(f"Detected language chatbot_response:- {detected_language}")
    
    chatbotResponse = "Loading bot response..........."

    if detected_language == "en":
        res = getResponse(predict_class(msg, model), intents)
        chatbotResponse = res
        print("en_sw chatbot_response:- ", res)
    elif detected_language == 'sw':
        translated_msg = translate_text_swa_eng(msg)
        res = getResponse(predict_class(translated_msg, model), intents)
        chatbotResponse = translate_text_eng_swa(res)
        print("sw_en chatbot_response:- ", chatbotResponse)

    return chatbotResponse


app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "database.db")}'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'
app.static_folder = 'static'

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fname = db.Column(db.String(100), nullable=False)
    lname = db.Column(db.String(100), nullable=True)
    dob = db.Column(db.Date, nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

    def __init__(self,username,password,fname,lname,dob):
        self.fname = fname
        self.lname = lname
        self.dob = dob
        self.username = username
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))

with app.app_context():
    db.create_all()
    print("✅ Database created or already exists.")

@app.route('/',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['username'] = user.username
            return redirect('/chatbot')
        else:
            return render_template('login.html',error='Invalid user')

    return render_template('login.html')


from datetime import datetime

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            print("📥 Form received:")

            fname = request.form.get('fname')
            lname = request.form.get('lname')

            # Check if names contain only letters
            if not fname.isalpha() or not lname.isalpha():
                flash("First and last names must contain only letters (A–Z).")
                return redirect('/signup')

            dob_str = request.form.get('dob')
            username = request.form.get('username')
            password = request.form.get('password')

            # Convert DOB to a date object
            dob = datetime.strptime(dob_str, '%Y-%m-%d').date()

            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

            if age < 12:
                flash("You must be at least 12 years old to register.")
                return redirect('/signup')  # Redirect back to signup page

            new_user = User(fname=fname, lname=lname, dob=dob, username=username, password=password)
            db.session.add(new_user)
            db.session.commit()

            # print("✅ New user registered:", username)
            flash("Signup successful! You can now log in.")
            return redirect('/')

        except Exception as e:
            db.session.rollback()
            print("❌ ERROR while registering user:", e)
            return render_template('signup.html', error="Registration failed: " + str(e))

    return render_template('signup.html')


@app.route('/chatbot')
def chatbot():
    if session['username']:
        user = User.query.filter_by(username=session['username']).first()
        return render_template('chatbot.html',user=user)
    
    return redirect('/')

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print("get_bot_response:- " + userText)

    doc = nlp(userText)
    detected_language = doc._.language['language']
    print(f"Detected language get_bot_response:- {detected_language}")

    bot_response_translate = "Loading bot response..........."  

    if detected_language == "en":
        bot_response_translate = userText  
        print("en_sw get_bot_response:-", bot_response_translate)
        
    elif detected_language == 'sw':
        bot_response_translate = translate_text_swa_eng(userText)  
        print("sw_en get_bot_response:-", bot_response_translate)

    chatbot_response_text = chatbot_response(bot_response_translate)

    if detected_language == 'sw':
        chatbot_response_text = translate_text_eng_swa(chatbot_response_text)

    return chatbot_response_text

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        user = User.query.filter_by(username=session['username']).first()
        return render_template('dashboard.html', user=user)  
    return redirect('/')




@app.route('/about')
def about():
    if 'username' in session:
        user = User.query.filter_by(username=session['username']).first()
        return render_template('about.html', user=user)
    return redirect('/')

@app.route('/admin')
def admin():
    users = User.query.all()
    return render_template('admin.html', users=users)

@app.route('/logout')
def logout():
    session.pop('username', None)  
    return redirect('/')  

if __name__ == '__main__':
    app.run(debug=True)
