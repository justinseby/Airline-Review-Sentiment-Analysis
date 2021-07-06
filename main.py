from flask import Flask, render_template, request
import requests
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from IPython import get_ipython
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import unicodedata
import re
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import fileinput
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import food_model,boarding_model,infrastructure_model,organization_model,payment_model,staff_model
from temp import load_csv
from csv import reader
from sklearn.feature_extraction.text import CountVectorizer
from array import array
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plotter; plt.rcdefaults()

#Dataset array creation
food_array=[]
infra_array=[]
web_array=[]
board_array=[]
organ_array=[]
pay_array=[]
staff_array=[]

app = Flask(__name__)



@app.route("/")
@app.route("/index")
def home():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/blog")
def blog():
    return render_template('blog.html')

@app.route("/contact")
def contact():
    return render_template('contact.html')
@app.route("/scrape", methods=["POST"])
def scrape():
    def remove_non_ascii(words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    
    def to_lowercase(words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    
    def remove_punctuation(words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    def stem_words(words):
        """Stem words in list of tokenized words"""
        stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems
    

    def lemmatize_verbs(words):
        """Lemmatize verbs in list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    def remove_stopwords(words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words
    def normalize(words):
        words = remove_non_ascii(words)
        words = to_lowercase(words)
        return words

    url = request.form['keyword']                                  #giving input
    print("URL :",url)
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urlopen(req).read()
    soup = BeautifulSoup(html, 'lxml')                             #webscrapping
    type(soup)
    name_box = soup.findAll('div', attrs={'class': 'text_content'})
    x = len(name_box)
    f = csv.writer(open('names.csv','w'))
    h = csv.writer(open('preprocessd_data.csv','w'))
    p = csv.writer(open('stop_words.csv','w'))
    file1 = open('splittedlines.txt','w')
    #f = csv.writer(open('AirIndia_append.csv', 'a'))
    g = csv.writer(open('Airindia_token.csv','a',encoding="utf-8"))
    for i in range(x):
       name = name_box[i].text.strip('| Verified Review Unverified Trip')
       name = name.lower()
       names = [item for item in name.split('\n')]
       names = ''.join(names)
       f.writerow([name])
       token = word_tokenize(name)
       g.writerow([token])
       words = normalize(token)
       h.writerows([words])
       words = remove_stopwords(words)
       p.writerow(words)
       datas = re.split(r'\. | and\ | but\ | *[\.\?!][\"\)\]]* *',names)
       for stuff in datas:
             file1.write(stuff)
             file1.write("\n")






    # Check sector
    file1 = open("splittedlines.txt","r")
    file2 = open("food.txt","w")
    file3 = open("infrastructure.txt","w")
    file4 = open("boarding.txt","w")
    file5 = open("payment.txt","w")
    file6 = open("staff.txt","w")
    file7 = open("organization.txt","w")
    line = file1.readline()
    food = ['water','hotel','voucher','meal','food','drinks','coke','coffee','caffeine','milk','refreshment','noodle','water','snacks','beverages','biriyani','sandwiches','food','menu','cappuccino','alcohol','breakfast']
    infrastructure = ['infrastructure','seats','toilet','flight','room','seat','chair','cabin','cockpit','wifi']
    boarding = ['check','boarding']
    payment = ['booking','pay','fee','ticket','cost','money','price','budget','ticket','currency','credit']
    staff = ['crew','staff','service','attendant','hostesses','pilot']
    organization = ['late','delay','re-schedule','schedule','landing','cancel','travel','off','depart','arrive','entertainment']
    while line:
        token = word_tokenize(line)
        for word in token:
            if word in food:
                file2.write(line)
                break
            elif word in infrastructure:
                file3.write(line)
                break
            elif word in boarding:
                file4.write(line)
                break
            elif word in payment:
                file5.write(line)
                break
            elif word in staff:
                file6.write(line)
                break
            elif word in organization:
                file7.write(line)
                break

        line = file1.readline()

    file1.close()
    file2.close()
    file3.close()
    file4.close()
    file5.close()
    file6.close()
    file7.close()

    a=[]
    file2 = open("food.txt","r")
    file4 = open("boarding.txt","r")
    file3 = open("infrastructure.txt","r")
    file5 = open("payment.txt","r")
    file6 = open("staff.txt","r")
    file7 = open("organization.txt","r")

    file9 = csv.writer(open('food_rating.csv','w'))
    file11 = csv.writer(open('boarding_rating.csv','w'))
    file13 = csv.writer(open('infrastructure_rating.csv','w'))
    file15 = csv.writer(open('payment_rating.csv','w'))
    file10 = csv.writer(open('staff_rating.csv','w'))
    file14 = csv.writer(open('organization_rating.csv','w'))

    for line in file2:

        a.append(line.strip())

    for line in a :
        blob = TextBlob(line)
        for sentence in blob.sentences:
            result = ((sentence.sentiment.polarity + 1)/2)*100
            if result != 50:
                food_array.append([line,sentence.sentiment.polarity,result])
    file2.close()

    a=[]
    for line in file4:

        a.append(line.strip())
    for line in a :
        blob = TextBlob(line)
        for sentence in blob.sentences:
            result = ((sentence.sentiment.polarity + 1)/2)*100
            if result != 50:
                board_array.append([line,sentence.sentiment.polarity,result])
    file4.close()

    a=[]
    for line in file3:

        a.append(line.strip())
    for line in a :
        blob = TextBlob(line)
        for sentence in blob.sentences:
            result = ((sentence.sentiment.polarity + 1)/2)*100
            if result != 50:
                infra_array.append([line,sentence.sentiment.polarity,result])
    file3.close()

    a=[]
    for line in file5:

        a.append(line.strip())
    for line in a :
        blob = TextBlob(line)
        for sentence in blob.sentences:
            result = ((sentence.sentiment.polarity + 1)/2)*100
            if result != 50:
                pay_array.append([line,sentence.sentiment.polarity,result])
    file5.close()

    a=[]
    for line in file6:

        a.append(line.strip())
    for line in a :
        blob = TextBlob(line)
        for sentence in blob.sentences:
            result = ((sentence.sentiment.polarity + 1)/2)*100
            if result != 50:
                staff_array.append([line,sentence.sentiment.polarity,result])
    file6.close()

    a=[]
    for line in file7:

        a.append(line.strip())
    for line in a :
        blob = TextBlob(line)
        for sentence in blob.sentences:
            result = ((sentence.sentiment.polarity + 1)/2)*100
            if result != 50:
                organ_array.append([line,sentence.sentiment.polarity,result])
    file7.close()

    food_data,food_target = load_csv(food_array)
    boarding_data,boarding_target = load_csv(board_array)
    infrastructure_data,infrastructure_target = load_csv(infra_array)
    organization_data,organization_target = load_csv(organ_array)
    payment_data,payment_target = load_csv(pay_array)
    staff_data,staff_target = load_csv(staff_array)

    p1 = food_model(food_data,food_target)
    p2 = boarding_model(boarding_data,boarding_target)
    p3 = infrastructure_model(infrastructure_data,infrastructure_target)
    p4 = organization_model(organization_data,organization_target)
    p5 = payment_model(payment_data,payment_target)
    p6 = staff_model(staff_data,staff_target)

    pieLabels = ['food','boarding','infrastructure','organisation','payment','staff']
#print(p1,p2,p3,p4,p5,p6,p7)
    pieValues = [p1,p2,p3,p4,p5,p6]
    figureObject, axesObject = plotter.subplots()
    axesObject.pie(pieValues, labels=pieLabels, autopct='%1.2f', shadow=True, startangle=90)
    plotter.title("Performance of each sector in Airline")
    plotter.show()


    return render_template('contact.html')
if __name__ == '__main__':
    app.run(debug=True)
