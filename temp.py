from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from csv import reader
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from array import array



def conversion(content):            #for converting string into proper format
    text = content
    text = [text]
    # create the transform
    vectorizer = CountVectorizer()

    vectorizer.fit(text)
    # encode document
    vector = vectorizer.transform(text)
    output = vector.toarray()
    return output
# Load a CSV filL
dt=np.dtype([('a',np.float32,1),('b',np.float32,1)])
def load_csv(filename):
    data = list()
    target = list()
    for row in filename:
        i = 100
        a = conversion(row[0])
        first = list()
        for j in a:
            for k in j:
                first.append(np.float32(k))
        len_first = len(first)
        for len_first in range(len(first),i):
            first.append(0)
        first.append(np.float32(row[1]))
        data.append(first)
        target.append(np.float32(row[2]))
    c=np.array(data)
    
    b=np.array(target)
    return c,b
