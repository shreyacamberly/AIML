paragraph = '''"Haikyu is a Japanese manga series written and illustrated by Haruichi Furudate. It was serialized in Shueisha's shōnen manga magazine Weekly Shōnen Jump from February 2012 to July 2020, with its chapters collected in 45 tankōbon volumes. The story follows Shoyo Hinata, a boy determined to become a great volleyball player despite his small stature.
An anime television series adaptation produced by Production I.G, aired on MBS from April to September 2014, with 25 episodes. A second season aired from October 2015 to March 2016, with 25 episodes. A third season aired from October to December 2016, with 10 episodes. A fourth season was released in two split cours from January to December 2020, with 25 episodes.
The anime film series titled Haikyu!! Final will be released in two parts, which serves as the finale of the series; the first was released in February 2024."'''
paragraph

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

#tokeniztion-brkin into sentences-words
sentences = nltk.sent_tokenize(paragraph)
print(sentences)

#eg of stemming
stemmer = PorterStemmer()
stemmer.stem("thinking")
#lemmatizeing eg:
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("thinking")

import re
corpus=[]
for i in range (len(sentences)):
    review = re.sub('[^a-zA-Z]',' ',sentences[i])
    review = review.lower()
    corpus.append(review)
corpus

#stemming:
for i in corpus:
    words = nltk.word_tokenize(i)
    for word in words:
        if word not in set(stopwords.words('english')):
            print(stemmer.stem(word))
        
#lemmatize
for i in corpus:
    words = nltk.word_tokenize(i)
    for word in words:
        if word not in set(stopwords.words('english')):
            print(lemmatizer.lemmatize(word))


#coverting to no. vetors
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus)
cv.vocabulary_





