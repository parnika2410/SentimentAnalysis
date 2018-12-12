import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from functools import partial
import re
import sys
import csv
def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    word = re.sub(r'(-|\')', '', word)
    #word = word.replace("'t", " not")
    #word = word.replace("'ll", " will")
    #word = word.replace("'m", " am")

    return word




def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))','Smiley',tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)','Laughing',tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)','Love',tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)','Wink',tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)','Sad',tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()','Cry',tweet)
    return tweet


def preprocess_tweet(tweet):
    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    tweet = re.sub(r"\'t", " not", tweet)
    tweet = re.sub(r"\'re", " are", tweet)
    tweet = re.sub(r"\'s", " is", tweet)
    tweet = re.sub(r"\'d", " would", tweet)
    tweet = re.sub(r"\'ll", " will", tweet)
    tweet = re.sub(r"\'t", " not", tweet)
    tweet = re.sub(r"\'ve", " have", tweet)
    tweet = re.sub(r"\'m", " am", tweet)
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', '', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = re.sub('[^A-Za-z0-9]+', ' ', tweet)
    tweet= re.sub(" \d+", " ", tweet)

    return tweet

""" Creates a dictionary with slangs and their equivalents and replaces them """
with open('slang.txt') as file:
    slang_map = dict(map(str.strip, line.partition('\t')[::2])
                     for line in file if line.strip())

slang_words = sorted(slang_map, key=len, reverse=True)  # longest first for regex
regex = re.compile(r"\b({})\b".format("|".join(map(re.escape, slang_words))))
replaceSlang = partial(regex.sub, lambda m: slang_map[m.group(1)])

output=[]

f = open("Sentiment Analysis Dataset.csv","r")
f2=open("anuj_pp.csv","w+")
#f3=open("polarity.txt","r+")


str1=['','']
res=[]
wr=csv.writer(f2)
mycsv = csv.reader(f)
k=0
res=[]
data=[]

for row in f:

    columns = row.split(',')
    o1= (columns[1])
    o1 = o1.strip(' "\'')
    o2 = (columns[3])



    str1[0]=o1
    o2 = handle_emojis(o2)
    o2=preprocess_word(o2)



    o2=preprocess_tweet(o2)
    o2 = replaceSlang(o2)
    str1[1]=o2
    tokens = word_tokenize(str1[1])
    print(tokens)
    stop_words = stopwords.words('english')
    print([i for i in tokens if i not in stop_words])

    str1[1]=([i for i in tokens if i not in stop_words])
    tweet = ' '.join(str1[1])
    if (k > 0):
        res.append(tweet)
        wr.writerow(str1)
        data.append(o1)
    k = k + 1;


f.close()
f2.close()




# feautre extraction
# bow
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(res)
print(bow)

# tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(res)
print(tfidf)



from sklearn import tree
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import *
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


# -----------------------------------LOGISTIC RERGRESSION-----------------------------------
bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(res)
train_bow = bow[8678:, :]
test_bow = bow[:8678, :]
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("----------------------------------------------------------------------------------")
print("________LOGISTIC RERGRESSION________")
print("BagOfWords ----->  ")
print("Confusion Matrix:" + str(confusion_matrix(y_test, y_pred)))
mat=confusion_matrix(y_test,y_pred)
tp=mat[1][1]
tn=mat[1][0]
fp=mat[0][1]
fn=mat[0][0]
TPR=tp/(fn+tp)
FPR=fp/(fp+tn)
Precision=tp/(tp+fp)
Recall=tp/(tp+fn)
F1=2/((1/Precision)+(1/Recall))
print('True Positive Rate: ',TPR)
print('False Positive Rate: ',FPR)
print("Accuracy:" + str(accuracy_score(y_test, y_pred) * 100) + "%")
print('Precision: ',Precision*100,"%")
print('Recall: ',Recall*100,"%")
print('F1 Score: ',F1*100,"%")

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(res)

X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("TF-IDF --------->  ")
print("Confusion Matrix:" + str(confusion_matrix(y_test, y_pred)))
mat=confusion_matrix(y_test,y_pred)
tp=mat[1][1]
tn=mat[1][0]
fp=mat[0][1]
fn=mat[0][0]
TPR=tp/(fn+tp)
FPR=fp/(fp+tn)
Precision=tp/(tp+fp)
Recall=tp/(tp+fn)
F1=2/((1/Precision)+(1/Recall))
print('True Positive Rate: ',TPR)
print('False Positive Rate: ',FPR)
print("Accuracy:" + str(accuracy_score(y_test, y_pred) * 100) + "%")
print('Precision: ',Precision*100,"%")
print('Recall: ',Recall*100,"%")
print('F1 Score: ',F1*100,"%")



# -----------------------------------DECISION TREE CLASSIFIER-----------------------------------

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(res)
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X=X_train, y=y_train)
y_pred = clf.predict(X_test)
print("----------------------------------------------------------------------------------")
print("_____DECISION TREE CLASSIFIER_____")
print("BagOfWords ----->  ")
print("Confusion Matrix:" + str(confusion_matrix(y_test, y_pred)))
mat=confusion_matrix(y_test,y_pred)
tp=mat[1][1]
tn=mat[1][0]
fp=mat[0][1]
fn=mat[0][0]
TPR=tp/(fn+tp)
FPR=fp/(fp+tn)
Precision=tp/(tp+fp)
Recall=tp/(tp+fn)
F1=2/((1/Precision)+(1/Recall))
print('True Positive Rate: ',TPR)
print('False Positive Rate: ',FPR)
print("Accuracy:" + str(accuracy_score(y_test, y_pred) * 100) + "%")
print('Precision: ',Precision*100,"%")
print('Recall: ',Recall*100,"%")
print('F1 Score: ',F1*100,"%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(res)

X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X=X_train, y=y_train)
y_pred = clf.predict(X_test)
print("TF-IDF --------->  ")
print("Confusion Matrix:" + str(confusion_matrix(y_test, y_pred)))
mat=confusion_matrix(y_test,y_pred)
tp=mat[1][1]
tn=mat[1][0]
fp=mat[0][1]
fn=mat[0][0]
TPR=tp/(fn+tp)
FPR=fp/(fp+tn)
Precision=tp/(tp+fp)
Recall=tp/(tp+fn)
F1=2/((1/Precision)+(1/Recall))
print('True Positive Rate: ',TPR)
print('False Positive Rate: ',FPR)
print("Accuracy:" + str(accuracy_score(y_test, y_pred) * 100) + "%")
print('Precision: ',Precision*100,"%")
print('Recall: ',Recall*100,"%")
print('F1 Score: ',F1*100,"%")


# -----------------------------------NAIVE BAYES-----------------------------------
bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(res)
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
log_model = MultinomialNB()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("----------------------------------------------------------------------------------")
print("________NAIVE BAYES________")
print("BagOfWords ----->  ")
print("Confusion Matrix:" + str(confusion_matrix(y_test, y_pred)))
mat=confusion_matrix(y_test,y_pred)
tp=mat[1][1]
tn=mat[1][0]
fp=mat[0][1]
fn=mat[0][0]
TPR=tp/(fn+tp)
FPR=fp/(fp+tn)
Precision=tp/(tp+fp)
Recall=tp/(tp+fn)
F1=2/((1/Precision)+(1/Recall))
print('True Positive Rate: ',TPR)
print('False Positive Rate: ',FPR)
print("Accuracy:" + str(accuracy_score(y_test, y_pred) * 100) + "%")
print('Precision: ',Precision*100,"%")
print('Recall: ',Recall*100,"%")
print('F1 Score: ',F1*100,"%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(res)

X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
clf = MultinomialNB()
clf = clf.fit(X=X_train, y=y_train)
y_pred = clf.predict(X_test)
print("TF-IDF --------->  ")
print("Confusion Matrix:" + str(confusion_matrix(y_test, y_pred)))
mat=confusion_matrix(y_test,y_pred)
tp=mat[1][1]
tn=mat[1][0]
fp=mat[0][1]
fn=mat[0][0]
TPR=tp/(fn+tp)
FPR=fp/(fp+tn)
Precision=tp/(tp+fp)
Recall=tp/(tp+fn)
F1=2/((1/Precision)+(1/Recall))
print('True Positive Rate: ',TPR)
print('False Positive Rate: ',FPR)
print("Accuracy:" + str(accuracy_score(y_test, y_pred) * 100) + "%")
print('Precision: ',Precision*100,"%")
print('Recall: ',Recall*100,"%")
print('F1 Score: ',F1*100,"%")


# -----------------------------------KNN-----------------------------------
bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(res)
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
log_model = KNeighborsClassifier()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("----------------------------------------------------------------------------------")
print("________K NEAREST NEIGHBORS________")
print("BagOfWords ----->  ")
print("Confusion Matrix:" + str(confusion_matrix(y_test, y_pred)))
mat=confusion_matrix(y_test,y_pred)
tp=mat[1][1]
tn=mat[1][0]
fp=mat[0][1]
fn=mat[0][0]
TPR=tp/(fn+tp)
FPR=fp/(fp+tn)
Precision=tp/(tp+fp)
Recall=tp/(tp+fn)
F1=2/((1/Precision)+(1/Recall))
print('True Positive Rate: ',TPR)
print('False Positive Rate: ',FPR)
print("Accuracy:" + str(accuracy_score(y_test, y_pred) * 100) + "%")
print('Precision: ',Precision*100,"%")
print('Recall: ',Recall*100,"%")
print('F1 Score: ',F1*100,"%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(res)

X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
clf = KNeighborsClassifier()
clf = clf.fit(X=X_train, y=y_train)
y_pred = clf.predict(X_test)
print("TF-IDF --------->  ")
print("Confusion Matrix:" + str(confusion_matrix(y_test, y_pred)))
mat=confusion_matrix(y_test,y_pred)
tp=mat[1][1]
tn=mat[1][0]
fp=mat[0][1]
fn=mat[0][0]
TPR=tp/(fn+tp)
FPR=fp/(fp+tn)
Precision=tp/(tp+fp)
Recall=tp/(tp+fn)
F1=2/((1/Precision)+(1/Recall))
print('True Positive Rate: ',TPR)
print('False Positive Rate: ',FPR)
print("Accuracy:" + str(accuracy_score(y_test, y_pred) * 100) + "%")
print('Precision: ',Precision*100,"%")
print('Recall: ',Recall*100,"%")
print('F1 Score: ',F1*100,"%")


# -----------------------------------RANDOM FOREST-----------------------------------
bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(res)
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
log_model = RandomForestClassifier()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("----------------------------------------------------------------------------------")
print("________RANDOM FOREST________")
print("BagOfWords ----->  ")
print("Confusion Matrix:" + str(confusion_matrix(y_test, y_pred)))
mat=confusion_matrix(y_test,y_pred)
tp=mat[1][1]
tn=mat[1][0]
fp=mat[0][1]
fn=mat[0][0]
TPR=tp/(fn+tp)
FPR=fp/(fp+tn)
Precision=tp/(tp+fp)
Recall=tp/(tp+fn)
F1=2/((1/Precision)+(1/Recall))
print('True Positive Rate: ',TPR)
print('False Positive Rate: ',FPR)
print("Accuracy:" + str(accuracy_score(y_test, y_pred) * 100) + "%")
print('Precision: ',Precision*100,"%")
print('Recall: ',Recall*100,"%")
print('F1 Score: ',F1*100,"%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(res)

X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
clf = RandomForestClassifier()
clf = clf.fit(X=X_train, y=y_train)
y_pred = clf.predict(X_test)
print("TF-IDF --------->  ")
print("Confusion Matrix:" + str(confusion_matrix(y_test, y_pred)))
mat=confusion_matrix(y_test,y_pred)
tp=mat[1][1]
tn=mat[1][0]
fp=mat[0][1]
fn=mat[0][0]
TPR=tp/(fn+tp)
FPR=fp/(fp+tn)
Precision=tp/(tp+fp)
Recall=tp/(tp+fn)
F1=2/((1/Precision)+(1/Recall))
print('True Positive Rate: ',TPR)
print('False Positive Rate: ',FPR)
print("Accuracy:" + str(accuracy_score(y_test, y_pred) * 100) + "%")
print('Precision: ',Precision*100,"%")
print('Recall: ',Recall*100,"%")
print('F1 Score: ',F1*100,"%")


# -----------------------------------SUPPORT VECTOR MACHINE-----------------------------------
bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(res)
X_train, X_test, y_train, y_test = train_test_split(bow, data, test_size=0.20, train_size=0.80, random_state=1234)
log_model = LinearSVC()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("----------------------------------------------------------------------------------")
print("________SUPPORT VECTOR MACHINE________")
print("BagOfWords ----->  ")
print("Confusion Matrix:" + str(confusion_matrix(y_test, y_pred)))
mat=confusion_matrix(y_test,y_pred)
tp=mat[1][1]
tn=mat[1][0]
fp=mat[0][1]
fn=mat[0][0]
TPR=tp/(fn+tp)
FPR=fp/(fp+tn)
Precision=tp/(tp+fp)
Recall=tp/(tp+fn)
F1=2/((1/Precision)+(1/Recall))
print('True Positive Rate: ',TPR)
print('False Positive Rate: ',FPR)
print("Accuracy:" + str(accuracy_score(y_test, y_pred) * 100) + "%")
print('Precision: ',Precision*100,"%")
print('Recall: ',Recall*100,"%")
print('F1 Score: ',F1*100,"%")


tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(res)

X_train, X_test, y_train, y_test = train_test_split(tfidf, data, test_size=0.20, train_size=0.80, random_state=1234)
clf = LinearSVC()
clf = clf.fit(X=X_train, y=y_train)
y_pred = clf.predict(X_test)
print("TF-IDF --------->  ")
print("Confusion Matrix:" + str(confusion_matrix(y_test, y_pred)))
mat=confusion_matrix(y_test,y_pred)
tp=mat[1][1]
tn=mat[1][0]
fp=mat[0][1]
fn=mat[0][0]
TPR=tp/(fn+tp)
FPR=fp/(fp+tn)
Precision=tp/(tp+fp)
Recall=tp/(tp+fn)
F1=2/((1/Precision)+(1/Recall))
print('True Positive Rate: ',TPR)
print('False Positive Rate: ',FPR)
print("Accuracy:" + str(accuracy_score(y_test, y_pred) * 100) + "%")
print('Precision: ',Precision*100,"%")
print('Recall: ',Recall*100,"%")
print('F1 Score: ',F1*100,"%")
