import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer
from collections import Counter 
import json

nltk.download('wordnet')
trainData = 'data_train.json'
tempData = 'data_test_wo_label_template.json'
testData = 'data_test.json'


with open(trainData) as json_file:
    data = json.load(json_file)
    all_words = []
    for i in range(len(data)):
        # tokenizes text without punctuation
        tokens = RegexpTokenizer(r'\w+').tokenize(data[i]['text'].lower())
        #tokens = word_tokenize(data[i]['text'].lower())

        # remove all stop words from word tokens
        stop_words = set(stopwords.words('english'))
        ## Add additional stop words to remove
        # add_stop_words = ('i', 'the')
        # stop_words.update(add_stop_words)
        #tokens = [w for w in tokens if not w in stop_words]

        # # lemmatization
        # #porter = WordNetLemmatizer()
        # porter = PorterStemmer()
        # # porter = LancasterStemmer()
        # stems = []
        # for t in tokens:    
        #     stems.append(porter.stem(t))
        #     #stems.append(porter.lemmatize(t))
        
        all_words.append(" ".join(tokens))
    
    ## OUTPUTS MOST FREQUENT WORDS
    #print(all_words)
    #features = Counter(all_words)
    #mostfreq = open('mostfreqword.json', 'w') 
    #print(features.most_common(1000), file = mostfreq)
    #mostfreq.close()

    ## OUTPUTS LIST OF CLEANED WORDS
    cleaned_words = open('cleanedwords_woPunct.json', 'w') 
    json.dump(all_words, cleaned_words)
    cleaned_words.close()
