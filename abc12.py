#!/usr/bin/env python
# coding: utf-8

# Business Objective:To get the nearest email template required by the user based on the key words specified by the user.

# In[1]:


import streamlit as st
st.title("MODEL DEPLOYEMENT: EMAIL TEMPLATES")
options=st.selectbox('Choose the category of the template you want to look into?',['Cold Email Templates','Researched Outreach Email Templates','LinkedIn Outreach InMail Templates','Follow Up Sales Templates','Break Up Sales Templates','Trial Sales Templates','Startup Sales Templates','Startup Sales Templates','Other Sales Email Templates','Sequence Sales Email Templates','Inbound Sales Email Templates','Best Subject Lines For Sales Emails','Opening Lines For Sales Emails','Discount Offer Email Templates','Persuasive Sales Email Templates','Sales Introduction Email Templates'])
st.write('y\You selected:',options)
# In[2]:


# approach 2


# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy
import string
import gensim
import operator
import re


# In[4]:


df_emails=pd.read_csv('C:/Users/ASUS/Desktop/Project 2/data modified.csv')
df_emails.head()


# In[5]:


df_emails.drop(columns={'Unnamed: 0','Template type',},axis=1,inplace=True)# droping columns
df_emails.ffill(axis = 0,inplace=True) # fills the null value with the previous value.
df_emails.head()


# In[6]:


# Data Cleaning and Pre-processing


# In[7]:


from spacy.lang.en.stop_words import STOP_WORDS

spacy_nlp = spacy.load('en_core_web_sm')

#create list of punctuations and stopwords
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

#function for data cleaning and processing
#This can be further enhanced by adding / removing reg-exps as desired.

def spacy_tokenizer(sentence):
 
    #remove distracting single quotes
    sentence = re.sub('\'','',sentence)

    #remove digits adnd words containing digits
    sentence = re.sub('\w*\d\w*','',sentence)

    #replace extra spaces with single space
    sentence = re.sub(' +',' ',sentence)

    #remove unwanted lines starting from special charcters
    sentence = re.sub(r'\n: \'\'.*','',sentence)
    sentence = re.sub(r'\n!.*','',sentence)
    sentence = re.sub(r'^:\'\'.*','',sentence)
    
    #remove non-breaking new line characters
    sentence = re.sub(r'\n',' ',sentence)
    
    #remove punctunations
    sentence = re.sub(r'[^\w\s]',' ',sentence)
    
    #creating token object
    tokens = spacy_nlp(sentence)
    
    #lower, strip and lemmatize
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    
    #remove stopwords, and exclude words less than 2 characters
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations and len(word) > 2]
    
    #return tokens
    return tokens


# In[29]:


print ('Cleaning and Tokenizing...')
df_emails['Context_tokenized'] =df_emails['Context'].map(lambda x: spacy_tokenizer(x))



# In[30]:


temp_plot = df_emails['Context_tokenized']



# In[31]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

series = pd.Series(np.concatenate(temp_plot)).value_counts()[:100]
wordcloud = WordCloud(background_color='white').generate_from_frequencies(series)

plt.figure(figsize=(15,15), facecolor = None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[32]:


#Building Word Dictionary¶


# Creating Dictionaries using In-Memory Objects
# It is super easy to create dictionaries that map words to IDs using Python's Gensim library.
# we first import the gensim library along with the corpora module from the library.
# We are now ready to create our dictionary. To do so, we can use the Dictionary object of the corpora module and pass it the list of tokens.
# Finally, to print the contents of the newly created dictionary, we can use the token2id object of the Dictionary class

# In[33]:


from gensim import corpora

#creating term dictionary
dictionary = corpora.Dictionary(temp_plot)

#list of few which which can be further removed
stoplist = set('hello and if this can would should could tell ask stop come go')
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)


# Token2id assigns numeric ID to unique words in our text.The word or token is the key of the dictionary and the ID is the value. 

# In[34]:


#print top 50 items from the dictionary with their unique token-id
dict_tokens = [[[dictionary[key], dictionary.token2id[dictionary[key]]] for key, value in dictionary.items() if key <= 50]]



# Dictionaries contain mappings between words and their corresponding numeric values. Bag of words corpora in the Gensim library are based on dictionaries and contain the ID of each word along with the frequency of occurrence of the word.

# In[35]:


#Feature Extraction (Bag of Words)


# # BOW

# Once we have the dictionary we can create a Bag of Word corpus using the doc2bow( ) function. This function counts the number of occurrences of each distinct word, convert the word to its integer word id and then the result is returned as a sparse vector

# The Dictionary object contains doc2bow method which basically performs 2 tasks :
# (1)It iterates through all the words in the text, if the word already exists in the corpus, it increments the frequency count for the word .
# (2)Otherwise it inserts the word into the corpus and sets its frequency count to 1 .

# In[36]:


corpus = [dictionary.doc2bow(desc) for desc in temp_plot]

word_frequencies = [[(dictionary[id], frequency) for id, frequency in line] for line in corpus[0:2]]


# The first tuple ('available', 1) basically means that the word 'available' occurred 1 time in the text. Similarly, ('company', 4) means that the word with 'company' occurred 4 times in the document.

# The bag of words approach works fine for converting text to numbers. However, it has one drawback. It assigns a score to a word based on its occurrence in a particular document. It doesn't take into account the fact that the word might also have a high frequency of occurrences in other documents as well. TF-IDF resolves this issue.

# In[37]:


# Term frequency = (Frequency of the word in a document)/(Total words in the document)


# In[38]:


# IDF(word) = Log((Total number of documents)/(Number of documents containing the word))


# In[39]:


# Using the Gensim library, we can easily create a TF-IDF corpus:


# # TF IDF

# Some words might not be stopwords but may occur more often in the documents and may be of less importance. Hence these words need to be removed or down-weighted in importance. The TFIDF model takes the text that share a common language and ensures that most common words across the entire corpus don’t show as keywords. 

# To find the TF-IDF value, we can use the TfidfModel class from the models module of the Gensim library. We simply have to pass the bag of word corpus as a parameter to the constructor of the TfidfModel class. In the output, you will see all of the words in the three sentences, along with their TF-IDF values:

# Latent Semantic Indexing, LSI (or sometimes LSA) transforms documents from either bag-of-words or (preferrably) TfIdf-weighted space into a latent space of a lower dimensionality. 
# Here we transformed our Tf-Idf corpus via Latent Semantic Indexing into a latent 300-D space (300-D because we set num_topics=300) and num_topics is usally choosen between 200-500)

# In[40]:


temp_tfidf_model = gensim.models.TfidfModel(corpus, id2word=dictionary)
temp_lsi_model = gensim.models.LsiModel(temp_tfidf_model[corpus], id2word=dictionary, num_topics=300)


# In[41]:


gensim.corpora.MmCorpus.serialize('temp_tfidf_model_mm', temp_tfidf_model[corpus])
gensim.corpora.MmCorpus.serialize('temp_lsi_model_mm',temp_lsi_model[temp_tfidf_model[corpus]])


# In[42]:


#Load the indexed corpus
temp_tfidf_corpus = gensim.corpora.MmCorpus('temp_tfidf_model_mm')
temp_lsi_corpus = gensim.corpora.MmCorpus('temp_lsi_model_mm') 




# In[43]:


from gensim.similarities import MatrixSimilarity

temp_index = MatrixSimilarity(temp_lsi_corpus, num_features = temp_lsi_corpus.num_terms)


# In[44]:


#Time for Semantic Search


# In[45]:


from operator import itemgetter

def search_similar_temp(search_term):

    query_bow = dictionary.doc2bow(spacy_tokenizer(search_term))
    query_tfidf = temp_tfidf_model[query_bow]
    query_lsi = temp_lsi_model[query_tfidf]

    temp_index.num_best = 1

    temp_list = temp_index[query_lsi]

    temp_list.sort(key=itemgetter(1), reverse=True)
    temp_names = []

    for j, mail in enumerate(temp_list):

        temp_names.append (
            {
                'Accuracy': round((mail[1] * 100),3),
                'email template Title': df_emails['Categories'][mail[0]],
                'template Subject': df_emails['Subject'][mail[0]],
                'template Context': df_emails['Context'][mail[0]]
            }

        )
        if j == (temp_index.num_best-1):
            break
    pd.set_option('Display.max_colwidth', -1)    
    return pd.DataFrame(temp_names, columns=['Accuracy','email template Title','template Subject','template Context'])
 

    


# In[46]:


if (st.button("Press here")):
    result1=search_similar_temp(options)
    title1=result1['email template Title']
    subject1=result1['template Subject']
    context1=result1['template Context']
    st.text(format(title1.to_string()))
    st.text(format(subject1.to_string()))
    st.text(format(context1.to_string()))
    
    

    
  
    
     
# search for movie tiles that are related to below search parameters



# In[47]:



# In[48]:




# #Reference links
# https://www.analyticsvidhya.com/blog/2021/06/part-16-step-by-step-guide-to-master-nlp-topic-modelling-using-lsa/
# https://stackabuse.com/python-for-nlp-working-with-the-gensim-library-part-1/
# https://www.geeksforgeeks.org/nlp-gensim-tutorial-complete-guide-for-beginners/
# https://towardsdatascience.com/latent-semantic-analysis-deduce-the-hidden-topic-from-the-document-f360e8c0614b
