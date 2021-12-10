#python -m spacy download en_core_web_sm

#125, 164, 182

import nltk
from nltk.corpus import stopwords
import string
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
import spacy
import csv
import bs4 as bs
import urllib.request

nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")
set(stopwords.words('english'))


#function that gets each url
def urlList(urlsFile):
  #Reads file and puts into list
  file = open(urlsFile, "r")
  #csv_reader = csv.reader(urlsFile)
  urlList = []
  for row in file:
    urlList.append(row)
  
  #remove the header
  del urlList[0]

  return(urlList)


#file grab text from webpage
def web_to_text(ourURL, x):
  #raw_html = urllib.request.urlopen('https://en.wikipedia.org/wiki/Natural_language_processing')
  raw_html = urllib.request.urlopen(ourURL)
  raw_html = raw_html.read()

  #article_html = bs.BeautifulSoup(raw_html, 'lxml')
  article_html = bs.BeautifulSoup(raw_html, "html.parser")

  article_paragraphs = article_html.find_all('p')

  article_text = ''

  for para in article_paragraphs:
      article_text += para.text


  og_corpus = article_text

  #print('here')

  file_name = 'webConv' + str(x) + '.csv'
  fileN = 'webConv' + str(x) + '.csv'


  #print(file_name)

  file_name = open(file_name, 'w')
  file_name.write(og_corpus)
  file_name.close()

  return fileN



def cred_test(file, webNum):

  #the website the test is being done on
  print('\n\n', '---------------\n\n', webNum)

  #List that wil contain contents of csv file
  documents = []

  #Reads file and puts into list
  file = open(str(file), "r")
  csv_reader = csv.reader(file)
  for row in csv_reader:
      documents.append(row)

  #puts each sentence into a different list...
  text =[]
  #...first converts sentences into a string...
  for r in range(len(documents)):
    t = ' '.join(documents[r])
    #...and then add full sentences as strings into list
    text.append(t)

  #function uses spacy to remove stopwords and puncutation. Used for bag of words and tf-idf
  def spacy_tokenizer(inputT):
      tokens = nlp(inputT)
      tokens = [token.lemma_ for token in tokens if (
          token.is_stop == False and \
          token.is_punct == False and \
          token.lemma_.strip()!= '')]
      return tokens

  # set of stop words and punctuation. Used for sentences put in cluster
  #stop_words = set(stopwords.words('english') + list(string.punctuation)) 

  #For sentences being put into cluster. 
  #Removes punctuation in each sentence
  sentences = {}
  for k in range(len(text)):
    eachSent = ""
    eachSent = ''. join (text[k])
    eachSent = eachSent.replace("'","").replace('"', '').replace('.', '').replace('!', '').replace('?', '').replace(':', '').replace(';', '').lower()
    sentences[k] = eachSent

  #input : defines how we are going to enter the text to analyse, the default value ‘content’ means we pass them on as string variables, and that is what we are doing in this example. It’s of course not a feasible approach with large documents, so later we will pass on a list of file paths to open and process. tokenizer : determines which tokenizer to use. The TfidfVectorizer class has a builtin one, we are going to override that for the one we created with spaCy.
  tfidf_vector = \
      TfidfVectorizer(input = 'content', tokenizer = spacy_tokenizer)

  #the fit method, it will learn the vocabulary and the IDF part of the formula, and the transform method will transform the corpus into a sparse matrix format containing the TF-IDF values. Combined they form fit_transform
  result = tfidf_vector.fit_transform(text)

  #Words that are actually considered from the sentences
  #print('\nThese are the feature names:\n', tfidf_vector.get_feature_names_out())

  #number of clusters
  n_clusters = 3

  #Use means means
  kmeans = KMeans(n_clusters).fit(result)

  #Converts sparse matrix format to a more readable, dense matrix format
  dense = result.todense()
  denselist = dense.tolist()
  df = pd.DataFrame(
      denselist,columns=tfidf_vector.get_feature_names_out())

  #print(df)

  #Dictionary of each cluster and corresponding sentences
  clusters = {}
  for k in range(n_clusters):
    clusters[k] = []

  #Array of clustering group sentence belongs to
  labs = kmeans.labels_

  #Used to iterate through list that contains the sentences
  k = 0

  #Adds the words in corresponding sentencesto its corresponding clustering group
  for cl in (labs):
    sentence = sentences[k].split()
    for w in sentence:
      clusters[cl].append(w)
      #use if want to remove stop words
      # if w not in stop_words: 
      #   clusters[cl].append(w)  
    k+=1

    if k > len(sentences):
      break


  #print('\nThese are the clusters (without stopwords):\n', clusters)

  #Words determine to diminish credibility
  nonCred = ['abbi', 'heeb', 'raghead', 'niggers', 'porch monkey', 'chink','coolie','coon', 'cracker','cushi','kushi','gypsy','half breed','hebe','Jigaboo', 'jiggabo', 'jigarooni', 'jijjiboo', 'zigabo', 'jig', 'jigg', 'jigger','nigger', 'niger', 'nig', 'nigor', 'nigra', 'nigre', 'nigar', 'niggur', 'nigga', 'niggah', 'niggar', 'nigguh','niggress', 'nigette', 'negro','neger','sand nigger','chink','colored','jungle bunny','mamy','nig-nog','pickaninny' ,'tarbaby','jap','nip','oriental','yellowman', 'yellowwoman', 'camel jockey','towelhead','wetback', 'tacohead','tranny', 'fag','faggot','dyke','fruity','retard', 'retarded','riceater','i', 'my', 'me', 'favorite', 'agree', 'disagree', 'better', 'worse', 'worst', 'should','shit','pussy','ass','whore','hoe','bitch','cunt','fuck','bullshit','mother fucker','fucker','ass wipe','son of a bitch','son of a whore','chinese virus','cotton picker', 'asian virus','radical','radical islamic terrorism']

  #Varibale for credibility score
  credScore = 0

  #List continaing non credible words
  ncwList = []

  #Finds non credible words in the sentences and adds them to the non credible words list
  for key in clusters:
    ncWord = set(nonCred).intersection(clusters[key])
    ncWord = list(ncWord)
    for w in ncWord:
      ncwList.append(w)
  
  #print('\nThese are the words found that diminish credibility:\n',ncwList)

  #Crediblity is determined by the number of words in the non credible words list
  credScore = len(ncwList)
  # credScore = 10 -len(ncwList)

  credStatement = crediblity(credScore)
  print('\n Credibility Score:', credScore, '--', credStatement)

  return [credScore, credStatement, ncwList]


#Credibility is detemined by score. 0-3 is Likely Credible, 4-6 is Modertately Credible, and 7-10 are Unlikely Credible
def crediblity(credScore):
  credState =  "" 
  if credScore <= 3:
    credState = "Likely Credible"
  elif (credScore >=4) and (credScore <= 6):
    credState = "Fairly Credible"
  else:
    credState = "Unlikely Credible"


  # credState =  "" 
  # if credScore <= 3:
  #   credState = "Unlikely Credible"
  # elif (credScore >=4) and (credScore <= 6):
  #   credState = "Modertately Credible"
  # else:
  #   credState = "Likely Credible"

  return credState


def results(filename, webNum):
  credResults = cred_test(filename, webNum)

  credScore = credResults[0]
  credStatement = credResults[1]
  ncwList = credResults[2]

  #The results with the crediblity score as well as a statement on crediblity
  data = [webNum, credScore, credStatement]

  #Adds the data to a file
  with open('credResults.csv', 'a', encoding='UTF8') as f:
    writer = csv.writer(f)
    # write the data
    writer.writerow(data)
    f.close()




def main():
  #header of output csv file.
  header = ['file_name', 'Credibilty Score', 'Credibilty Statement']

  #Makes a file and writes the header to it
  with open('credResults.csv', 'w', encoding='UTF8') as f:
      writer = csv.writer(f)
      # write the header
      writer.writerow(header)
      # # write the data
      # writer.writerow(data)

  #name of the table w the urls
  urlF = 'urls.csv'

  #call the function that gets each url
  urls = urlList(urlF)

  #number of urls
  urlN = 0

  for w in urls:
    urlN += 1
    #calls function
    filename = web_to_text(w, urlN)
    
    webNum = 'Website ' + str(urlN)
    results(filename, webNum)


main()



