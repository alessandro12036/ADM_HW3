import requests
from bs4 import BeautifulSoup as bs
import os
import pickle
import numpy as np
import time
import re
import datetime as dt
import csv
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import heapq

def take_n_urls(n):

"""This function extracts the url of each animes and returns
them as a list. Given the high number of requests it has to perform, it checks
the status code of each request, and, if an error code has occured,
waits an incremental amount of time before making a new one."""

    main_url = "https://myanimelist.net/topanime.php"

    # this list will contain all the urls we'll retrieve
    urls = []

    # each page shows 50 elements and we can retrieve each page by manipulating the "limit" query
    for limit in range(0, n, 50):
        content = requests.get(main_url,
                               params={"limit": limit})
        if content.status_code == 404:
            print(f"Page with limit {limit} was not found. Interrumpting and returning pages found")
            break
        soup = bs(content.content, "html.parser")

        # from the content of each page we retrieve the portions that contain the urls
        results = soup.find_all("a",
                                class_= "hoverinfo_trigger fl-l ml12 mr8")

        # finally, we take the string containing each url by taking the attribute href,
        # and we append them in the urls list
        for result in results:
            url = result["href"]
            if url not in urls:  # check for duplicates
                urls.append(url)

    return urls


def save_html_pages(urls):

"""Extracts the html of each url provided in the input
and saves it in a folder corresponding to its page in the 
anime ranking list. The path of each file has the structure
html_pages/ranking_page_i/article_j. Given the long time needed
to crawl all the animes, we created a counter variable and saved it as
a binary file in order to be able to continue from where the last 
person running the function left off."""

    if "counter_pages" not in os.listdir():
        start = 0
    else:
        with open("counter_pages", "rb") as counter_file:
            start = pickle.load(counter_file) + 1

    print(f"Starting from anime #{start}")
    n = len(urls)
    for i in range(start, n):
        ranking_page = str(int(np.floor(i/50)))
        if i % 50 == 0 or f"ranking_page_{ranking_page}" not in os.listdir("./html_pages"):
            os.mkdir(f"html_pages/ranking_page_{ranking_page}")
        html_page = requests.get(urls[i])
        sleep_timer = 60
        while html_page.status_code != 200: # if the status_code is not 200, we've exceeded the number of requests and have to wait
            print(f"Exceeded number of requests while retrieving page #{i}.\nWaiting {sleep_timer} seconds")
            html_page.close()
            time.sleep(sleep_timer)
            html_page = requests.get(urls[i])
            sleep_timer += 10
        with open (f"html_pages/ranking_page_{ranking_page}/article_{i}.html", "w", encoding="utf-8") as file:
            file.write(html_page.text)
        with open ("counter_pages", "wb") as counter_file:
            pickle.dump(i, counter_file)


def collect_info(num_article, folder='tsv_files'):

"""This function extracts all the information we need for our dataset
from each html page and saves it as a file named 'anime_i'."""

    ranking_page = str(int(np.floor(num_article / 50)))
    article = f'{path_ex_aurelie}/html_pages/ranking_page_{ranking_page}/article_{num_article}.html'
    with open(article, "r", encoding="utf-8") as file:
        art = bs(file.read(), 'html.parser')

    # animeTitle
    animeTitle = art.find('h1', {'class': "title-name h1_bold_none"}).string
    # print('animeTitle :',animeTitle)

    # animeType
    animeType = art.find('span', {'class': "information type"}).string
    # print('animeType :',animeType)

    # animeNumEpisode and Dates (there is not specific name for those two info)
    # list lines with tag <div class="spaceit_pad">
    lines = art.find_all('div', {'class': "spaceit_pad"})
    for line in lines:
        # for each div tag there is one span, so here we look for the span tag with 'Episodes:' and 'Aired'
        sp = line.find('span', {'class': "dark_text"})
        # to avoid error if there is no span
        if sp is not None:
            # for span 'Episodes' (and the div tag which corresponds)
            if sp.string == 'Episodes:':
                # extract the content of the right div tag and take the third line which correspond to the number of episodes
                if line.contents[2] != '\n  Unknown\n  ':
                    animeNumEpisode = int(line.contents[2])
                    # animeNumEpisode = int(re.findall(r'-?\d+\.?\d*', str(line))[0])           #if we want to use regex
                else:
                    animeNumEpisode = ''
            # for span 'Aired' (and the div tag which corresponds)
            if sp.string == 'Aired:':
                str_dates = line.contents[2].split('\n  ')[1]
                if str_dates == 'Not available':
                    releaseDate = ''
                    endDate = ''
                else:
                    # if "Status: Finished Airing" (there is a endDate)
                    if ('to' in str_dates) and ('?' not in str_dates):
                        # extract the content of the right div tag and take the third line which correspond to the dates (fix the issue of '\n')
                        str_releaseDate, str_endDate = str_dates.split(' to ')

                        # choose the right datetime format of str_releaseDate
                        if len(str_releaseDate.split(' ')) == 3:
                            date_format_releaseDate = "%b %d, %Y"
                        elif len(str_releaseDate.split(' ')) == 2:
                            date_format_releaseDate = "%b %Y"
                        else:
                            date_format_releaseDate = "%Y"
                        # convert str_releaseDate into a datetime
                        releaseDate = dt.datetime.strptime(str_releaseDate, date_format_releaseDate)

                        # choose the right datetime format of str_endDate
                        if len(str_endDate.split(' ')) == 3:
                            date_format_endDate = "%b %d, %Y"
                        elif len(str_endDate.split(' ')) == 2:
                            date_format_endDate = "%b %Y"
                        else:
                            date_format_endDate = "%Y"
                        # convert str_releaseDate into a datetime
                        endDate = dt.datetime.strptime(str_endDate, date_format_endDate)

                    else:
                        str_releaseDate = str_dates.split(' to ')[0]
                        # choose the right datetime format of str_releaseDate
                        if len(str_releaseDate.split(' ')) == 3:
                            date_format_releaseDate = "%b %d, %Y"
                        elif len(str_releaseDate.split(' ')) == 2:
                            date_format_releaseDate = "%b %Y"
                        else:
                            date_format_releaseDate = "%Y"
                        # convert str_releaseDate into a datetime
                        releaseDate = dt.datetime.strptime(str_releaseDate, date_format_releaseDate)

                        endDate = ''
    # print('animeNumEpisode :',animeNumEpisode)
    # print('releaseDate :',releaseDate)
    # print('endDate :',endDate)

    # animeNumMembers
    animeNumMembers = int(art.find('span', {'class': "numbers members"}).contents[1].string.replace(',', ''))
    # print('animeNumMembers :',animeNumMembers)

    # animeScore
    score = art.find('div', {'class': "score-label"}).string
    if score == 'N/A':
        animeScore = ''
    else:
        animeScore = float(score)
    # print('animeScore :',animeScore)

    # animeUsers
    if art.find('span', {'itemprop': "ratingCount"}) is not None:
        animeUsers = int(art.find('span', {'itemprop': "ratingCount"}).string)
    else:
        animeUsers = ''
    # print('animeUsers :',animeUsers)

    # animeRank
    if art.find('span', {'class': "numbers ranked"}).contents[1].string != 'N/A':
        animeRank = int(art.find('span', {'class': "numbers ranked"}).contents[1].string.replace('#', ''))
    else:
        animeRank = ''
    # print('animeRank :',animeRank)

    # animePopularity
    animePopularity = int(art.find('span', {'class': "numbers popularity"}).contents[1].string.replace('#', ''))
    # print('animePopularity :',animePopularity)

    # animeDescription
    desc = art.find('p', {'itemprop': "description"}).contents
    animeDescription = ''
    # remove <br/> Tag and '\n'
    for ele in desc:
        # delete tags with regex
        ele = re.sub(re.compile('<.*?>'), '', str(ele))
        animeDescription += ele
        animeDescription = animeDescription.replace('\n', '')
    # print('animeDescription :',animeDescription.replace('\n',''))

    # animeRelated
    animeRelated = []
    # store the table which contain related animes
    table = art.find('table', {'class': "anime_detail_related_anime"})
    if table is not None:
        # store all links/anime related with 'a' Tag
        links = table.find_all('a')
        for link in links:
            # check if there is a hyperlink and add it in the list if yes
            if (link.get('href') is not None) and (link.string is not None):
                animeRelated += [link.string]
        animeRelated = list(set(animeRelated))
    else:
        animeRelated = ''
    # print('animeRelated :',animeRelated)

    # animeCharacters
    animeCharacters = art.find_all('h3', {'class': "h3_characters_voice_actors"})
    animeCharacters = [char.string for char in animeCharacters]
    # print('animeCharacters :',animeCharacters)

    # animeVoices
    td_Voices = art.find_all('td', {'class': "va-t ar pl4 pr4"})
    animeVoices = [voice.find('a').string for voice in td_Voices]
    # print('animeVoices :',animeVoices)

    # animeStaff
    # if there is a staff, the div which correspond to the table Staff is the second one (there are div with {'class':"detail-characters-list clearfix"})
    if len(art.find_all('div', {'class': "detail-characters-list clearfix"})) > 1:
        div_staff = art.find_all('div', {'class': "detail-characters-list clearfix"})[1]
        td_staff = div_staff.find_all('td', {'class': "borderClass"})
        animeStaff = []
        for td in td_staff:
            if td.get('width') == None:
                animeStaff.append([td.find('a').string, td.find('small').string])
    # if there is not staff
    else:
        animeStaff = ''
    # print('animeStaff :',animeStaff)

    # create a .tsv file with attributes
    with open(f'{folder}/anime_{num_article}', 'wt', encoding="utf8") as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow([animeTitle, animeType, animeNumEpisode, releaseDate, endDate, animeNumMembers, \
                             animeScore, animeUsers, animeRank, animePopularity, animeDescription, animeRelated, \
                             animeCharacters, animeVoices, animeStaff])


def process_text(text, stemmer_type="porter"):

"""This function process the text provided as a input and returns
a list of stemmed tokenized words, with stopwords and punctuation filtered
out."""

    # For identifying the stop words
    eng_stopwords = stopwords.words("english")

    # For stemming
    if stemmer_type == "lancaster":
        stemmer = nltk.stem.LancasterStemmer()
    elif stemmer_type == "porter":
        stemmer = nltk.stem.PorterStemmer()

    try:
        tokenized = nltk.word_tokenize(text)
        stemmed = [stemmer.stem(word) for word in tokenized if ((word.lower() not in eng_stopwords) and (word not in string.punctuation))]
    except TypeError as e:
        print(text)
        raise TypeError
    return stemmed


def alphanumeric_sort(key):

"""This function provides a way to correctly order files without
having them to be named with leading zeros."""

    num = int(re.search("([0-9]+)", key).group(0))
    return num


def merge_tsvs(path, colnames):

"""This function merges each tsv in a single dataframe"""

    files = sorted(os.listdir(path), key=alphanumeric_sort)
    df = pd.read_csv(path+files[0],
                     names=colnames,
                     sep="\t", engine='python')
    for file_name in files[1:]:
        df2 = pd.read_csv(path+file_name,
                          names=colnames,
                          sep="\t", engine='python')
        df = pd.concat([df, df2], ignore_index=True)
    return df


def create_vocabulary(corpus, name_voc_file = "vocabulary.pkl"):

"""This function creates a vocabulary of all the words found in the
corpus and assigns to each a specific integer term_id. It then saves 
this vocabulary as a binary file (so to avoid having to recreate it
each time) and returns it."""

    voc = set()
    i=0
    for doc in corpus :
        #print(i)
        #i+=1
        voc = voc.union(set(doc))

    dict_voc = dict(zip(sorted(voc),range(len(voc))))
    with open(name_voc_file, "wb") as file:
        pickle.dump(dict_voc, file)
    return dict_voc


def inverted_index(corpus, voc, name_inv_ind_file="inverted_index.pkl"):

"""This function creates an inverted index, meaning a dictionary with
the term_id corresponding to each word in the vocabulary 'voc' as 
key and the documents containing that specific word as values.
As for the previous function, we both return the final inverted_index
and save it as a binary file for further use."""

    # create a inverted_index "empty", i.e. only with term_id of vocabulary
    inverted_index = dict()
    for term, term_id in voc.items():
        inverted_index[term_id] = set()

    for doc, num_doc in zip(corpus, range(len(corpus))):
        # print(num_doc)
        words_checked = []
        for word in doc:
            if word not in words_checked:  # to avoid looking for the same word more than once in the same doc
                words_checked.append(word)
                term_id = voc[word]
                inverted_index[term_id] = inverted_index[term_id].union(set([num_doc]))

    for term_id, docs in inverted_index.items():
        inverted_index[term_id] = sorted(list(inverted_index[term_id]))

    # save the inverted_index as a .pkl file
    with open(name_inv_ind_file, "wb") as file:
        pickle.dump(inverted_index, file)

    return inverted_index


def download_corpus(name_file_corpus = 'df_with_tokens.p'):

"""This function extracts or downloads the corpus (depending on
whether we've already created it) and returns it."""

    print('Downloading corpus... ', end ='')
    with open(name_file_corpus, 'rb') as file:
        df = pickle.load(file)

    corpus = list(df['tokenized_desc'])
    print('Done')
    return corpus


def download_voc(corpus, name_voc_file):

"""This function extracts or downloads the vocabulary (depending on
whether we've already created it) and returns it."""

    print('Downloading vocabulary... ', end ='')
    if name_voc_file not in os.listdir():
        voc = create_vocabulary(corpus, name_voc_file)
    else :
        with open(name_voc_file, "rb") as file:
            voc = pickle.load(file)
    print('Done')
    return voc


def download_inverted_index(corpus,voc, name_inv_ind_file):

"""This function extracts or downloads the inverted index (depending 
on whether we've already created it) and returns it."""

    print('Downloading inverted index... ', end ='')
    if name_inv_ind_file not in os.listdir():
        inv_ind = inverted_index(corpus,voc, name_inv_ind_file)
    else :
        with open(name_inv_ind_file, "rb") as file:
            inv_ind = pickle.load(file)
    print('Done')
    return inv_ind


def search_engine_1(voc, inverted_index, urls):

"""This function takes a query from the user and returns all the
animes whose synopsis include all the words in the query."""

    path_ex_aurelie ='C:/Users/aurel/OneDrive/Bureau/IMT/3ème année IMT/0_Cours Sapienza/ADM/Homework/Homework 3'
    path_ex_alessandro = "."
    # ask the query to the user
    query = input('What is your query ?')

    # apply the preprocessing to the query
    query = process_text(query.lower())

    # initialization with the set of the document which the first word of the query
    first_word = query[0]
    # check if first_word is in the vocabulary (otherwise, is doesn't exist any document to answer to the query)
    if first_word in voc:
        first_term_id = voc[first_word]
        docs_list = set(inverted_index[first_term_id])

        for word in query[1:]:
            # if the next word is in the voc, it means that it exists documents with this word
            if word in voc:
                # store the term_id and find the documents which contain this word in the inverted_index
                term_id = voc[word]
                docs = inverted_index[term_id]

                # compute the intersection because every word of the query has to be in the description of the doc
                docs_list = docs_list.intersection(set(inverted_index[first_term_id]))

                # if the intersection between the previous docs_list and the set of document with the next word,\
                # no document answers to the query
                if len(docs_list) == 0:
                    print('Nothing correspond to your queries')
                    return

            else:  # no document answers to the query
                print('Nothing corresponds to your queries')
                return

        # Now we have the doc IDs so we can merge interesting information
        html_df = pd.read_csv(path_ex_alessandro + "/html_df.csv")  # csv which contains tsv line of each document
        cols = ["animeTitle", "animeDescription"]
        result = html_df.iloc[sorted(list(docs_list))][cols]
        result['Url'] = [urls[i] for i in sorted(list(docs_list))]

        return result

    else:  # no document answers to the query
        print('Nothing correspond to your queries')
        return


def get_tfidf(word, doc, corpus, idf=None):

"""This function computes the document's tfidf score for
the word given in input and then returns it. Since the idf only
depends on the word and the corpus and not on the specific text 
we're computing the score for, we also return the calculated idf so 
we can store it and use it every time that word occurs again."""

    tf = doc.count(word) / len(doc)
    counter_docs = 0
    # if the idf parameter has not been provided, we compute it
    if idf == None:
        for text in corpus:
            if word in text:
                counter_docs += 1
        idf = np.log(len(corpus) / counter_docs)
    tfidf = tf * idf
    return idf, tfidf


def second_inverted_index(corpus, voc, name_inv_ind_tf_idf_file="inverted_index_2.p", name_idfs_file="idfs.p"):

"""This function creates an inverted index, meaning a dictionary with
the term_id corresponding to each word in the vocabulary 'voc' as 
key and lists [document_id, tfidf] as values.
It then sorts the documents for each word accorsing to their tfidf
scores, saves both the inverted index and the idfs computed as 
binary pickle files and returns them."""

    inverted_index_2 = dict()
    # first, we initialize each field in the inverted_index
    for term_id in voc.values():
        inverted_index_2[term_id] = list()

    idf_calculated = {}  # the idfs can be calculated once for each word since idf = np.log(len(corpus) / documents_with_word)

    for doc, num_doc in zip(corpus, range(len(corpus))):
        words_checked = []
        for word in doc:
            if word not in words_checked:  # to avoid looking for the same word more than once in the same doc
                term_id = voc[word]
                # if this is the first time we encounter this word, we need to obtain the idf and save it for future use
                if word not in idf_calculated.keys():
                    idf, tfidf = get_tfidf(word, doc, corpus)
                    idf_calculated[word] = idf
                # otherwise, we provide it to the function directly
                else:
                    _, tfidf = get_tfidf(word, doc, corpus, idf)
                # we add the doc index and the tfidf score to the dictionary
                inverted_index_2[term_id].append([num_doc, tfidf])
                # we mark this word as "checked" for this document
                words_checked.append(word)

    # we order the items by tfidf score for that term
    for term_id, docs in inverted_index_2.items():
        inverted_index_2[term_id] = sorted(inverted_index_2[term_id], key=lambda x: x[1])

    # finally we save the item in order to avoid having to create the index again
    with open(name_inv_ind_tf_idf_file, "wb") as file:
        pickle.dump(inverted_index_2, file)
    # and also the idfs array
    with open(name_idfs_file, "wb") as file:
        pickle.dump(idf_calculated, file)

    return inverted_index_2, idf_calculated  # we also return the calculated idfs so to avoid calculating them over and over


def download_inverted_index_tfidf(corpus,voc, name_inv_ind_tf_idf_file, name_idfs_file):

"""This function extracts or downloads the inverted index with the tfidfs
(depending on whether we've already created it) and returns it."""

    print('Downloading inverted index tf.idf... ', end ='')
    if (name_inv_ind_tf_idf_file not in os.listdir()) or (name_idfs_file not in os.listdir()):
        ii_2, idfs = second_inverted_index(corpus,voc, name_inv_ind_tf_idf_file, name_idfs_file)
    else :
        with open(name_inv_ind_tf_idf_file, "rb") as file:
            ii_2 = pickle.load(file)
        with open(name_idfs_file, "rb") as file:
            idfs = pickle.load(file)
    print('Done')
    return ii_2, idfs


def cosine_similarity(vec1, vec2):

"""This function computes the cosine similarity between the two
vectors provided as input"""

    num = np.dot(vec1, vec2)
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    cos = num / denom
    return cos


def tanimoto_distance(vec1, vec2):

"""This function computes the tanimoto similarity between the two
vectors provided as input"""

    num = np.dot(vec1, vec2)
    denom = np.square(np.linalg.norm(vec1)) + np.square(np.linalg.norm(vec2)) - num
    tanimoto = num / denom
    return tanimoto


def search_k_matches(query, corpus, voc, ii, idfs, urls, k=10):

"""This function finds the documents that match the query provided
by the user, creates a max-heap out of them according to their 
cosine similarity with the query and then returns the top k results."""

    # store the file with all information about the set of html pages (use at the end to return information of relevant documents)
    df = pd.read_csv("./html_df.csv")

    # apply preprocessing the query
    query = process_text(query.lower())

    # initialize the dictionary of the result
    dict_relevant = {}

    for word in query:
        if word in voc.keys():  # checks if query is in our vocabulary
            term_id = voc[word]
            # find documents with non_zero tf_idf with the "word"
            for doc_info in ii[term_id]:
                # doc_info[0] is the document_id (doc_info[1] is the associated tf-idf )
                # if it is not already in the dict_relevant, add it as key
                if doc_info[0] not in dict_relevant.keys():
                    dict_relevant[doc_info[0]] = []
                # add the associated tf-idf (term-document)
                dict_relevant[doc_info[0]].append(doc_info[1])

    len_query = len(query)
    # if a word of query is not in idfs.keys(), it means that idf(x) = 0, so no need to keep it in our vector
    query_vector = [(query.count(x) / len_query) * idfs[x] for x in query if
                    x in idfs.keys()]  # we treat the query as a document
    distances = []
    for key in dict_relevant.keys():
        vector = dict_relevant[key]
        if len(vector) == len(
                query_vector):  # this assures the conjuctive (and) property of the search engine (i.e. documents description contains every word of query)
            distances.append(
                (-cosine_similarity(query_vector, vector), key))  # negative of cosine_similarity to get max heap

    # convert "distances" as a heap
    heapq.heapify(distances)
    n_matches = len(distances)
    final_results = []
    # deal with the case where k > n_matches
    for i in range(min(k, n_matches)):
        # return and store [document_id, cosine_similarity] of the i-th best document according to the query in "final_results"
        el = heapq.heappop(distances)
        final_results.append([el[1], -el[0]])  # make the cosine distance positive again for the output

    # print(final_results)
    indices = [x[0] for x in final_results]
    distances = [x[1] for x in final_results]
    cols = ["animeTitle", "animeDescription"]
    # keep only the relevant lines
    partial_df = df.iloc[indices][cols]
    # add two columns : "Url" and "Similarity"
    final_df = partial_df.assign(Url=[urls[i] for i in indices],
                                 Similarity=distances)
    return final_df


def process_query(query):

"""This function processes a query string with the structure
'main_query [parameter1=parameter1_query, parameter2=parameter2_query...'
into its single components and returns them as a dictionary query_dict."""

    query_dict = dict()
    main_query = re.search("^(.+)\[", query)
    anime_voices = re.search("voices=\(([^\)]+)\)", query)
    anime_chars = re.search("characters=\(([^\)]+)\)", query)
    anime_related = re.search("related=\(([^\)]+)\)", query)
    year = re.search("year=([0-9]+)[,\]]", query)
    anime_type = re.search("type=([a-zA-Z]+)[,\]]", query)

    if year:
        query_dict["release_year"] = year.groups()[0]
    if anime_type:
        query_dict["anime_type"] = anime_type.groups()[0]
    # transform these fields in lists before putting them in the dictionary
    if anime_voices:
        anime_voices = anime_voices.groups()[0]
        query_dict["anime_voices"] = anime_voices.strip().split(",")

    if anime_chars:
        anime_chars = anime_chars.groups()[0]
        query_dict["anime_characters"] = anime_chars.strip().split(",")
    if anime_related:
        anime_related = anime_related.groups()[0]
        query_dict["anime_related"] = anime_related.strip().split(",")

    # preprocesses main query before putting it in the dictionary
    query_dict["main_text_query"] = process_text(main_query.groups()[0])
    # print(query_dict)

    return query_dict


# we need to treat the main query as it was its own document (I found references online on this).
# This is the same thing i implemented in 2.2.2, I just put it in a function here.
def evaluate_main_query(query, corpus, voc, ii, idfs):

"""this function finds all the documents that match at least part
of the main query"""

    dict_relevant = {}
    for word in query:
        if word in voc.keys():  # checks if query is in our vocabulary
            term_id = voc[word]
            for doc_info in ii[term_id]:
                if doc_info[0] not in dict_relevant.keys():
                    dict_relevant[doc_info[0]] = []
                dict_relevant[doc_info[0]].append(doc_info[1])

    return dict_relevant


def evaluate_parameters(query_d, df, anime_num):

"""For each parameter contained in the query dictionary, this function
evaluates the correspondance with the field in the anime_num provided
as input and returns an array of boolean and float values."""

    relevant_row = df.iloc[anime_num]
    vector = []
    for dict_key in sorted(query_d.keys()):
        if dict_key == "release_year":
            year = dt.datetime.strptime(relevant_row["releaseDate"], "%Y-%m-%d %H:%M:%S").year
            vector.append(int(str(year) == query_d[dict_key]))  # evaluates the boolean to an integer

        elif dict_key == "anime_type":
            anime_type = relevant_row["animeType"]
            score = int(anime_type.lower() == query_d[dict_key].lower())
            vector.append(score)

        elif dict_key == "anime_characters":
            chars = relevant_row["animeCharacters"].lower()
            matches = 0
            for char_query in query_d[dict_key]:
                char_query = char_query.lower()
                if char_query in chars:
                    matches += 1
            score = matches / len(query_d[dict_key])
            vector.append(score)

        elif dict_key == "anime_related":
            related = relevant_row["animeRelated"].lower()
            matches = 0
            for rel_anime_query in query_d[dict_key]:
                rel_anime_query = rel_anime_query.lower()
                if rel_anime_query in related:
                    matches += 1
            score = matches / len(query_d[dict_key])
            vector.append(score)

        elif dict_key == "anime_voices":
            voices = relevant_row["animeVoices"].lower()
            matches = 0
            for voices_query in query_d[dict_key]:
                voices_query = voices_query.lower()
                if voices_query in voices:
                    matches += 1
            score = matches / len(query_d[dict_key])
            vector.append(score)

    return vector


def get_query_with_form():

"""This function gets the query string via a form. It returns a string
with the structure 'main_query [parameter1=parameter1_query, 
parameter2=parameter2_query...' so that regardless of whether it was 
btained directly or through this form, the inner logic of the overall
algorithm doesn't change."""

    query_d = dict()
    main_query = input("Enter your query: ")
    year = input("Year it was released: ")
    anime_type = input("Type of anime: ")
    voices = input("Voice actors: ")
    characters = input("Characters: ")
    related = input("Related animes: ")

    query_string = (f"{main_query} [year={year}, anime_type={anime_type}, related=({related}),"
                    f"voices=({voices}), characters=({characters})]")

    return query_string


def search_k_matches_2(corpus, voc, ii, idfs, urls, query=None, k=10):

"""This function finds the documents that match the query provided
by the user using both the main query and the parameterized elements.
It uses the cosine distance for what concerts the tfidfs and the 
tanimoto distance (more suited to binary values) to evaluate the
correspondance between the parameters and the anime fields. 
It then obtains a single score by averaging these two distances out,
creates a max-heap and then returns the top k results."""

    df = pd.read_csv("./html_df.csv")
    if not query:
        query = get_query_with_form()
    query_dict = process_query(query)
    # here we process the main query as it was its own document with perfect match parameters.
    main_query = query_dict["main_text_query"]
    len_query = len(main_query)
    query_main_vector = [(main_query.count(x) / len_query) * idfs[x] for x in main_query if x in idfs.keys()]
    query_parameters_vector = [1] * (len(query_dict.keys()) - 1)

    # initialize the distances array that will contain all our distances and extract the synopsis
    # that match the main query
    distances = []
    dict_relevant = evaluate_main_query(main_query, corpus, voc, ii, idfs)

    # for each synopsis that matches the full main query (a condition guaranteed by the
    # "if len(vector) == len(query_main_vector)" check), we compute the cosine distance between the tfidfs
    # and the tanimoto distance between the binary values for the parameters. Then we average these
    # two values out and obtain a single score that we will put, together with the anime_num, in a tuple.
    # It's important that the first value is the distance and not the anime_num because this makes heapify
    # order the tuples correctly.
    # The strategy of taking the negative of the final score seems to be the standard approach for obtaining
    # a max heap from heapq.
    for key in dict_relevant.keys():
        vector = dict_relevant[key]
        if len(vector) == len(query_main_vector):  # this assures the conjuctive (and) property of the search engine
            cosine = cosine_similarity(query_main_vector, vector)
            vector_parameters = evaluate_parameters(query_dict, df, key)
            tanimoto = tanimoto_distance(query_parameters_vector, vector_parameters)
            distances.append((-np.mean([cosine, tanimoto]), key))  # here we negativize the mean to get a max heap
    heapq.heapify(distances)
    n_matches = len(distances)
    final_results = []
    for i in range(min(k, n_matches)):
        el = heapq.heappop(distances)
        final_results.append([el[1], -el[0]])  # make the cosine distance positive again for the output

    indices = [x[0] for x in final_results]
    distances = [x[1] for x in final_results]
    cols = ["animeTitle", "animeDescription"]
    partial_df = df.iloc[indices][cols]
    final_df = partial_df.assign(Url=[urls[i] for i in indices],
                                 Similarity=distances)
    return final_df




