# ADM 
## Homework 3 - What is the best anime in the world?

##### Group members : Alessandro Caioli, Ludovico Lentini, Aurélie Pommier and Syed Muhamman Hassan Raza 

**This assignment aims to develop good habits for Web-scraping and Text Mining.**


This reposistory contains 
- One notebook ```main.ipynb``` which gathers the answers of the homework. In this notebook, there are both the commands that display the answers to the questions and also markdowns that explain the steps.
- One python file ```utilities.py``` which gathers every functions that we wrote used in ```main.ipynb```. There are markdowns to explain it.
- Different files used to store intermediate results :
	- ```urls.txt``` : text file which contains URLs of pages
	- ```counter_pages``` : counter of last saved page (1.2)
	- ```df_with_tokens.p``` : dataset of preprocessed descriptors (corpus)
	- ```vocabulary.pkl``` : whole vocabulary of the corpus
	- ```idfs.p``` : binary file of the list of idfs
	- ```inverted_index.pkl``` : inverted index (without tf-idf) (2.1.1)
	- ```inverted_index_2.p``` : inverted index with tf-idf (2.2.1)

However, ```html_pages```, ```tsv_files``` folders and ```html_df.csv``` mentionned in the notebook are not in the github repository because they are to heavy.
