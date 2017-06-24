# Sentiment Analysis with GloVe

Project to study the power of GloVe word embeddings 
in predicting the sentiment of a restaurant review.

Author: Álvaro Parafita

In order to execute all the necessary code, one should download all data files 
from <a href="https://mega.nz/#!OoI2iT7Z!ulpOM3p-gZy4AZutDch9rsdbNUQOapg1rQ_K8qeXoHg">here</a> and store them in a *data* folder in the project root folder. 

Additionally, for the GloVe vectors, one should download them from <a href="https://nlp.stanford.edu/projects/glove/">here</a>, the dataset called <a href="http://nlp.stanford.edu/data/glove.6B.zip>glove.6B.zip</a>, and store the text files inside this ZIP in a folder called *glove* in the project root folder.


# Usage:

The 'python' folder contains the language_tagger.py Python module, 
which gives the tool to tag more entries with a language.

In order to execute it, one should use Python3 
and run the following command line in a UNIX terminal:

python3 python/language_tagger.py data/reviews.json.gz data/language_tagging.json


The 'scripts' folder contains the R scripts that analyize the dataset 
(data_exploration.R) and the ones that compute all the steps in the model:
* language_imputation: language identifier model.
* embeddings: creates the document embeddings.
* sentiment_analysis: uses the results from the previous scripts to create the sentiment analysis models.


Folder 'plots' contains all the plots created in the R scripts.


In order to reproduce the results explained in the report, 
one need only execute all the scripts in the *scripts* folder in the order
depicted above. At the start of each script, the needed packages are loaded.
One should install all those packages before running the script.

Note that this project worked with a huge dataset and, 
thus, some instructions take minutes or even hours to execute.
