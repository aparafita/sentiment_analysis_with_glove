# This script will create the document embeddings for the reviews,
# using the word embeddings from the GloVe dataset
# as the intermediate step in the creation of the document embeddings.

library(jsonlite)
library(stringr)
library(slam)
library(tm)
library(tidyverse)

# Preprocessing -----------------------------------------------------------

reviews <- fromJSON('data/reviews.json.gz')

# Drop any rows with NULL in nreviews (no reviews for that restaurant)
reviews <- reviews[map(reviews, ~!is.null(.$nreviews)) %>% unlist]

reviews <- tibble(
  review_id = reviews %>% 
    map(~.$reviews$review_id) %>% 
    combine(),
  review = reviews %>% 
    map(~.$reviews$review_text) %>% 
    combine(),
  rating = reviews %>% 
    map(~.$reviews$review_rating %>% str_sub(., 1, 1) %>% as.integer) %>% 
    combine()
)

# Drop the duplicates
reviews <- reviews[!duplicated(reviews$review_id), ]

lang <- read_csv('data/lang.csv')

# Notice that lang has the previous filtered entries, 
# so we don't need to run the filters again, just the duplicates one
reviews <- left_join(reviews, lang, by='review_id')

# Now, we will focus on English reviews, so filter them already
reviews <- filter(reviews, english)

# Finally, how many reviews do we have?
nrow(reviews) # 151014


# GloVe loading and preprocessing -----------------------------------------
glove_dim <- 300

glove <- read.table(
  'glove/glove.6B.300d.txt.gz', sep=' ', quote="", comment.char="",
  col.names=c('word', sprintf('dim%.3d', 1:glove_dim)),
  stringsAsFactors=FALSE
)

glove <- as.matrix(glove[, -1]) %>% 'rownames<-'(glove$word)
glove_vocabulary <- rownames(glove)


# Train/test split --------------------------------------------------------
nrow(reviews) # 151014

set.seed(123)
train <- sample.int(nrow(reviews), 100000)
test <- setdiff(1:nrow(reviews), train)

length(train) # 100000
length(test) # 51014

# Don't save it yet, because there will be an additional filter just ahead

# Document embeddings (train) ---------------------------------------------

# We now need to split all reviews in sequences of words
# Let's use the tm (text mining) to preprocess the corpus and tokenize words
corpus <- reviews$review[train] %>% 
  VectorSource() %>% 
  Corpus() %>% 
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removeNumbers) %>% 
  tm_map(removeWords, stopwords("english")) %>%  
  tm_map(removePunctuation) %>% 
  tm_map(stripWhitespace)

dtm <- DocumentTermMatrix(corpus)

vocabulary <- colnames(dtm)

length(vocabulary) # 87740
vocabulary <- intersect(vocabulary, glove_vocabulary)
length(vocabulary) # 39950

save(vocabulary, file='data/vocabulary.RData')

dtm <- dtm[, vocabulary]

# Some reviews might end up with no words, 
# because the words it contains don't appear in the vocabulary.
# Filter those:
train_filter <- rowapply_simple_triplet_matrix(dtm > 0, any)

# As you see, there might be a filter here, so we need to replicate it to train
train <- train[train_filter]

# Now, we can save
train_ids <- reviews$review_id[train]
save(train_ids, file='data/train.RData')

# And finally filter dtm
dtm <- dtm[train_filter, ]

# Now, compute IDF
df <- col_sums(dtm > 0)
idf <- log(nrow(dtm) / df)

# As an example, let's look at which words have the lowest IDF
idf %>% sort %>% head
# food       good restaurant      place    service  barcelona 
# 0.5769910  0.8341249  0.9675214  0.9743602  1.0288057  1.1291337 

doc2vec <- function(doc, idf){
  # Given the TF scores of words for a given doc, computes the document embedding
  # based on the weighted average of the corresponding word GloVe vectors,
  # weighted by the TF-IDF score of those words in this document.
  non_zero <- doc > 0
  doc <- doc[non_zero] * idf[non_zero] # these are the weights
  doc <- doc / sum(doc) # weights sum to 1
  
  if (length(doc) > 1){
    colSums(doc * glove[names(doc), ]) # weighted average
  } else {
    glove[names(doc), ] # just return the unique word vector
  }
}

train_docs <- dtm %>% 
  rowapply_simple_triplet_matrix(doc2vec, idf) %>% 
  combine %>% 
  matrix(ncol=glove_dim, byrow=TRUE) %>% 
  'colnames<-'(colnames(glove)) %>% 
  as_tibble

write_csv(train_docs, 'data/train_embeddings.csv')


# Document embeddings (test) ----------------------------------------------

# We now need to split all reviews in sequences of words
# Let's use the tm (text mining) to preprocess the corpus and tokenize words
corpus_test <- reviews$review[test] %>% 
  VectorSource() %>% 
  Corpus() %>% 
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removeNumbers) %>% 
  tm_map(removeWords, stopwords("english")) %>%  
  tm_map(removePunctuation) %>% 
  tm_map(stripWhitespace)

dtm_test <- DocumentTermMatrix(corpus_test)

# test might have another vocabulary, since new words could appear
test_vocabulary <- colnames(dtm_test)

length(test_vocabulary) # 7356
test_vocabulary <- intersect(test_vocabulary, vocabulary)
length(test_vocabulary) # 5679

dtm_test <- dtm_test[, test_vocabulary]

# Some reviews might end up with no words, 
# because the words it contains don't appear in the vocabulary.
# Filter those:
test_filter <- rowapply_simple_triplet_matrix(dtm_test > 0, any)

# As you see, there might be a filter here, so we need to replicate it to train
test <- test[test_filter]

# Now, we can save
test_ids <- reviews$review_id[test]
save(test_ids, file='data/test.RData')

# And finally filter dtm
dtm_test <- dtm_test[test_filter, ]

# Now, DON'T compute IDF. We'll use the IDF from train instead
filtered_idf <- idf[test_vocabulary]

# We can use doc2vec directly
test_docs <- dtm_test %>% 
  rowapply_simple_triplet_matrix(doc2vec, filtered_idf) %>% 
  combine %>% 
  matrix(ncol=glove_dim, byrow=TRUE) %>% 
  'colnames<-'(colnames(glove)) %>% 
  as_tibble

write_csv(test_docs, 'data/test_embeddings.csv')