# This script will create a Language Identifier for English reviews
# based on the language_tagging.json file found in data, that comes as a result
# from the language tagging module in the python folder

library(jsonlite)
library(stringr)
library(randomForest)
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
    combine()
)

# Notice that there are some duplicates review_id
duplicated(reviews$review_id) %>% sum # 1368

# Drop the duplicates
reviews <- reviews[!duplicated(reviews$review_id), ]

# Some reviews end with \nMore. Filter them too
reviews <- reviews[!(reviews$review %>% str_detect('\\nMore\\s*$')), ]

nrow(reviews) # 504037

# Language tags
lang <- fromJSON('data/language_tagging.json')
lang <- tibble(
  review = names(lang),
  lang = unlist(lang)
)

lang <- lang %>% mutate(
  english = lang == 'english'
)

lang

lang$review_id <- str_split(lang$review, '\\|') %>% map(2) %>% unlist
lang <- lang %>% 
  select(review_id, english) %>% 
  right_join(reviews, by='review_id')


# Trigram extractor -------------------------------------------------------

# Our classifier will use as input the BOW vectors of the trigram character sequences
# in the reviews. As a result, we need to create a trigram extractor function

# We want to split by any non-letter character
# First, if we encounter any (possibly multiple) whitespace or punctuation, 
# we'll replace it by |.
# Then, we'll split by |.

extract_ngrams <- function(s, n, return_shorter=FALSE){
  if (str_length(s) >= n){ 
    indices <- 1:(str_length(s) - n + 1)
    str_sub(s, indices, indices + (n - 1))
  } else if (return_shorter & str_length(s) > 0) {
    s
  } else {
    character()
  }
}

lang$trigrams <- lang$review %>% 
  map(
    ~str_to_lower(.) %>% 
      str_replace_all("[^a-z]+", '\\|') %>% 
      str_split('\\|') %>% unlist %>% 
      map(extract_ngrams, 3, return_shorter = TRUE) %>% combine
  )


# Train/test split --------------------------------------------------------
(!is.na(lang$english)) %>% sum # 604

lang_dataset <- (!is.na(lang$english))

set.seed(123)
train <- sample((1:length(lang_dataset))[lang_dataset], size=500)
test <- setdiff((1:length(lang_dataset))[lang_dataset], train)

length(train) # 500
length(test) # 104
sum(lang_dataset) # 604


# Vocabulary definition ---------------------------------------------------
trigram_usage <- lang[train, ]$trigrams %>% combine %>% table %>% sort

length(trigram_usage) # 3518
barplot(trigram_usage)

quantile(trigram_usage, .75) # 31
vocabulary <- trigram_usage[trigram_usage >= quantile(trigram_usage, .75)] %>% names

vocabulary %>% head
# "abb" "ans" "ars" "azi" "bad" "bec"

vocabulary %>% length # 886


# Bag-Of-Trigrams ---------------------------------------------------------
bot <- lang$trigrams %>% 
  map(
    function (trigrams) {
      vocabulary %>% map(~. %in% trigrams) %>% unlist
    }
  ) %>% 
  unlist %>% 
  matrix(ncol=length(vocabulary), byrow=TRUE)


# Random Forest hyperparameter optimization -------------------------------

ntree_range <- c(seq(10, 100, 10), 150, 200, 300, 400, 500)
max_depth_range <- 1:5

parameters <- c()
oob_precision <- c()
oob_sensitivity <- c()

set.seed(1234)
for (ntree in ntree_range){
  for (max_depth in max_depth_range) {
    print(paste('ntree=', ntree, ', max_depth=', max_depth, sep=''))
    rf <- randomForest(
      bot[train, ],
      factor(lang[train, ]$english),
      
      ntree=ntree,
      maxnodes = 2 ^ max_depth
    )
    
    # Store the parameters
    parameters <- c(parameters, ntree, max_depth)
  
    # rf$confusions contains the OOB confusion matrix.
    # Thus, we can use that to evaluate the test error for each combination
    conf <- rf$confusion[, 1:2]
    
    # Compute and store the oob precision
    oob_precision <- c(oob_precision, conf[1, 1] / sum(conf[, 1]))
    oob_sensitivity <- c(oob_sensitivity, conf[1, 1] / sum(conf[1, ]))
  }
}

# Create a matrix with the OOB scores
oob <- matrix(parameters, ncol=2, byrow=TRUE) %>% 
  as_tibble %>% 
  'colnames<-'(c('ntree', 'max_depth')) %>% 
  mutate(
    precision=oob_precision, 
    sensitivity=oob_sensitivity
  )

# Plot the results
oob %>% 
  gather('metric', 'value', precision, sensitivity) %>% 
  mutate(max_depth=factor(max_depth)) %>% 
  ggplot(aes(ntree, value, color=max_depth)) +
  geom_line() + geom_text(aes(label=max_depth)) + 
  facet_wrap(~metric) +
  guides(color=FALSE) +
  ggtitle('OOB metrics for hyperparameter optimization') + 
  theme(plot.title = element_text(hjust = 0.5))

ggsave('plots/language_identifier_oob_hypopt.png')

# Our main objective is to optimize precision:
# we want to be sure that those entries that are classified as English are in fact in English.
# In that case, max_depth=4 or 5 seems the better option. 
# However, we don't want to lose data, 
# so a low sensitivity would mean that we're omitting lots of English entries in the process.
# In the plot we see that sensitivity is not highly affected by those choices,
# so we needn't worry about that.

# As for the differences in the number of trees, 
# let's plot only precision for max_depth=4 and max_depth=5.
oob %>% filter(max_depth %in% 4:5) %>% 
  mutate(max_depth = factor(max_depth)) %>% 
  ggplot(aes(ntree, precision, color=max_depth)) +
  geom_line() + geom_text(aes(label=max_depth)) +
  guides(color=FALSE) +
  ggtitle('OOB precision for hyperparameter optimization') + 
  theme(plot.title = element_text(hjust = 0.5))

ggsave('plots/language_identifier_oob_hypopt2.png')

# As we can see, oob is practically constant starting from ntree=90
oob %>% filter(max_depth %in% 4:5) %>% '$'('precision') %>% mean # 0.9942563

# We decide to keep only ntree=100 and max_depth=4 to use a simpler enough model
# that still has a very good performance


# Language Identifier Prediction ------------------------------------------

# Now, let's train the model with that combination and test it on the test set
ntree <- 100
max_depth <- 4

set.seed(12345)
rf <- randomForest(
  bot[train, ],
  factor(lang$english[train]),
  
  ntree=ntree,
  maxnodes = 2 ^ max_depth
)

conf <- table(
  real=lang$english[test],
  pred=predict(rf, bot[test, ])
)

conf # 100% accuracy; as a result, precision and sensitivity are also 100%
# This doesn't mean that the model is perfect, but it should still have a very good performance

# Lastly, let's train the model with both the train and test set 
# to finally predict the language for all the entries in the dataset
set.seed(123456)
rf <- randomForest(
  bot[c(train, test), ],
  factor(lang$english[c(train, test)]),
  
  ntree=ntree,
  maxnodes = 2 ^ max_depth
)

conf <- rf$confusion[, 1:2]

# In this case, the OOB precision and sensitivity are not 100%, but still very high
print(paste('Precision', conf[1, 1] / sum(conf[, 1]), sep=': '))
# "Precision: 0.997326203208556"

print(paste('Sensitivity', conf[1, 1] / sum(conf[1, ]), sep=': '))
# "Sensitivity: 0.994666666666667"

# Predict all the other entries
pred <- predict(rf, bot) # now bot contains ALL entries

table(pred)
# FALSE   TRUE 
# 353023 151014

# Assign the predictions to lang
lang <- mutate(lang, english = pred)

# Keep only the columns review_id and english
lang <- select(lang, review_id, english)

# And save the result for later use
write_csv(lang, 'data/lang.csv')
