# This script will use the predefined document embeddings
# to create a predictor of the rating of a review.

library(jsonlite)
library(stringr)
library(MASS)
library(FactoMineR)
library(tidyverse)

# Preprocessing -----------------------------------------------------------

reviews <- fromJSON('data/reviews.json.gz')

# Drop any rows with NULL in nreviews (no reviews for that restaurant)
reviews <- reviews[map(reviews, ~!is.null(.$nreviews)) %>% unlist]

reviews <- tibble(
  eatery_id = names(reviews) %>%
    map(~rep(., length(reviews[[.]]$reviews$review_id))) %>% 
    combine(),
  review_id = reviews %>% 
    map(~.$reviews$review_id) %>% 
    combine(),
  rating = reviews %>% 
    map(~.$reviews$review_rating) %>% 
    combine()
)

# Drop the duplicates
reviews <- reviews[!duplicated(reviews$review_id), ]

reviews$rating <- reviews$rating %>% 
  map(~str_sub(., 1, 1) %>% as.integer) %>% 
  unlist

# Load train/test splitting
load('data/train.RData')
load('data/test.RData')

ids <- 1:nrow(reviews) %>% 'names<-'(reviews$review_id)
train <- ids[train_ids]
test <- ids[test_ids]
rm(ids)

# Load document embeddings
train_docs <- read_csv('data/train_embeddings.csv')
train_docs$review_id <- reviews$review_id[train]
train_docs <- left_join(train_docs, reviews, by='review_id')

test_docs <- read_csv('data/test_embeddings.csv')
test_docs$review_id <- reviews$review_id[test]
test_docs <- left_join(test_docs, reviews, by='review_id')


# Trials ------------------------------------------------------------------

train <- train_docs
test <- test_docs

X <- train %>% 
  mutate(rating=factor(rating)) %>% 
  select(-review_id, -eatery_id) %>% 
  as.data.frame

Xtest <- test %>% 
  mutate(rating=factor(rating)) %>% 
  select(-review_id, -eatery_id) %>% 
  as.data.frame

pca <- PCA(
  rbind(X, Xtest), ncp=300,
  quali.sup=301,
  ind.sup=(nrow(X) + 1):(nrow(X) + nrow(Xtest)),
  graph=FALSE
)

# Plot screeplot
# Using Kaiser Rule, obtain number of components
ncomp <- sum(pca$eig$eigenvalue > mean(pca$eig$eigenvalue))
ncomp

par(mfrow=c(1, 2))
plot(pca$eig$eigenvalue, type='o', cex=.5, main='PCA screeplot')
abline(v=ncomp, h=mean(pca$eig$eigenvalue), lty='dashed', col='red')

plot(
  pca$eig$`cumulative percentage of variance`, 
  type='o', cex=.5, 
  main='Cumulative percentage of explained variance'
)
abline(v=ncomp, lty='dashed', col='red')
par(mfrow=c(1, 1))

# Plot first factorial plane
tibble(
  dim1=pca$ind$coord[, 1],
  dim2=pca$ind$coord[, 2],
  rating=X[, 301]
) %>% 
  mutate(sentiment=as.integer(rating) > 3) %>% 
  filter(rating != 3) %>% 
  ggplot(aes(dim1, dim2, color=sentiment)) +
  geom_point(alpha=.25)

tibble(
  dim1=pca$ind.sup$coord[, 1],
  dim2=pca$ind.sup$coord[, 2],
  rating=Xtest[, 301]
) %>% 
  mutate(sentiment=as.integer(rating) > 3) %>% 
  filter(rating != 3) %>% 
  ggplot(aes(dim1, dim2, color=sentiment)) +
  geom_point(alpha=.25)

ytrain <- train_docs$rating
ytest <- test_docs$rating

Xtrain <- pca$ind$coord[ytrain != 3, 1:ncomp]
Xtest <- pca$ind.sup$coord[ytest != 3, 1:ncomp]

ytrain <- (ytrain > 3)[ytrain != 3]
ytest <- (ytest > 3)[ytest != 3]

lda_model <- lda(Xtrain, grouping=ytrain)
lda_pred <- predict(lda_model, Xtest)$class

table(real=ytest, pred=lda_pred)
mean(ytest == lda_pred) # Accuracy: 0.8861893

tibble(
  dim1 = Xtest[, 1],
  dim2 = Xtest[, 2],
  rating = (test_docs$rating)[test_docs$rating != 3]
) %>% 
  mutate(
    sentiment = rating > 3,
    rating = factor(rating),
    pred = lda_pred,
    correct = ytest == pred
  ) %>% 
  gather('real_pred', 'value', sentiment, pred, correct) %>% 
  arrange(dim1, dim2) %>% 
  ggplot(aes(dim1, dim2, color=value)) +
  geom_point() + facet_wrap(~real_pred)
  

# Division in positive/negative sentiment ---------------------------------
# We'll create a training set consisting of negative/positive entries,
# both of them balanced (with approximately the same number of entries).

# Firstly, if we want to divide in positive/negative, we should delete rating=3,
# which is ambiguous and as such cannot be tagged as positive or negative with certainty.
train_docs <- train_docs[reviews$rating[train] != 3, ]
train <- train[reviews$rating[train] != 3]
ytrain <- reviews$rating[train] > 3

table(ytrain)

# Additionally, balance train so that it contains aproximately the same number of reviews
# both positive and negative (which means, there's no bias in the resulting training set).
set.seed(123)
balanced <- sample(1:length(train), 20000, prob=(1 / 2 / table(ytrain))[ytrain + 1])

Xtrain <- train_docs[balanced, ]
ytrain <- ytrain[balanced]

Xtest <- test_docs
ytest <- ifelse(reviews$rating[test] == 3, NA, reviews$rating[test] > 3)

is.na(ytest) %>% sum # 5086
table(ytest)
# FALSE  TRUE 
# 6883 39045 


# PCA ---------------------------------------------------------------------
pca <- PCA(
  rbind(Xtrain, Xtest), 
  ncp=300, 
  ind.sup=(nrow(Xtrain) + 1):(nrow(Xtrain) + nrow(Xtest)),
  graph=FALSE
)

# Using Kaiser Rule, obtain number of components
ncomp <- sum(pca$eig$eigenvalue > mean(pca$eig$eigenvalue))
ncomp

# Plot screeplot
par(mfrow=c(1, 2))
plot(pca$eig$eigenvalue, type='o', cex=.5, main='PCA screeplot')
abline(v=ncomp, h=mean(pca$eig$eigenvalue), lty='dashed', col='red')

plot(
  pca$eig$`cumulative percentage of variance`, 
  type='o', cex=.5, 
  main='Cumulative percentage of explained variance'
)
abline(v=ncomp, lty='dashed', col='red')
par(mfrow=c(1, 1))

# Plot first factorial plane
tibble(
  dim1=pca$ind$coord[, 1],
  dim2=pca$ind$coord[, 2],
  sentiment=ytrain
) %>% 
  sample_n(10000) %>% 
  ggplot(aes(dim1, dim2, color=sentiment)) +
  geom_point(alpha=.25)

tibble(
  dim1=pca$ind.sup$coord[, 1],
  dim2=pca$ind.sup$coord[, 2],
  sentiment=ytest
) %>% 
  sample_n(10000) %>% 
  ggplot(aes(dim1, dim2, color=sentiment)) +
  geom_point(alpha=.25)


# LDA ---------------------------------------------------------------------

lda_model <- lda(Xtrain, grouping=ytrain)
lda_pred <- predict(lda_model, Xtest)$class

lda_pred[is.na(ytest)] %>% table
# FALSE  TRUE 
# 3238  1848 

table(real=ytest[!is.na(ytest)], pred=lda_pred[!is.na(ytest)])
mean(ytest[!is.na(ytest)] == lda_pred[!is.na(ytest)], na.rm=TRUE) # Accuracy: 0.301133

mean(sample(c(TRUE, FALSE), sum(!is.na(ytest)), replace=TRUE) == ytest[!is.na(ytest)])


# SVM ---------------------------------------------------------------------
library(e1071)
model <- svm(pca$ind$coord[, 1:ncomp], as.factor(ytrain), kernel='radial')
svm_pred <- predict(model, test_docs)

mean(ytest[!is.na(ytest)] == svm_pred[!is.na(ytest)], na.rm=TRUE) # Accuracy: 0.5138913

# Old trials --------------------------------------------------------------



qda_model <- qda(Xtrain, grouping=ytrain)
qda_pred <- predict(qda_model, Xtest)
qda_pred <- levels(qda_pred$class)[qda_pred$class] %>% as.integer

table(real=ytest, pred=qda_pred)
mean(ytest == qda_pred) # Accuracy: 0.2526169
mean((ytest - qda_pred) ^ 2) # MSE: 7.21945

