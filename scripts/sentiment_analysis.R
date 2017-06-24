# This script will use the predefined document embeddings
# to create a predictor of the rating of a review.

library(plotly)
library(jsonlite)
library(stringr)
library(MASS)
library(e1071)
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


# PCA ---------------------------------------------------------------------

Xtrain <- train_docs %>% 
  mutate(rating=factor(rating)) %>% 
  select(-review_id, -eatery_id) %>% 
  as.data.frame

Xtest <- test_docs %>% 
  mutate(rating=factor(rating)) %>% 
  select(-review_id, -eatery_id) %>% 
  as.data.frame

row_weights <- as.vector((1 / 5 / table(train_docs$rating))[train_docs$rating])
pca <- PCA(
  rbind(Xtrain, Xtest), ncp=300,
  ind.sup=(nrow(Xtrain) + 1):(nrow(Xtrain) + nrow(Xtest)),
  quali.sup=301,
  row.w=row_weights,
  graph=FALSE
)

# Plot screeplot
# Using Kaiser Rule, obtain number of components
ncomp <- sum(pca$eig$eigenvalue > mean(pca$eig$eigenvalue))
ncomp # 75

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

pca$eig$`cumulative percentage of variance`[ncomp] # 70.00494

# Plot first factorial plane
tibble(
  dim1=pca$ind$coord[, 1],
  dim2=pca$ind$coord[, 2],
  rating=factor(Xtrain[, 301])
) %>% 
  sample_n(10000) %>% 
  ggplot(aes(dim1, dim2, color=rating)) +
  geom_point(alpha=.25) + 
  ggtitle('Train PCA first factorial plane with rating') + 
  theme(plot.title = element_text(hjust = 0.5))

ggsave('plots/pca_ffp_train_rating.png')

tibble(
  dim1=pca$ind$coord[, 1],
  dim2=pca$ind$coord[, 2],
  rating=Xtrain[, 301]
) %>% 
  mutate(sentiment=as.integer(rating) > 3) %>% 
  filter(rating != 3) %>% 
  sample_n(10000) %>% 
  ggplot(aes(dim1, dim2, color=sentiment)) +
  geom_point(alpha=.25) + 
  ggtitle('Train PCA first factorial plane with sentiment') + 
  theme(plot.title = element_text(hjust = 0.5))

ggsave('plots/pca_ffp_train_sentiment.png')

x <- tibble(
  dim1=pca$ind$coord[, 1],
  dim2=pca$ind$coord[, 2],
  dim3=pca$ind$coord[, 3],
  rating=Xtrain[, 301]
) %>% 
  mutate(sentiment=as.integer(rating) > 3) %>% 
  filter(rating != 3) %>% 
  sample_n(10000)

plot_ly(
  x, x = ~dim1, y = ~dim2, z = ~dim3, 
  color = ~sentiment, colors = c('#FF0000', '#0000FF'),
  alpha=.75
) %>%
  add_markers %>% 
  layout(title='Train PCA first 3 components with sentiment')


# Finally, prepare the model input
Xtrain <- pca$ind$coord[, 1:ncomp]
Xtest <- pca$ind.sup$coord[, 1:ncomp]
ytrain <- train_docs$rating
ytest <- test_docs$rating

sent_Xtrain <- Xtrain[ytrain != 3, ]
sent_Xtest <- Xtest[ytest != 3, ]
sent_ytrain <- ytrain[ytrain != 3] > 3
sent_ytest <- ytest[ytest != 3] > 3


# LDA ---------------------------------------------------------------------

lda_model <- lda(Xtrain, ytrain)
lda_pred <- predict(lda_model, Xtest)

(lda_conf <- table(real=ytest, pred=lda_pred$class))
# pred
# real     1     2     3     4     5
# 1  2607   324   312   154   555
# 2   917   425   560   310   712
# 3   495   289  1007  1402  1887
# 4   262   112   642  3703  8376
# 5   223    60   301  2637 22742

(lda_acc <- sum(diag(lda_conf)) / sum(lda_conf)) # 0.5975615
(lda_mse <- mean((ytest - as.numeric(lda_pred$class)) ^ 2)) # 1.018446


lda_sent_model <- lda(sent_Xtrain, sent_ytrain)
lda_sent_pred <- predict(lda_sent_model, sent_Xtest)

(lda_sent_conf <- table(real=sent_ytest, pred=lda_sent_pred$class))
# pred
# real    FALSE  TRUE
# FALSE  4616  2260
# TRUE    846 38212

(lda_sent_acc <- sum(diag(lda_sent_conf)) / sum(lda_sent_conf)) # 0.9323812
(lda_sent_prec_pos <- lda_sent_conf[2, 2] / sum(lda_sent_conf[, 2])) # 0.9441589
(lda_sent_prec_neg <- lda_sent_conf[1, 1] / sum(lda_sent_conf[, 1])) # 0.8451117
(lda_sent_sens <- lda_sent_conf[2, 2] / sum(lda_sent_conf[2, ])) # 0.9783399
(lda_sent_spec <- lda_sent_conf[1, 1] / sum(lda_sent_conf[1, ])) # 0.6713205
(lda_sent_fscore_pos <- 2 * lda_sent_prec_pos * lda_sent_sens /
    (lda_sent_prec_pos + lda_sent_sens)) # 0.9609456
(lda_sent_fscore_neg <- 2 * lda_sent_prec_neg * lda_sent_spec / 
    (lda_sent_prec_neg + lda_sent_spec)) # 0.7482574

# QDA ---------------------------------------------------------------------

qda_model <- qda(Xtrain, ytrain)
qda_pred <- predict(qda_model, Xtest)

(qda_conf <- table(real=ytest, pred=qda_pred$class))
# pred
# real     1     2     3     4     5
# 1  2761   514   299   196   182
# 2  1240   684   486   266   248
# 3   885   918  1154  1204   919
# 4   786   951  1433  4123  5802
# 5  1189   974   996  4434 18370

(qda_acc <- sum(diag(qda_conf)) / sum(qda_conf)) # 0.5310699
(qda_mse <- mean((ytest - as.numeric(qda_pred$class)) ^ 2)) # 1.471498


qda_sent_model <- qda(sent_Xtrain, sent_ytrain)
qda_sent_pred <- predict(qda_sent_model, sent_Xtest)

(qda_sent_conf <- table(real=sent_ytest, pred=qda_sent_pred$class))
# pred
# real    FALSE  TRUE
# FALSE  5629  1247
# TRUE   4651 34407

(qda_sent_acc <- sum(diag(qda_sent_conf)) / sum(qda_sent_conf)) # 0.8715984
(qda_sent_prec_pos <- qda_sent_conf[2, 2] / sum(qda_sent_conf[, 2])) # 0.965025
(qda_sent_prec_neg <- qda_sent_conf[1, 1] / sum(qda_sent_conf[, 1])) # 0.5475681
(qda_sent_sens <- qda_sent_conf[2, 2] / sum(qda_sent_conf[2, ])) # 0.8809207
(qda_sent_spec <- qda_sent_conf[1, 1] / sum(qda_sent_conf[1, ])) # 0.8186446
(qda_sent_fscore_pos <- 2 * qda_sent_prec_pos * qda_sent_sens /
    (qda_sent_prec_pos + qda_sent_sens)) # 0.9210569
(qda_sent_fscore_neg <- 2 * qda_sent_prec_neg * qda_sent_spec / 
    (qda_sent_prec_neg + qda_sent_spec)) # 0.6562136


# SVM ---------------------------------------------------------------------

# We need a littler dataset to train, because SVM would take too much time
# We also need a train/validation splitting for choosing the right C
set.seed(123)
subtrain <- sample(1:nrow(Xtrain), 10000)
subtest <- sample(setdiff(1:nrow(Xtrain), subtrain), 1000)

c_range <- 10^(-2:3)

svm_accs <- c()
svm_mses <- c()

for (C in c_range) {
  print(paste('C = ', C, sep=''))
  
  svm_model <- svm(
    Xtrain[subtrain, ], ytrain[subtrain], 
    type='C-classification', kernel='radial', 
    class.weights=length(subtrain) / table(ytrain[subtrain]),
    cost=C
  )
  
  svm_pred <- predict(svm_model, Xtrain[subtest, ])
  
  svm_conf <- table(real=ytrain[subtest], pred=svm_pred)
  
  svm_accs <- c(svm_accs, sum(diag(svm_conf)) / sum(svm_conf))
  svm_mses <- c(svm_mses, mean((ytrain[subtest] - as.numeric(svm_pred)) ^ 2))
}

tibble(
  c = c_range,
  accuracy = svm_accs,
  mse = svm_mses
) %>% 
  gather('metric', 'value', accuracy, mse) %>% 
  ggplot(aes(log10(c), value)) +
  geom_line() + geom_text(aes(label=c)) + 
  facet_wrap(~metric) + 
  ggtitle('SVM C optimization') + 
  theme(plot.title = element_text(hjust = 0.5))

ggsave('plots/svm_c_opt.png')

# C=1 has the highest accuracy with almost the lowest MSE
C <- 1

svm_model <- svm(
  Xtrain[subtrain, ], ytrain[subtrain], 
  type='C-classification', kernel='radial', 
  class.weights=length(subtrain) / table(ytrain[subtrain]),
  cost=C
)

svm_pred <- predict(svm_model, Xtest)

(svm_conf <- table(real=ytest, pred=svm_pred))
# pred
# real     1     2     3     4     5
# 1  2226   555   495   326   350
# 2   814   506   731   492   381
# 3   390   459  1294  1824  1113
# 4   206   195  1264  5537  5893
# 5   161   152   937  6072 18641

(svm_acc <- sum(diag(svm_conf)) / sum(svm_conf)) # 0.5528678
(svm_mse <- mean((ytest - as.numeric(svm_pred)) ^ 2)) # 0.9773984


# C optimization for the sentiment problem
set.seed(123)
subtrain <- sample(1:nrow(sent_Xtrain), 10000)
subtest <- sample(setdiff(1:nrow(sent_Xtrain), subtrain), 1000)

svm_accs <- c()

for (C in c_range) {
  print(paste('C = ', C, sep=''))

  svm_sent_model <- svm(
    sent_Xtrain[subtrain, ], sent_ytrain[subtrain], 
    type='C-classification', kernel='radial', 
    class.weights=length(subtrain) / table(sent_ytrain[subtrain]),
    cost=C
  )
  
  svm_sent_pred <- predict(svm_sent_model, sent_Xtrain[subtest, ])
  
  svm_sent_conf <- table(real=sent_ytrain[subtest], pred=svm_sent_pred)
  svm_accs <- c(svm_accs, sum(diag(svm_sent_conf)) / sum(svm_sent_conf))
}

tibble(
  c = c_range,
  accuracy = svm_accs
) %>% 
  ggplot(aes(log10(c), accuracy)) +
  geom_line() + geom_text(aes(label=c)) +
  ggtitle('SVM C optimization') + 
  theme(plot.title = element_text(hjust = 0.5))

ggsave('plots/svm_sent_c_opt.png')

# C=1 wins again
C <- 1

svm_sent_model <- svm(
  sent_Xtrain[subtrain, ], sent_ytrain[subtrain], 
  type='C-classification', kernel='radial', 
  class.weights=length(subtrain) / table(sent_ytrain[subtrain]),
  cost=C
)

svm_sent_pred <- predict(svm_sent_model, sent_Xtest)
(svm_sent_conf <- table(real=sent_ytest, pred=svm_sent_pred))
# pred
# real    FALSE  TRUE
# FALSE  5404  1472
# TRUE   1695 37363

(svm_sent_acc <- sum(diag(svm_sent_conf)) / sum(svm_sent_conf)) # 0.9310533
(svm_sent_prec_pos <- svm_sent_conf[2, 2] / sum(svm_sent_conf[, 2])) # 0.962096
(svm_sent_prec_neg <- svm_sent_conf[1, 1] / sum(svm_sent_conf[, 1])) # 0.761234
(svm_sent_sens <- svm_sent_conf[2, 2] / sum(svm_sent_conf[2, ])) # 0.956603
(svm_sent_spec <- svm_sent_conf[1, 1] / sum(svm_sent_conf[1, ])) # 0.785922
(svm_sent_fscore_pos <- 2 * svm_sent_prec_pos * svm_sent_sens /
    (svm_sent_prec_pos + svm_sent_sens)) # 0.9593417
(svm_sent_fscore_neg <- 2 * svm_sent_prec_neg * svm_sent_spec / 
    (svm_sent_prec_neg + svm_sent_spec)) # 0.773381
