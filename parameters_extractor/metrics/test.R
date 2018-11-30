library(dplyr)
library(data.table)
library(readr)
library(rvest)
library(XML)
library(tm)
library(openNLP)
library(syuzhet)
library(dplyr)
library(slam)
library(koRpus)
library(randomForest)
library(xgboost)  # the main algorithm
library(archdata) # for the sample dataset
library(caret)

tagged.text <- tokenize("rm_me.txt", lang = "en")

textFeatures(tagged.text) -> textFeatures
uniqWd_text <- as.numeric(textFeatures[, 1])

