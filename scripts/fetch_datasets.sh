#!/bin/bash
## fetch datasets
cd data

## dredze multi domain sentiment 
wget http://www.cs.jhu.edu/~mdredze/datasets/sentiment/unprocessed.tar.gz
wget http://www.cs.jhu.edu/~mdredze/datasets/sentiment/processed_acl.tar.gz
wget http://www.cs.jhu.edu/~mdredze/datasets/sentiment/processed_stars.tar.gz
mkdir Multi_Domain_Sentiment_Dataset
mv processed_acl.tar.gz processed_stars.tar.gz unprocessed.tar.gz Multi_Domain_Sentiment_Dataset/
cd Multi_Domain_Sentiment_Dataset
tar -xzf processed_acl.tar.gz 
tar -xzf processed_stars.tar.gz 
tar -xzf unprocessed.tar.gz
cd ..

## small amazon reviews set
wget http://times.cs.uiuc.edu/~wang296/Data/LARA/Amazon/AmazonReviews.zip
mkdir AmazonReviews 
mv AmazonReviews.zip AmazonReviews
cd AmazonReviews 
unzip AmazonReviews.zip 
cd ..

## stanford sentiment
wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebankRaw.zip
wget http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
mkdir StanfordSentiment
mv stanfordSentimentTreebank.zip stanfordSentimentTreebankRaw.zip trainDevTestTrees_PTB.zip StanfordSentiment 
cd StanfordSentiment
unzip stanfordSentimentTreebank.zip 
unzip stanfordSentimentTreebankRaw.zip 
unzip trainDevTestTrees_PTB.zip
cd ..
