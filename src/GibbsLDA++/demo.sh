#!/bin/sh

echo "First, convert raw data format to satisfy the requirements of Gibbs LDA"
python convertDataFormat.py -d 0

echo "Second, train the LDA model by Gibbs sampling"
src/lda -est -alpha 0.5 -beta 0.1 -ntopics 100 -niters 1000 -savestep 100 \
        -twords 20 -dfile train_reviews.dat -dir ./models/

echo "Finally, test the perplexity on testing data"
src/lda -inf -dir models/casestudy/ -model model-01800 -niters 30 \ 
        -twords 20 -dfile newdocs.dat

