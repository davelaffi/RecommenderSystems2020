# RecSys kaggle competition

The application domain is book recommendation. The datasets provided contains both interactions between users and books, and tokens (words) extracted from the book text. The main goal of the competition is to discover which item (book) a user will interact with.

https://www.kaggle.com/c/recommender-system-2020-challenge-polimi

#### This repo contains a Python implementation of:
 - Item-based KNN collaborative
 - Item-based KNN content
 - User-based KNN
 - PureSVD: Matrix factorization applied using the simple SVD decomposition of the URM
 - WRMF or IALS: Matrix factorization developed for implicit interactions (Papers: <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.4684&rep=rep1&type=pdf" target="_blank">WRMF</a>, <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.5120&rep=rep1&type=pdf" target="_blank">IALS</a>)
 - P3alpha, RP3beta: graph based algorithms modelling a random walk and representing the item-item similarity as a transition probability (Papers: <a href="https://dl.acm.org/doi/abs/10.1145/2567948.2579244" target="_blank">P3alpha</a>, <a href="https://dl.acm.org/doi/10.1145/2955101" target="_blank">RP3beta</a>)
 - SLIM ElasticNet Item-item similarity matrix machine learning algorithm optimizing prediction error (MSE)
 

