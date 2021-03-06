# NLP-Redditproject
_By Kailhan Hokstam and Ruben Bijl_
## Detecting threads on Reddit needing moderator attention using NLP

### THE PLAN
Reddit is one of the largest communities on the internet. Users of Reddit, also called “Redditors” can submit, view and discuss information on subcommunities called “Subreddits”. With more and more people using the internet and online discourse hardening, it seems that phenomena like cyberbullying, hate speech, and online abuse, colloquially named “toxicity” are becoming more commonplace. Because moderation can cost a lot of time, and specifically on Reddit moderators are not paid, and with the rise of machine learning related methods like deep learning, trying to apply machine learning to moderation is of interest.

The aim of this project is to build a classifier that can detect threads that need moderator attention. More specifically, we want to look at a subreddit called “/r/BlackPeopleTwitter”. The idea being that afterwards a bot could periodically assess if newly made threads are likely to be locked in the future and could then be lead in the right direction by extra moderation to nip the problem in the bud, while in the normal scenario moderators cannot check every thread and will lock the thread after the thread got out of hand.

The data used to analyze ‘/r/BlackPeopleTwitter’ has been obtained using Python and the Reddit API, but because of recently imposed limitations on this API only a 1000 posts can be retrieved from Reddit itself. For every thread the comments were flattened using a breadth-first traversal using a queue, the resulting representation of comments, the thread ID, if the thread is locked was then stored in a database. This traversal does mean that a significant amount of information with regards to the hierarchical structure of comments is lost. Because there is about a 90/10 ratio of non-locked to locked threads and there was only a 1000 posts, we decided to use oversampling to balance the locked and non-locked classes out.

For oversampling the data a method is made to divide the data equally. This is a important process for the Reddit threads, since the data consists of almost 1000 threads, all different amount of comments. This means that a few threads with the most amount of comments will be as large a dataset as a big set of threads with a small amount of comments. For oversampling the data of the locked threads, the data is divided in locked threads and non-locked threads, sorted by the amount of comments in those threads. The total amount of either locked or not locked comments is divided by 10. 

![alt text](https://i.imgur.com/p04GK4G.png)

Our model consisted of Stanford’s GloVe pre trained word embedding layer provided by Stanford GloVe’s algorithm, specifically the 25 dimensional Twitter word vectors. This is connected with a LSTM layer of 128 units, which corresponds with the dimensionality of this layers output space. The last layer is a single standard densely-connected NN layer using a sigmoid activation function. The model is compiled with as objective function  a standard binary cross-entropy loss, the adam optimizer and as metric accuracy.

### THE RESULTS
The results indicate that it is possible to detect if a thread can be locked using our model using LSTM and therefore we can say with >90% according to our results if a thread is locked or not, but a livestream of threads would be needed to determine how accurately this can be determined when a thread is not actually locked yet.

The results obtained using the custom made model were very good, however these results could not be replicated on Pang and Lee's Movie Review Dataset strongly suggesting that the data scraped contains features that only appear after a thread has been locked for the locked threads or the dataset is in some other way strongly biased, because the model performs very weakly on a benchmark that is known to be valid. The fact that the SVM actually performs on par and even slightly better than the model using LSTM also supports this fact, because according to other benchmarks, a LSTM based model should outperform a SVM on sentiment analysis related tasks, but this is not the case in these experiments.

In further projects it would be interesting to not ‘flatten’ the comments of a thread, so that the structure of the thread could be taken into account, longer comment chains might imply discussions, which can lead to abusive behaviour. Due to limited resources it was not possible to implement a neural network as proposed by Tang, Qin & Lio (2015), but such a method that tries to represent the structure of in their case a document with paragraphs and in this paper a thread with comments, seems promising.

| Metrics | GloVe Twitter embeddings + LSTM on Reddit Dataset (n = 1770) | 
| --- | --- |
| Accuracy | 0.937677 | 
| Precision | 0.889447 | 
| Recall | 1.000000 | 
| F1 score | 0.941489 |
| Cohen’s kappa | 0.875309 |
| ROC AUC | 0.962555 | 

