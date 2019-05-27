# NLP-Redditproject
_By Kailhan Hokstam and Ruben Bijl_
## Detecting threads on Reddit needing moderator attention using NLP

Make a website (github) about your project. Use that to advertise
your work! Place everything there (your code, your report, your
figures, etc.)
• The website should highlight your problem, dataset & approach,
tailored for general audience (include figures, examples, etc.)

### Introduction 
Reddit is a website that hosts a collection of forums, “subreddits”, where users can make posts. These posts, also called “threads”, including for example links to other websites and text posts, other users can then comment on these posts, and “upvote” and “downvote” each other. These subreddits are moderated by users and therefore they use their own free time to ensure a high quality of threads on the subreddits they moderate. Certain subreddits such as /r/politics can attract unwanted, uncivil discussion and /r/AskScience answers, anecdotes that do not meet their guidelines on threads leading to them being “locked” by the moderators. Thus people can not comment anymore. Locking is often used when it is not possible to look at every single comment on a thread, moderators can also delete individuals comments or whole posts.

We would like to develop a classifier that can detect if a thread should be locked, based on the comments in the thread. This could be used in combination with PRAW, “The Python Reddit API Wrapper”, to automatically lock threads or notify the moderators that a thread likely needs extra moderation, which is thought to be the case if a thread has the characteristics of a locked thread.


### The plan

We use Pushshift in combination with Python to scrape Reddit and build a dataset of threads, comments and if they were locked or not. The threads **could** be picked such that there is a bias to newer threads to reflect current moderator policy. More specifically we **would** first like to find a subreddit with a high percentage of locked threads and build the dataset specifically for that subreddit. This **will** help the classifier learn because the different subreddits have different rules and moderators that might lock threads for different reasons, which hardens the problem. From then on we **could** look at how it generalizes to other subreddits and/or create a classifier trained on multiple subreddits.  From the dataset, we **want** to create a representation per thread with a locked or unlocked label that we can feed to a classifier. _In the end, we expect to have a script in Python that can, given a thread, a document, predict how likely it is that this thread eventually would need to be locked by the moderators._ While creating this, we **want** to run experiments on this to obtain the best accuracy in finding threads needing to be locked. For evaluating our model we **want** to test the model’s performance on data we haven’t seen from Reddit before. For each of the classifiers created, we can measure recall, precision, and accuracy. Out of those results, plots of precision/recall **will be made.**

We **want** to use LSTM to transform word vectors based on the words in the threads to sentence vectors and _then use a Gated Recurrent Neural Network (GRNN)_ to combine the sentence vectors and have a document representation on which we can use the softmax function and get an output that tells how likely the corresponding thread is to be locked as described by Tang (2015) and implement this in Python using Keras. To begin with pre-trained word vectors like the Twitter set, to capture a more informal way of speaking used on social media platforms, which can be obtained from Stanford’s GloVe will be used and afterwards, GloVe can also be used to obtain vector representations based on training data gotten from Reddit. This structure **would** allow us to take relations between sentences into account. _An idea is also to insert tags into the document that represents the “level” of a comment and capture the fact that comments on a thread are not independent and might be part of a (sub)discussion._



# USE:

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/rubenbijl/NLP-Redditproject/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
