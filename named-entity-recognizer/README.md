# Named-entity recognition 

Folder that exemplifies use of [**spaCy**](https://spacy.io/) to train a customized named-entity recognition model.
The datasets included are fairly small as the main purpose of this folder is to provide a simple idea on how the user
can do tagging of text with entities of interest. 

There are several sub-folders included in this folder:

* data. Contains training and test data in csv format as well as a csv with two columns that capture the pairs of words
and labels the user wants to identify with the customized trained NER.
The dataset included are fairly small as the main intention is to show the process of building a customized NER.

* models. Captures the customized trained models

* plots. Includes a plot of loss vs. iteration number for both training and test sets.

* output_metrics. Sub-folder that stores metrics.txt file with precision, recall and F1 for both training and test sets.
It also includes as csv file with loss and F1 for both training and test sets. 

* notebooks. Jupyter notebooks that are used to quickly show the results of the customized NER.
