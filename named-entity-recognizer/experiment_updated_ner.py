#!/usr/bin/env python
# coding: utf8
"""Example of training an additional entity type

This script shows how to add a new entity type to an existing pretrained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.

After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.1.0+
Last tested with: v2.2.4
"""
from __future__ import unicode_literals, print_function

import plac
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

# test the trained model
# test_text = "I went to England and I was surprised to see James Bond riding a horse"
test_text = "The name is Bond, James Bond riding a horse"

# test the saved model
output_dir = 'models/animals_persons/'
print("Loading from", output_dir)
output_dir = Path(output_dir)
nlp2 = spacy.load(output_dir)
# Check the classes have loaded back consistently
doc2 = nlp2(test_text)
for ent in doc2.ents:
    print(ent.label_, ent.text)


#if __name__ == "__main__":
#    plac.call(main)