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


# new entity label
LABEL_ANIMAL = 'ANIMAL'
LABEL_PERSON = 'PERSON'
LABEL_ORGANIZATION = 'ORG'

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
TRAIN_DATA = [
    (
        "Horses are too tall and they pretend to care about your feelings",
        {"entities": [(0, 6, LABEL_ANIMAL)]},
    ),
    ("Do they bite?", {"entities": []}),
    (
        "horses are too tall and they pretend to care about your feelings",
        {"entities": [(0, 6, LABEL_ANIMAL)]},
    ),
    ("horses pretend to care about your feelings", {"entities": [(0, 6, LABEL_ANIMAL)]}),
    (
        "they pretend to care about your feelings, those horses",
        {"entities": [(48, 54, LABEL_ANIMAL)]},
    ),
    ("horses?", {"entities": [(0, 6, LABEL_ANIMAL)]}),

    ("James Bond is very cool", {"entities": [(0, 10, LABEL_PERSON)]}),

    ("The name is Bond, James Bond", {"entities": [(18, 28, LABEL_PERSON)]}),

    ("Is he around?", {"entities": []}),

    ("Working is not fun", {"entities": []}),

    ("I do not like working!", {"entities": []}),

    ("Would you like to work for Northwest?", {"entities": [(27, 36, LABEL_ORGANIZATION)]}),

    ("James Bond seems to be very strong. Himself however, James Bond, does not think so",
     {"entities": [(0, 10, LABEL_PERSON), (53, 63, LABEL_PERSON)]}),

    ("James Bond has a beautiful horse", {"entities": [(0, 10, LABEL_PERSON), (27, 32, LABEL_ANIMAL)]}),

    ("Northwest is a fantastic company!, let's go to Northwest", {"entities": [(0, 9, LABEL_ORGANIZATION),
                                                                               (47, 56, LABEL_ORGANIZATION)]}),

    ("I would love to work for Northwest", {"entities": [(25, 34, LABEL_ORGANIZATION)]}),



]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, new_model_name="animal", output_dir='models/wild/', n_iter=30):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(2020)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(LABEL_ANIMAL)  # add new entity label to entity recognizer
    ner.add_label(LABEL_PERSON)
    ner.add_label(LABEL_ORGANIZATION)
    # Adding extraneous labels shouldn't mess anything up
    ner.add_label("VEGETABLE")
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes) and warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_text_list = ["Do you like horses?", "Who is James Bond", "Is Northwest a good company?"]
    for test_text in test_text_list:
        doc = nlp(test_text)
        print("Entities in '%s'" % test_text)
        for ent in doc.ents:
            print(ent.label_, ent.text)
        print("===========================================================================")

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":
    plac.call(main)