from __future__ import unicode_literals, print_function

import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from spacy import displacy
from spacy.util import minibatch, compounding
import plac
import random
import warnings
from pathlib import Path
import pandas as pd


def evaluate_ner(spacy_ner_model, list_labeled_examples: list):
    count_ = 0
    scorer = Scorer()
    for input_, annotation_ in list_labeled_examples:
        count_ += 1
        doc_gold_text = spacy_ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annotation_['entities'])
        pred_value = spacy_ner_model(input_)
        scorer.score(pred_value, gold)

    return scorer.scores


def tag_all_text(text_in: str, word_to_tag: str):
    len_word = len(word_to_tag)
    # get initial positions
    matches = plac.re.finditer(word_to_tag, text_in)
    initial_positions = [match.start() for match in matches]
    # final positions
    list_positions = [(x, x + len_word) for x in initial_positions]
    return list_positions


def get_all_tagged_sentence_many_entity(text_in: str, entity_dictionary: dict):
    list_entities_positions = []
    list_words = entity_dictionary.keys()
    for word_ in list_words:
        # get list of positions for given entities
        aux_list_positions = tag_all_text(text_in, word_)
        # get instance
        aux_instance = entity_dictionary[word_]
        # update list with instance
        list_positions_instance = [(x[0], x[1], aux_instance) for x in aux_list_positions]
        # update list
        list_entities_positions.extend(list_positions_instance)

    dict_to_tag = {'entities': list_entities_positions}
    tagged_sentence = (text_in, dict_to_tag)

    return tagged_sentence


def train_ner(list_train_data: list, model=None, output_dir: str = 'models/customized_ner/', n_iter: int=100):
    """ Load the model, set up the pipeline and train the NER"""
    if model is not None:
        nlp = spacy.load(model)
        print("Loaded model '%s'" % model)
    else:
        # create blank language class
        nlp = spacy.blank("en")
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spacy
    # add labels
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in list_train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes) and warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # reset and initialize the weights randomly - but only if we are trinaing
        # a new model
        if model is None:
            nlp.begin_training(learn_rate=0.001)

        for itn in range(n_iter):
            random.shuffle(list_train_data)
            losses = {}
            # batch up the examples using spacy's minibatch
            batches = minibatch(list_train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts, # batch of text
                    annotations, # batch of annotations
                    drop=0.5, # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

        # display predictions on training data
        for text, _ in list_train_data:
            doc = nlp(text)
            print("Entities", [(ent.text, ent.label) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type, t.ent_iob) for t in doc])

        # save model to directory
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.to_disk(output_dir)
            print("Saved model to ", output_dir)

            # test the saved model
            print("Loading model from ", output_dir)
            nlp2 = spacy.load(output_dir)
            for text, _ in list_train_data:
                doc = nlp2(text)
                print("Entities", [(ent.text, ent.label) for ent in doc.ents])
                print("Tokens", [(t.text, t.ent_type, t.ent_iob) for t in doc])


def all_training_process():
    # read data and tags
    print("Starting ... ")
    df_train_data = pd.read_csv('data/training_data.csv')
    df_entities = pd.read_csv('data/entities.csv')
    print(df_entities.to_dict())
    #df_train_data['TEXT_TAGGED'] = df_train_data['NOTE_TEXT'].apply(lambda x: get_all_tagged_sentence_many_entity(x,))
    df_entities = pd.read_csv('data/entities.csv')


if __name__ == '__main__':
    # wiki_link = 'https://en.wikipedia.org/wiki/20th_century'
    #all_training_process()
    pass
