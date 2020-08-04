from spacy.gold import GoldParse
from spacy.scorer import Scorer
from spacy import displacy
from spacy.util import minibatch, compounding
import plac
import random
import warnings
from pathlib import Path


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
