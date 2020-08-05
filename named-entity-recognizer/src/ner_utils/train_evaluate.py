from __future__ import unicode_literals, print_function
import spacy
from spacy.gold import GoldParse
from spacy.matcher import Matcher
from spacy.scorer import Scorer
from spacy import displacy
from spacy.util import minibatch, compounding
import plac
import random
import warnings
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt


def get_tagged_data(path_to_csv_data: str, path_to_csv_entities: str):
    # read data and get it tagged
    df_data = pd.read_csv(path_to_csv_data)
    df_entities = pd.read_csv(path_to_csv_entities)
    list_term = list(df_entities['TERM'])
    list_entity = list(df_entities['ENTITY'])
    dict_entities = dict(zip(list_term, list_entity))

    df_data['TEXT_TAGGED'] =\
        df_data['NOTE_TEXT'].apply(lambda x: get_all_tagged_sentence_many_entity(x,
                                                                                 entity_dictionary=dict_entities))

    list_tagged_data = list(df_data['TEXT_TAGGED'])

    return list_tagged_data


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


class CustomizedNer:
    def __init__(self, number_iterations=30, learning_rate=0.01, drop_out=0.5,
                 model=None, output_dir: str = '../models/customized_ner/'):
        self.number_iterations = number_iterations
        self.learning_rate = learning_rate
        self.drop_out = drop_out
        self.model = model
        self.output_dir = output_dir
        self.dict_iter_losses = None
        self.dict_performance = None

    def train(self, list_train_data):

        print("Tagged text")
        print(list_train_data)
        print("==========================================")

        """ Load the model, set up the pipeline and train the NER"""
        if self.model is not None:
            nlp = spacy.load(self.model)
            print("Loaded model '%s'" % self.model)
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
            if self.model is None:
                nlp.begin_training(learn_rate=self.learning_rate)

            list_iters = []
            list_losses = []

            for itn in range(self.number_iterations):
                random.shuffle(list_train_data)
                losses = {}
                # batch up the examples using spacy's minibatch
                batches = minibatch(list_train_data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(
                        texts,  # batch of text
                        annotations,  # batch of annotations
                        drop=self.drop_out,  # dropout - make it harder to memorise data
                        losses=losses,
                    )
                list_iters.append(itn + 1)
                list_losses.append(losses['ner'])
                print("Losses", losses)

            # display predictions on training data
            print('Showing output in training data...')
            for text, _ in list_train_data:
                doc = nlp(text)
                matcher = Matcher(nlp.vocab)

                print("Entities", [(ent.text, matcher.vocab.strings[ent.label]) for ent in doc.ents])
                print("Tokens", [(t.text, matcher.vocab.strings[t.ent_type], t.ent_iob) for t in doc])
            print("=====================================================================")

            # save model to directory
            if self.output_dir is not None:
                output_dir = Path(self.output_dir)
                if not output_dir.exists():
                    output_dir.mkdir()
                nlp.to_disk(output_dir)
                print("Saved model to ", output_dir)

                # test the saved model
                print("Loading model from ", output_dir)
                nlp2 = spacy.load(output_dir)
                for text, _ in list_train_data:
                    doc = nlp2(text)

                    matcher = Matcher(nlp2.vocab)

                    # print("Entities", [(ent.text, ent.label) for ent in doc.ents])
                    # print("Tokens", [(t.text, t.ent_type, t.ent_iob) for t in doc])
                    print("Entities", [(ent.text, matcher.vocab.strings[ent.label]) for ent in doc.ents])
                    print("Tokens", [(t.text, matcher.vocab.strings[t.ent_type], t.ent_iob) for t in doc])

        # plot loss vs iteration
        plt.plot(list_iters, list_losses)
        plt.title("Plot of training loss for NER")
        plt.xlabel("Iteration number")
        plt.ylabel("Loss")
        plt.savefig('../plots/loss_vs_iteration.png')
        self.dict_iter_losses = dict(zip(list_iters, list_losses))
        return self.dict_iter_losses

    def predict(self, list_test_data):
        nlp_custom = spacy.load(self.output_dir)

        count_ = 0
        scorer = Scorer()
        for input_, annotation_ in list_test_data:
            count_ += 1
            doc_gold_text = nlp_custom.make_doc(input_)
            gold = GoldParse(doc_gold_text, entities=annotation_['entities'])
            pred_value = nlp_custom(input_)
            scorer.score(pred_value, gold)

        dict_perf = scorer.scores
        dict_perf_out = dict()
        dict_perf_out['precision'] = dict_perf['ents_p']
        dict_perf_out['recall'] = dict_perf['ents_r']
        dict_perf_out['F1'] = dict_perf['ents_f']
        self.dict_performance = dict_perf_out

        return self.dict_performance
