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


def get_tagged_data(path_to_csv_data: str, path_to_csv_entities: str) -> list:
    """
    Function that does tagging of text captured in a csv file based on a second file containing 
    two columns: word and corresponding entity for it.
    :param path_to_csv_data: string capturing the path to the csv for the data to be tagged
    :param path_to_csv_entities: string capturing the path the csv containing 
    :return: list_tagged_data: list of tagged data.
              E.g. [(This is a nice summer, {"entities":(15, 21, SEASON)})]
    """
    # read data and file that contains pair of words and entities
    df_data = pd.read_csv(path_to_csv_data)
    df_entities = pd.read_csv(path_to_csv_entities)
    # turn columns for TERM and ENTITY into lists
    list_term = list(df_entities['TERM'])
    list_entity = list(df_entities['ENTITY'])
    # make a dictionary out of those two lists
    dict_entities = dict(zip(list_term, list_entity))

    # create list of tagged text in the format expected by spaCy
    list_tagged_data = [get_all_tagged_sentence_many_entity(text_x, entity_dictionary=dict_entities) for
                        text_x in list(df_data['NOTE_TEXT'])]

    return list_tagged_data


def get_list_pos_one_word(text_in: str, word_to_tag: str) -> list:
    """
    Function that, based on input text and a words that will be tagged, returns
    a list of initial and final positions of the string where the tagged word is located
    :param text_in: string containing the text that will be tagged
    :param word_to_tag: string containing the word within text_in that will be tagged
    :return: list_positions: list containing initial and final positions within text_in where word_to_tag is located

    E.g. text_in: "This is fun. I like having fun"
         word_to_tag: "fun"
         list_positions = [(8,11),(27,30)]

    """

    #
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
        aux_list_positions = get_list_pos_one_word(text_in, word_)
        # get instance
        aux_instance = entity_dictionary[word_]
        # update list with instance
        list_positions_instance = [(x[0], x[1], aux_instance) for x in aux_list_positions]
        # update list
        list_entities_positions.extend(list_positions_instance)

    dict_to_tag = {'entities': list_entities_positions}
    tagged_sentence = (text_in, dict_to_tag)

    return tagged_sentence


def predict_on_iteration(nlp_model, list_data: list):
    count_ = 0
    scorer = Scorer()
    for input_, annotation_ in list_data:
        count_ += 1
        doc_gold_text = nlp_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annotation_['entities'])
        pred_value = nlp_model(input_)
        scorer.score(pred_value, gold)

    dict_perf = scorer.scores
    dict_perf_out = dict()
    dict_perf_out['precision'] = dict_perf['ents_p']
    dict_perf_out['recall'] = dict_perf['ents_r']
    dict_perf_out['F1'] = dict_perf['ents_f']

    return dict_perf_out


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

    def train_early_stop(self, list_train_data: list, list_test_data: list,
                         enhancement_iteration_factor: float = 0.01,
                         use_mini_batch: bool = True, early_stop: bool = True, verbose: bool = False):

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

            # reset and initialize the weights randomly - but only if we are training
            # a new model
            if self.model is None:
                nlp.begin_training(learn_rate=self.learning_rate)

            list_iters = []
            list_losses_train = []
            list_losses_test = []
            list_F1_train = []
            list_F1_test = []

            if not use_mini_batch:
                all_texts_training = [x[0] for x in list_train_data]
                all_annotations_training = [x[1] for x in list_train_data]

            all_texts_test = [x[0] for x in list_test_data]
            all_annotations_test = [x[1] for x in list_test_data]

            initial_loss = None
            for itn in range(self.number_iterations):
                random.shuffle(list_train_data)
                losses = {}
                losses_test = {}

                # batch up the examples using spacy's minibatch
                if use_mini_batch:
                    batches = minibatch(list_train_data, size=compounding(4.0, 32.0, 1.001))
                    for batch in batches:
                        texts, annotations = zip(*batch)
                        nlp.update(
                            texts,  # batch of text
                            annotations,  # batch of annotations
                            drop=self.drop_out,  # dropout - make it harder to memorise data
                            losses=losses,
                        )
                else:
                    nlp.update(all_texts_training, all_annotations_training, drop=self.drop_out, losses=losses,)

                # useful hint picked up from
                # https://github.com/explosion/spaCy/issues/3272
                nlp.update(all_texts_test, all_annotations_test, sgd=None, losses=losses_test)
                list_iters.append(itn + 1)
                if itn == 0:
                    initial_loss = losses['ner']

                list_losses_train.append(losses['ner'])
                list_losses_test.append(losses_test['ner'])

                # get metrics for training and test
                metrics_training = predict_on_iteration(nlp, list_train_data)
                metrics_test = predict_on_iteration(nlp, list_test_data)

                list_F1_train.append(metrics_training['F1'])
                list_F1_test.append(metrics_test['F1'])

                print("Losses train ", losses['ner'], " F1 train ", metrics_training['F1'])
                print("Losses test", losses_test['ner'], " F1 test ", metrics_test['F1'])
                print("===================================================")

                if early_stop:
                    if (losses_test['ner'] < losses['ner']) & (
                            losses['ner'] <= enhancement_iteration_factor * initial_loss):
                        print('Stopping at iteration number ', itn + 1)
                        break

            # display predictions on training data
            if verbose:
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

                if verbose:
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

        # dataframe for loss and F1
        df_iterations = pd.DataFrame(list(zip(list_iters,
                                              list_losses_train,
                                              list_losses_test,
                                              list_F1_train,
                                              list_F1_test)),
                                     columns=['ITERATION', 'LOSS_TRAINING', 'LOSS_TEST',
                                              'F1_TRAINING', 'F1_TEST'])
        df_iterations.to_csv('../output_metrics/iterations_metrics.csv', index=False)
        # plot loss vs iteration
        plt.plot(list_iters, list_losses_train, 'b-o', label='training')
        plt.plot(list_iters, list_losses_test, 'r-s', label='test')
        plt.legend(loc='upper right', numpoints=1)
        plt.title("Plot of training and test loss for NER")
        plt.xlabel("Iteration number")
        plt.ylabel("Loss")
        plt.savefig('../plots/loss_vs_iteration.png')
        self.dict_iter_losses = dict(zip(list_iters, list_losses_train))
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
