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
from ner_utils.logger import get_log_object
from matplotlib import pyplot as plt


# instantiate log object
log = get_log_object()

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

    # get length of word based on which the tagging will be done
    len_word = len(word_to_tag)
    # Find each instance on input text where word_to_tag is found and ...
    matches = plac.re.finditer(word_to_tag, text_in)
    # get initial positions for each instance ...
    initial_positions = [match.start() for match in matches]
    # ... and from it find the boundaries, (initial_position, final_position), as a list
    list_positions = [(x, x + len_word) for x in initial_positions]

    # this is the list to be returned
    return list_positions


def get_tuple_training_spacy(text_in: str, entity_dictionary: dict) -> tuple:
    """
    Function that, based on an input txt and a dictionary of entities, returns tuple
    of input text and dictionary of positions and entities. This output is the format
    required by spacy to do the training
    :param text_in: string provided  that will be tagged
    :param entity_dictionary: dictionary that will be used to do the tagging.
                              keys: words to tag
                              values: entities corresponding to each key
    :return: tagged_sentence: tuple in the expected spacy format to do training
                              E.g. ("Summer in Canada is very nice",{"entities":[(0, 6, "SEASON"), (10, 16, "GPE")]})
    """

    # initialize list of entities with corresponding positions
    list_entities_positions = []
    # extract list of entities from dictionary
    list_words = entity_dictionary.keys()
    # loop over words in list of keys just extracted...
    for word_ in list_words:
        # get list of positions for given entities
        aux_list_positions = get_list_pos_one_word(text_in, word_)
        # get instance
        aux_instance = entity_dictionary[word_]
        # create list of instances and positions for each element in list of positions
        list_positions_instance = [(x[0], x[1], aux_instance) for x in aux_list_positions]
        # update list
        list_entities_positions.extend(list_positions_instance)

    # create dictionary of entities-list of positions
    dict_to_tag = {'entities': list_entities_positions}
    # create required tuple....
    tagged_sentence = (text_in, dict_to_tag)

    # and return it
    return tagged_sentence


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
    list_tagged_data = [get_tuple_training_spacy(text_x, entity_dictionary=dict_entities) for
                        text_x in list(df_data['NOTE_TEXT'])]

    return list_tagged_data


def predict_on_iteration(nlp_model, list_data: list) -> dict:
    count_ = 0
    scorer = Scorer()
    for input_, annotation_ in list_data:
        count_ += 1
        doc_gold_text = nlp_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annotation_['entities'])
        pred_value = nlp_model(input_)
        scorer.score(pred_value, gold)

    dict_perf = scorer.scores
    dict_perf_out = {'precision': dict_perf['ents_p'], 'recall': dict_perf['ents_r'], 'F1': dict_perf['ents_f']}

    return dict_perf_out


class CustomizedNer:
    """
    Class used to train a NER from scratch
    It contains the following methods:

    - train_early_stop. Method that performs training of NER and allows for the following options:
         * mini_batch as part of training
         * "early stop" based on enhancement_iteration_factor. The training is stopped at iteration ith if
            the loss at that iteration is less than or equal to the loss if the first iteration times
            the enhancement_iteration_factor

    - predict. Make predictions on a given dataset

    """
    def __init__(self, number_iterations: int = 50, learning_rate: float = 0.01,
                 drop_out: float = 0.5,
                 model=None, output_dir: str = '../models/customized_ner/'):
        """
        Initialization method
        :param number_iterations: integer that sets the number of iterations for training - default = 50
        :param learning_rate: float set for the learning rate during the training process - default = 0.01
        :param drop_out: float used to reduce the possibilities of over-fitting - default = 0.5
        :param model: object that will store the trained NER - default = None
        :param output_dir: string indicating the path where the trained model will be saved -
                           default = ../models/customized_ner/
        """
        # initialize arguments based on description above
        self.number_iterations = number_iterations
        self.learning_rate = learning_rate
        self.drop_out = drop_out
        self.model = model
        self.output_dir = output_dir
        self.dict_iter_losses = None
        self.dict_performance = None

    def train_early_stop(self, list_train_data: list, list_test_data: list,
                         early_stop: bool = True,
                         enhancement_iteration_factor: float = 0.01,
                         use_mini_batch: bool = True, verbose: bool = False):
        """
        Method that performs training of customized NER
        :param list_train_data: list of training data given the format expected by spaCy
                                E.g. [(This is a nice summer, {"entities":(15, 21, SEASON)})]
        :param list_test_data: list of test data given the format expected by spaCy.
                               Formatting like the one for training
        :param early_stop:
        :param enhancement_iteration_factor: float used to judge when the training process
                                             should be stopped - default = 0.01
        :param use_mini_batch:boolean to indicate if mini-batch processing will be
                              used during training - default = True
        :param verbose: boolean - option used to control the info displayed shown to the user during training
                                - default = False
        :return: dict_iter_losses - dictionary where keys are iteration number and values are loss value at such

        """

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

            # initialize lists capturing number of iterations, loss value and F1 for both test and training
            list_iters = []
            list_losses_train = []
            list_losses_test = []
            list_F1_train = []
            list_F1_test = []

            # decide if mini-batch will be used or not
            if not use_mini_batch:
                all_texts_training = [x[0] for x in list_train_data]
                all_annotations_training = [x[1] for x in list_train_data]

            # extract separate lists for texts and corresponding annotations
            all_texts_test = [x[0] for x in list_test_data]
            all_annotations_test = [x[1] for x in list_test_data]

            # initialize value for loss in the first iteration
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
                    # if not mini-batch option selected, update model using whole training dataset
                    nlp.update(all_texts_training, all_annotations_training, drop=self.drop_out, losses=losses,)

                # obtain loss value, at this iteration, for test set.
                # sgd = None indicates the weights will not be updated
                # useful hint picked up from
                # https://github.com/explosion/spaCy/issues/3272
                nlp.update(all_texts_test, all_annotations_test, sgd=None, losses=losses_test)

                # append iteration number to list of iterations..
                list_iters.append(itn + 1)
                # get the value of loss for the first iteration; it will be needed to asses when to stop the
                # training process
                if itn == 0:
                    initial_loss = losses['ner']

                # append losses for training and test
                list_losses_train.append(losses['ner'])
                list_losses_test.append(losses_test['ner'])

                # get metrics for training and test
                metrics_training = predict_on_iteration(nlp, list_train_data)
                metrics_test = predict_on_iteration(nlp, list_test_data)
                list_F1_train.append(metrics_training['F1'])
                list_F1_test.append(metrics_test['F1'])

                # display values of loss and F1 for both training and test
                log.info("Losses train=%f,  F1 train=%f ", losses['ner'], metrics_training['F1'])
                log.info("Losses test=%f,  F1 test=%f ", losses['ner'], metrics_test['F1'])
                log.info("===================================================")

                # if early stop is being used, compare the loss at the given iteration with
                # the loss of the initial iteration factored by enhancement_iteration_factor...
                if early_stop:
                    if (losses_test['ner'] < losses['ner']) & (
                            losses['ner'] <= enhancement_iteration_factor * initial_loss):
                        log.info('Stopping at iteration number %i', itn + 1)
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

        # plot loss vs iteration for both training and test..
        plt.plot(list_iters, list_losses_train, 'b-o', label='training')
        plt.plot(list_iters, list_losses_test, 'r-s', label='test')
        plt.legend(loc='upper right', numpoints=1)
        plt.title("Plot of training and test loss for NER")
        plt.xlabel("Iteration number")
        plt.ylabel("Loss")
        plt.savefig('../plots/loss_vs_iteration.png')
        self.dict_iter_losses = dict(zip(list_iters, list_losses_train))
        return self.dict_iter_losses

    def predict(self, list_data):
        """
        Method that performs prediction on a given dataset
        :param list_data: list of data given in the format expected by spaCy
                                E.g. [(This is a nice summer, {"entities":(15, 21, SEASON)})]
        :return: dict_performance - dictionary where keys are precision, recall and F1 and values are the
                                    corresponding values of such metrics
        """

        # load customized NER model
        nlp_custom = spacy.load(self.output_dir)

        # instantiate scorer
        scorer = Scorer()

        # loop over list of data given
        for input_, annotation_ in list_data:
            doc_gold_text = nlp_custom.make_doc(input_)
            gold = GoldParse(doc_gold_text, entities=annotation_['entities'])
            pred_value = nlp_custom(input_)
            scorer.score(pred_value, gold)

        # create dictionary to be returned...
        dict_perf = scorer.scores
        dict_perf_out = {'precision': dict_perf['ents_p'], 'recall': dict_perf['ents_r'], 'F1': dict_perf['ents_f']}
        self.dict_performance = dict_perf_out

        # ... and return it
        return self.dict_performance
