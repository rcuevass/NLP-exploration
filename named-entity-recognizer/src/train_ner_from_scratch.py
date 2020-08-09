from ner_utils.train_evaluate import get_tagged_data, CustomizedNer
from ner_utils.logger import get_log_object


if __name__ == '__main__':

    # instantiate logs
    log = get_log_object()
    # read data and get it tagged for both training and test
    log.info('Reading training and test data...')
    list_training_data = get_tagged_data(path_to_csv_data='../data_sample/training_data_small.csv',
                                         path_to_csv_entities='../data_sample/entities.csv')

    list_test_data = get_tagged_data(path_to_csv_data='../data_sample/test_data.csv',
                                     path_to_csv_entities='../data_sample/entities.csv')

    # instantiate customized NER with selected number of iterations
    num_iters = 50
    ner_model = CustomizedNer(number_iterations=num_iters)
    # train it; early stop is set by default
    log.info('Training customized NER with %i iterations', num_iters)
    ner_model.train_early_stop(list_train_data=list_training_data,
                               list_test_data=list_test_data,
                               enhancement_iteration_factor=0.050,
                               use_mini_batch=False, verbose=False)

    # get dictionary of performance metrics for training
    dict_metrics_training = ner_model.predict(list_training_data)
    log.info('Dictionary of metrics on training data %s', str(dict_metrics_training))
    precision_train = dict_metrics_training['precision']
    recall_train = dict_metrics_training['recall']
    f1_train = dict_metrics_training['F1']

    # get dictionary of performance metrics for test
    dict_metrics_test = ner_model.predict(list_test_data)
    log.info('Dictionary of metrics on test data %s', str(dict_metrics_test))
    precision_test = dict_metrics_test['precision']
    recall_test = dict_metrics_test['recall']
    f1_test = dict_metrics_test['F1']

    # create a simple txt files that captures the precision, recall and F1 for both training and test sets
    path_to_metrics = "../output_metrics/metrics.txt"
    log.info('Generating file with metrics and saving to %s', path_to_metrics)
    with open(path_to_metrics, 'w') as outfile:
        outfile.writelines(["Number of iterations: " + str(round(num_iters, 2)) + "\n",
                            "Precision training: " + str(round(precision_train, 2)) + "\n",
                            "Precision test: " + str(round(precision_test, 2)) + "\n",
                            "Recall training: " + str(round(recall_train, 2)) + "\n",
                            "Recall test: " + str(round(recall_test, 2)) + "\n",
                            "F1 training: " + str(round(f1_train, 2)) + "\n",
                            "F1 test: " + str(round(f1_test, 2)) + "\n"])


