from ner_utils.train_evaluate import get_tagged_data, CustomizedNer


if __name__ == '__main__':
    # read data and get it tagged for both training and test
    list_training_data = get_tagged_data(path_to_csv_data='../data/training_data_small.csv',
                                         path_to_csv_entities='../data/entities.csv')

    list_test_data = get_tagged_data(path_to_csv_data='../data/test_data.csv',
                                     path_to_csv_entities='../data/entities.csv')

    # instantiate customized NER
    num_iters = 50
    ner_model = CustomizedNer(number_iterations=num_iters)
    # train it
    #ner_model.train(list_training_data)
    ner_model.train_early_stop(list_train_data=list_training_data,
                               list_test_data=list_test_data,
                               enhancement_iteration_factor=0.08,
                               use_mini_batch=False, verbose=False)
    # get dictionary of performance metrics for training
    dict_metrics_training = ner_model.predict(list_training_data)
    print(dict_metrics_training)
    precision_train = dict_metrics_training['precision']
    recall_train = dict_metrics_training['recall']
    f1_train = dict_metrics_training['F1']

    # get dictionary of performance metrics for test
    dict_metrics_test = ner_model.predict(list_test_data)
    print(dict_metrics_test)
    precision_test = dict_metrics_test['precision']
    recall_test = dict_metrics_test['recall']
    f1_test = dict_metrics_test['F1']

    with open("../output_metrics/metrics.txt", 'w') as outfile:
        outfile.writelines(["Number of iterations: " + str(round(num_iters, 2)) + "\n",
                            "Precision training: " + str(round(precision_train, 2)) + "\n",
                            "Precision test: " + str(round(precision_test, 2)) + "\n",
                            "Recall training: " + str(round(recall_train, 2)) + "\n",
                            "Recall test: " + str(round(recall_test, 2)) + "\n",
                            "F1 training: " + str(round(f1_train, 2)) + "\n",
                            "F1 test: " + str(round(f1_test, 2)) + "\n"])


