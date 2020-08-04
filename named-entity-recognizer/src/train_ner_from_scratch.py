from ner_utils.train_evaluate import get_tagged_data, CustomizedNer


if __name__ == '__main__':
    # read data and get it tagged for both training and test
    list_training_data = get_tagged_data(path_to_csv_data='../data/training_data.csv',
                                         path_to_csv_entities='../data/entities.csv')

    list_test_data = get_tagged_data(path_to_csv_data='../data/test_data.csv',
                                     path_to_csv_entities='../data/entities.csv')

    # instantiate customized NER
    ner_model = CustomizedNer(number_iterations=35)
    # train it
    ner_model.train(list_training_data)
    # get dictionary of performance metrics for training
    dict_metrics_training = ner_model.predict(list_training_data)
    print(dict_metrics_training)

    # get dictionary of performance metrics for test
    dict_metrics_test = ner_model.predict(list_test_data)
    print(dict_metrics_test)




