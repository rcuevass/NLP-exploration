from ner_utils.train_evaluate import get_tagged_data, CustomizedNer


if __name__ == '__main__':
    # read data and get it tagged
    list_training_data = get_tagged_data(path_to_csv_data='../data/training_data.csv',
                                         path_to_csv_entities='../data/entities.csv')

    # instantiate customized NER
    ner_model = CustomizedNer()
    # train it
    ner_model.train(list_training_data)
    # get dictionary of performance metrics
    dict_metrics = ner_model.predict(list_training_data)
    print(dict_metrics)




