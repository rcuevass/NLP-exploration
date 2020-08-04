import spacy
from ner_utils.train_evaluate import get_tagged_data, train_customized_ner, evaluate_ner


if __name__ == '__main__':
    # read data and get it tagged
    list_training_data = get_tagged_data(path_to_csv_data='../data/training_data.csv',
                                         path_to_csv_entities='../data/entities.csv')

    dict_losses = train_customized_ner(list_train_data=list_training_data, n_iter=30)
    print(dict_losses)

    nlp_custom = spacy.load('../models/customized_ner/')
    ner_train_performance = evaluate_ner(spacy_ner_model=nlp_custom, list_labeled_examples=list_training_data)
    print(ner_train_performance)



