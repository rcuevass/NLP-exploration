from spacy.gold import GoldParse
from spacy.scorer import Scorer


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
