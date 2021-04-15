import json
import os
import pandas as pd
from operator import itemgetter

from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

# //TODO only for this project.
meteor_jar = 'meteor-1.5.jar'
filepath = '/content/drive/My Drive/NLP/evaluation/'


def calc_metrics(refs, hyps, metric="all", meteor_jar=meteor_jar):
    """
    Calculate all the metrics given list of refs and list of hyps. Both as list of strings.
    
    Output: dictionary in the form of {metric: score, metric2: score} 
    
    """
    metrics = dict()
    metrics["count"] = len(hyps)

    many_refs = [[r] if r is not list else r for r in refs]

    if metric in ("bleu", "all"):
        metrics["bleu"] = corpus_bleu(many_refs, hyps)
    if metric in ("rouge", "all"):
        rouge = Rouge()
        scores = rouge.get_scores(hyps, refs, avg=True)
        f_scores = {score: scores[score].get('f') for score in scores}
        metrics.update(f_scores)

    # //TODO fix meteor.
    # https://github.com/nltk/nltk/issues/2365  no corpus level in NLTK
    # if metric in ("meteor", "all") and meteor_jar is not None and os.path.exists(meteor_jar):
    # could not convert string to float: 'Error: specify SCORE or EVAL or SING'
    # meteor = Meteor(meteor_jar, 'en')  # est language pack is somewhere maybe
    # metrics["meteor"] = meteor.compute_score(hyps, many_refs)
    # nltk issues
    # metrics["meteor"] = meteor(hyps, many_refs)
    return metrics


def print_metrics(refs, hyps, metric="all"):
    """
    Prints out all the metrics in a readable format.  
    
    """
    metrics = calc_metrics(refs, hyps, metric=metric)
    print(metrics)
    print("-------------METRICS-------------")
    print("Count:\t", metrics["count"])

    if "bleu" in metrics:
        print("BLEU:     \t{:3.1f}".format(metrics["bleu"] * 100.0))
    if "rouge-1" in metrics:
        print("ROUGE-1-F:\t{:3.1f}".format(metrics["rouge-1"] * 100.0))
        print("ROUGE-2-F:\t{:3.1f}".format(metrics["rouge-2"] * 100.0))
        print("ROUGE-L-F:\t{:3.1f}".format(metrics["rouge-l"] * 100.0))
    if "meteor" in metrics:
        print("METEOR:   \t{:3.1f}".format(metrics["meteor"] * 100.0))
    return metrics


def create_json(dictionary_, filename=None, custom='', filepath=filepath):
    
    """
    Given a dictionary creates a json usable for calc_all. 
    
    Input format should be [{id: int, summary:'str'}] or [{id:int, summary:'str', model:'str'}] , where:
    
    id: id of the article, 
    summary: summary for that article
    model:  model name
    
    output: json file
     """
    if "id" not in dictionary_[0] or 'summary' not in dictionary_[0]:
        raise TypeError(
            "List of dictionaries should be in this format: [{id:int, summary:'str'}] or [{id:int, summary:'str', model:'str'}]")
    if "model" in dictionary_[0] and filename == None:
        filename = dictionary_[0]['model']
    with open(filepath + filename + custom + ".json", 'w') as f:
        f.write(json.dumps(dictionary_))
    print("Wrote to file:", filepath + filename + custom + ".json")

    
def read_json(filename, filepath=filepath):
    """ 
    aux function for jsons created by create_json
    """
    list_ = []
    with open(filepath + filename, 'r') as f:
        for line in f:
            list_.append(json.loads(line))
    return list_[0]


def calc_all(gold_filename='golds.json', filepath=filepath):
    
    """
    calculates all metrics for all the models in the filepath.
    """
    models = []

    for root, directories, file in os.walk(filepath):
        for f in file:
            if f.endswith(".json"):
                models.append(file)

    gold = read_json(gold_filename)
    models = models[0]  # //TODO <- ???
    models.remove(gold_filename)

    outputs = []

    for model in models:
        output = read_json(model)
        keys = list(map(itemgetter('id'), output))[:10]
        summaries = list(map(itemgetter('summary'), output))[:10]
        golds = [gold[key] for key in keys]

        output = {model.split('.')[0]: calc_metrics(golds, summaries)}

        outputs.append(output)

    return outputs


def pandas_calc_all(gold_filename='golds.json', filepath=filepath):
    """
    convenince function: output of calc all as pandas dataframe 
    """
    dict_ = calc_all()
    df = pd.DataFrame(dict_[0])

    for i in range(1, len(dict_)):
        col_name = str(list(dict_[i].keys())[0])
        df[col_name] = df.index.map(dict_[i][col_name])

    return df
