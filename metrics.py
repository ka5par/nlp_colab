def calc_metrics(refs, hyps, metric="all", meteor_jar=meteor_jar):

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
    #https://github.com/nltk/nltk/issues/2365  no corpus level in NLTK
    # if metric in ("meteor", "all") and meteor_jar is not None and os.path.exists(meteor_jar):
        # could not convert string to float: 'Error: specify SCORE or EVAL or SING'
        # meteor = Meteor(meteor_jar, 'en')  # est language pack is somewhere maybe 
        # metrics["meteor"] = meteor.compute_score(hyps, many_refs)
        # nltk issues
        # metrics["meteor"] = meteor(hyps, many_refs)
    return metrics
    
def print_metrics(refs, hyps, metric="all"):
    metrics = calc_metrics(refs, hyps, metric=metric)
    print(metrics)
    print("-------------METRICS-------------")
    print("Count:\t", metrics["count"])
    #print("Ref:\t", metrics["ref_example"])
    #print("Hyp:\t", metrics["hyp_example"])

    if "bleu" in metrics:
        print("BLEU:     \t{:3.1f}".format(metrics["bleu"] * 100.0))
    if "rouge-1" in metrics:
        print("ROUGE-1-F:\t{:3.1f}".format(metrics["rouge-1"] * 100.0))
        print("ROUGE-2-F:\t{:3.1f}".format(metrics["rouge-2"] * 100.0))
        print("ROUGE-L-F:\t{:3.1f}".format(metrics["rouge-l"] * 100.0))
    if "meteor" in metrics:
        print("METEOR:   \t{:3.1f}".format(metrics["meteor"] * 100.0))
    return metrics

    