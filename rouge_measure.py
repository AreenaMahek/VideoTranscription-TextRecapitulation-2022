from rouge import Rouge
import numpy as np

def rouge_scoring(summary, text):
    rouge = Rouge()
    score = rouge.get_scores(summary, text)[0]
    print(score)
    recall_score = np.mean([score["rouge-1"]["r"], score["rouge-2"]["r"], score["rouge-l"]["r"]])
    precision_score = np.mean([score["rouge-1"]["p"], score["rouge-2"]["p"], score["rouge-l"]["p"]])
    mean_score = np.mean([score["rouge-1"]["f"], score["rouge-2"]["f"], score["rouge-l"]["f"]])
    print("Average Recall=",recall_score)
    print("Average Precision=", precision_score)
    print("Average F-Score=", mean_score)
    return mean_score
    

