from rouge_score import rouge_scorer


def calculate_rouge_score(reference_answer, model_answer, w_r1=.5, w_rl=0.5):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    
    scores = scorer.score(reference_answer, model_answer)
    score = scores['rouge1'].fmeasure * w_r1 + scores['rougeL'].fmeasure * w_rl
    results = {
        "rouge1_f1": round(scores['rouge1'].fmeasure, 4),
        "rougeL_f1": round(scores['rougeL'].fmeasure, 4),
        "score": round(score, 4)
    }
    
    return results
