"""Evaluation metrics for summarization quality."""
from typing import List, Dict, Any, Union, Optional
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import sacrebleu


class SummarizationMetrics:
    """
    Comprehensive evaluation metrics for text summarization.
    
    Includes:
    - ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
    - BLEU
    - BERTScore
    """
    
    def __init__(
        self,
        rouge_types: Optional[List[str]] = None,
        use_stemmer: bool = True
    ):
        """
        Initialize evaluation metrics.
        
        Args:
            rouge_types: Types of ROUGE to compute (default: rouge1, rouge2, rougeL)
            use_stemmer: Whether to use stemming for ROUGE
        """
        self.rouge_types = rouge_types or ['rouge1', 'rouge2', 'rougeL']
        self.rouge_scorer = rouge_scorer.RougeScorer(
            self.rouge_types,
            use_stemmer=use_stemmer
        )
    
    def compute_rouge(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str]]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            predictions: Generated summary/summaries
            references: Reference summary/summaries
            
        Returns:
            Dictionary of ROUGE scores
        """
        # Handle single string inputs
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]
        
        # Compute scores for each prediction-reference pair
        all_scores = {rouge_type: [] for rouge_type in self.rouge_types}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            for rouge_type in self.rouge_types:
                all_scores[rouge_type].append(scores[rouge_type].fmeasure)
        
        # Average scores
        avg_scores = {}
        for rouge_type in self.rouge_types:
            avg_scores[rouge_type] = np.mean(all_scores[rouge_type])
        
        return avg_scores
    
    def compute_bleu(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str], List[List[str]]]
    ) -> Dict[str, float]:
        """
        Compute BLEU score.
        
        Args:
            predictions: Generated summary/summaries
            references: Reference summary/summaries (can be multiple per prediction)
            
        Returns:
            Dictionary with BLEU score
        """
        # Handle single string inputs
        if isinstance(predictions, str):
            predictions = [predictions]
        
        # Format references for sacrebleu
        if isinstance(references, str):
            references = [[references]]
        elif isinstance(references, list) and isinstance(references[0], str):
            references = [[ref] for ref in references]
        
        # Transpose references for sacrebleu format
        # sacrebleu expects [[ref1_for_all], [ref2_for_all], ...]
        transposed_refs = []
        max_refs = max(len(refs) for refs in references)
        for i in range(max_refs):
            transposed_refs.append([
                refs[i] if i < len(refs) else refs[0]
                for refs in references
            ])
        
        # Compute BLEU
        bleu = sacrebleu.corpus_bleu(predictions, transposed_refs)
        
        return {
            'bleu': bleu.score,
            'bleu_precisions': bleu.precisions
        }
    
    def compute_bertscore(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str]],
        lang: str = 'en',
        model_type: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute BERTScore.
        
        Args:
            predictions: Generated summary/summaries
            references: Reference summary/summaries
            lang: Language code
            model_type: BERT model to use (default: auto-select)
            
        Returns:
            Dictionary with BERTScore metrics
        """
        # Handle single string inputs
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]
        
        # Compute BERTScore
        P, R, F1 = bert_score(
            predictions,
            references,
            lang=lang,
            model_type=model_type,
            verbose=False
        )
        
        return {
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item()
        }
    
    def evaluate_all(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str]],
        compute_bertscore: bool = True
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            predictions: Generated summary/summaries
            references: Reference summary/summaries
            compute_bertscore: Whether to compute BERTScore (can be slow)
            
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        # ROUGE
        rouge_scores = self.compute_rouge(predictions, references)
        results.update(rouge_scores)
        
        # BLEU
        bleu_scores = self.compute_bleu(predictions, references)
        results.update(bleu_scores)
        
        # BERTScore (optional, can be slow)
        if compute_bertscore:
            bert_scores = self.compute_bertscore(predictions, references)
            results.update(bert_scores)
        
        return results
    
    def evaluate_dataset(
        self,
        predictions: List[str],
        references: List[str],
        compute_bertscore: bool = False,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Evaluate predictions on a full dataset.
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            compute_bertscore: Whether to compute BERTScore
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with aggregated metrics and statistics
        """
        all_results = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': [],
            'bleu': []
        }
        
        if compute_bertscore:
            all_results['bertscore_f1'] = []
        
        # Evaluate in batches
        for i in range(0, len(predictions), batch_size):
            batch_preds = predictions[i:i+batch_size]
            batch_refs = references[i:i+batch_size]
            
            # ROUGE
            rouge_scores = self.compute_rouge(batch_preds, batch_refs)
            for key, value in rouge_scores.items():
                all_results[key].append(value)
            
            # BLEU
            bleu_scores = self.compute_bleu(batch_preds, batch_refs)
            all_results['bleu'].append(bleu_scores['bleu'])
            
            # BERTScore (if requested)
            if compute_bertscore:
                bert_scores = self.compute_bertscore(batch_preds, batch_refs)
                all_results['bertscore_f1'].append(bert_scores['bertscore_f1'])
        
        # Aggregate results
        aggregated = {}
        for key, values in all_results.items():
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated
    
    def print_scores(self, scores: Dict[str, float], title: str = "Evaluation Results"):
        """
        Pretty print evaluation scores.
        
        Args:
            scores: Dictionary of scores
            title: Title for the output
        """
        print(f"\n{title}")
        print("=" * 50)
        for metric, score in scores.items():
            if isinstance(score, list):
                print(f"{metric:20s}: {score}")
            else:
                print(f"{metric:20s}: {score:.4f}")
        print("=" * 50)
