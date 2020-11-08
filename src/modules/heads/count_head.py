from typing import Dict, Any, Union

import torch

from allennlp.nn.util import replace_masked_values, logsumexp, masked_softmax
from allennlp.modules import FeedForward

from src.modules.heads.head import Head

@Head.register('count_head')
class CountHead(Head):
    def __init__(self,
                 output_layer: FeedForward,
                 max_count: int,
                 training_style: str = "soft_em") -> None:
        super().__init__()
        self._output_layer = output_layer
        self._max_count = max_count
        self._training_style = training_style

    def forward(self,
                passage_summary_vector: torch.LongTensor,
                **kwargs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, max_count)
        logits = self._output_layer(passage_summary_vector)
        log_probs = torch.nn.functional.log_softmax(logits, -1)

        # Info about the best count number prediction
        # Shape: (batch_size,)
        best_count_number = torch.argmax(log_probs, -1)

        output_dict = {
            'log_probs': log_probs,
            'logits': logits,
            'best_count_number': best_count_number
        }
        return output_dict

    def _get_logprobs_for_contrastive_training(self, answer_as_counts, contrastive_answer_as_counts, log_probs):
        """
        Parameters:
        -----------
        answer_as_counts: `(B, A)` B=batch_size,  A=#-of-count-answers
        contrastive_answer_as_counts: `(B, C)` B=batch_size,  A=#-of-count-answers
        log_probs: (B, N) log-prob-distribution over all possible counts

        Returns:
        renormalized_log_probs: (B, N)
        """
        device = log_probs.device

        # Idea is to create a (B, N) sized mask which is 1 for gold and contrastive count values, 0 otherwise
        # TO achieve this, create two (B, N) masks, one each for gold and contrastive, and combine them

        # Shape: (B, C)
        contrastive_answers_mask = (contrastive_answer_as_counts != -1).long()
        # Replace -1 indices with 0
        clamped_contrastive_counts = replace_masked_values(contrastive_answer_as_counts, contrastive_answers_mask, 0)
        # Create tensor of binary values with v=0 for masked count values clamped as 1 above
        values = (torch.ones(*contrastive_answers_mask.size(), device=device) * contrastive_answers_mask).long()
        # Shape: (B, N)
        count_mask_contrast = log_probs.new_zeros(*log_probs.size(), device=device).long()
        count_mask_contrast.scatter_(1, clamped_contrastive_counts, values)
        count_mask_contrast = (count_mask_contrast >= 1).long()

        # Whole row of count_mask_contrast would be zero if no contrastive counts; in such a case, make the whole row=1
        # effectively performing no renormalization for such an instance
        # Shape: (B) mask = 1 if instance does NOT have contrastive counts
        contrast_mask = (count_mask_contrast.sum(1) == 0).long()
        # If contrast_mask[i] == 1 (instance w/ no contrastive spans), then use original mask instead
        count_mask_contrast = ((count_mask_contrast +
                                (torch.ones(*log_probs.size(), device=device) * contrast_mask.unsqueeze(1)))
                               >= 1).long()

        # Similar mask for gold count values
        gold_answers_mask = (answer_as_counts != -1).long()
        # Replace -1 indices with 0
        clamped_gold_counts = replace_masked_values(answer_as_counts, gold_answers_mask, 0)
        # Create tensor of binary values with v=0 for masked count values clamped as 1 above
        values = (torch.ones(*gold_answers_mask.size(), device=device) * gold_answers_mask).long()
        # Shape: (B, N)
        count_mask_gold = log_probs.new_zeros(*log_probs.size(), device=device).long()
        count_mask_gold.scatter_(1, clamped_gold_counts, values)
        count_mask_gold = (count_mask_gold >= 1).long()
        # Shape: (B, N)
        count_candidate_mask = ((count_mask_gold + count_mask_contrast) >= 1).long()

        renormalized_probs = masked_softmax(log_probs, count_candidate_mask, memory_efficient=True)
        renormalized_probs = replace_masked_values(renormalized_probs,
                                                   mask=count_candidate_mask, replace_with=1e-32)
        renormalized_logprobs = torch.log(renormalized_probs)

        return renormalized_logprobs


    def gold_log_marginal_likelihood(self,
                                 gold_answer_representations: Dict[str, torch.LongTensor],
                                 log_probs: torch.LongTensor,
                                 number_indices: torch.LongTensor,
                                 **kwargs: Any):
        answer_as_counts = gold_answer_representations['answer_as_counts']

        if self._training_style == "contrastive":
            # Shape:
            contrastive_answer_as_counts = gold_answer_representations["contrastive_answer_as_counts"]
            log_probs = self._get_logprobs_for_contrastive_training(answer_as_counts, contrastive_answer_as_counts,
                                                                    log_probs)

        # Count answers are padded with label -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        # Shape: (batch_size, # of count answers)
        gold_count_mask = (answer_as_counts != -1).long()
        # Shape: (batch_size, # of count answers)
        clamped_gold_counts = replace_masked_values(answer_as_counts, gold_count_mask, 0)
        log_likelihood_for_counts = torch.gather(log_probs, 1, clamped_gold_counts)
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood_for_counts = \
            replace_masked_values(log_likelihood_for_counts, gold_count_mask, -1e7)
        # Shape: (batch_size, )
        log_marginal_likelihood = logsumexp(log_likelihood_for_counts)
        return log_marginal_likelihood

    def decode_answer(self,
                      best_count_number: torch.LongTensor,
                      **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        predicted_count = best_count_number.detach().cpu().numpy().tolist()
        predicted_answer = str(predicted_count)

        answer_dict = {
            'value': predicted_answer,
            'count': predicted_count
        }
        return answer_dict
