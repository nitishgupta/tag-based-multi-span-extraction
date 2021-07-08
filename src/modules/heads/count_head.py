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

    def _get_contrast_mask(self, contrastive_answer_as_counts):
        # contrastive_answer_as_counts : (B, C)
        # contrast_mask: (B)
        contrast_mask = (torch.max(contrastive_answer_as_counts, dim=1)[0] >= 0).long()
        return contrast_mask

    def get_renormalized_logprob(self, gold_logprobs, contrastive_logprobs, contrast_mask):
        """
        gold_logprobs: (B, A)
        contrastive_logprobs: (B, C)
        contrast_mask: (B)

        output: renormalized_gold_logprobs:
        """
        batch_size, num_gold_counts = gold_logprobs.size()
        _, num_contrast_spans = contrastive_logprobs.size()

        # This is exactly the same as single-span contrastive loss
        # log(p(x)_CE) = log(p) - log(p(x) + p(x'))
        # log(p(x) + p(x')) -- can be computed as log(exp(log(p(x))) + exp(log(p(x'))))
        # That is, run logsumexp on gold_logprobs + contrastive_logprobs. Since for ever gold answer in gold_logprobs,
        # we want to normalize over all contrastive_logprobs, we'll create a (B, A, C+1) sized-tensor where logsumexp
        # would be performed on the last dimension.
        # combined_logprob (B, A, C+1) would be concat of G=(B, A, 1) and C=(B, 1_ex, C).

        # Shape: (B, A, C)
        contrastive_logprobs_ex = contrastive_logprobs.unsqueeze(1).expand((batch_size, num_gold_counts,
                                                                            num_contrast_spans))
        # Shape: (B, A, C+1)
        combined_logprobs = torch.cat([gold_logprobs.unsqueeze(2), contrastive_logprobs_ex], dim=2)
        # Shape: (B, A)
        log_denominator = logsumexp(combined_logprobs, dim=2)
        # log(p(x) + p(x')) = 0 for instances without contrastive labels
        log_denominator = log_denominator * contrast_mask.unsqueeze(1).float()
        # Shape: (B, A)
        renormalized_logprob = gold_logprobs - log_denominator

        return renormalized_logprob


    def _get_countanswer_logprobs(self, answer_as_counts, log_probs):
        """
        answer_as_counts: (B, A)
        log_probs: (B, 10)

        log_likelihood_for_counts: (B, A)
        """
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

        return log_likelihood_for_counts


    def _get_contrastive_loss(self, answer_as_counts, contrastive_answer_as_counts, log_probs):
        """
        answer_as_counts: (B, A)
        contrastive_answer_as_counts: (B, C)
        log_probs: (B, 10)
        """

        # (B, A)
        gold_count_logprobs = self._get_countanswer_logprobs(answer_as_counts, log_probs)
        # (B, C)
        contrast_count_logprobs = self._get_countanswer_logprobs(contrastive_answer_as_counts, log_probs)

        # Shape: (B)
        contrast_mask = self._get_contrast_mask(contrastive_answer_as_counts)

        # Shape: (B, A)
        renormalized_logprobs = self.get_renormalized_logprob(gold_count_logprobs, contrast_count_logprobs,
                                                              contrast_mask)

        # Shape: (B, A)
        gold_count_mask = (answer_as_counts != -1).long()

        renormalized_logprobs = \
            replace_masked_values(renormalized_logprobs, gold_count_mask, -1e7)

        log_marginal_likelihood = logsumexp(renormalized_logprobs)

        return log_marginal_likelihood


    def gold_log_marginal_likelihood(self,
                                 gold_answer_representations: Dict[str, torch.LongTensor],
                                 log_probs: torch.LongTensor,
                                 number_indices: torch.LongTensor,
                                 **kwargs: Any):
        answer_as_counts = gold_answer_representations['answer_as_counts']

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

        if self._training_style == 'contrastive':
            contrastive_answer_as_counts = gold_answer_representations["contrastive_answer_as_counts"]
            c_log_marginal_likelihood = self._get_contrastive_loss(answer_as_counts, contrastive_answer_as_counts,
                                                                   log_probs)
            mle_log_marginal_likelihood = logsumexp(log_likelihood_for_counts)
            contrast_mask = self._get_contrast_mask(contrastive_answer_as_counts)
            contrast_mask = contrast_mask.float()
            # mle + m*ce
            log_marginal_likelihood = mle_log_marginal_likelihood + (contrast_mask * c_log_marginal_likelihood)
        elif self._training_style == 'only_contrastive':
            contrastive_answer_as_counts = gold_answer_representations["contrastive_answer_as_counts"]
            c_log_marginal_likelihood = self._get_contrastive_loss(answer_as_counts, contrastive_answer_as_counts,
                                                                   log_probs)
            contrast_mask = self._get_contrast_mask(contrastive_answer_as_counts)
            c_log_marginal_likelihood = replace_masked_values(c_log_marginal_likelihood, contrast_mask, 1e-7)
            log_marginal_likelihood = c_log_marginal_likelihood
        elif self._training_style == 'topk_contrastive':
            _, topk_counts = torch.topk(log_probs, k=4, dim=-1)
            batch_size, numg = answer_as_counts.size()
            _, numk = topk_counts.size()
            # Shape: (B, K, A)
            gold_counts_ex = answer_as_counts.unsqueeze(1).expand(batch_size, numk, numg)
            topk_counts_ex = topk_counts.unsqueeze(2).expand(batch_size, numk, numg)
            # Shape: (B, K)
            topk_counts_mask = (gold_counts_ex != topk_counts_ex).long().prod(2)
            topk_counts = replace_masked_values(topk_counts, topk_counts_mask, -1)

            c_log_marginal_likelihood = self._get_contrastive_loss(answer_as_counts, topk_counts,
                                                                   log_probs)
            mle_log_marginal_likelihood = logsumexp(log_likelihood_for_counts)
            contrast_mask = self._get_contrast_mask(topk_counts)
            contrast_mask = contrast_mask.float()
            # mle + m*ce
            log_marginal_likelihood = mle_log_marginal_likelihood + (contrast_mask * c_log_marginal_likelihood)
        else:
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

    def decode_topk_answers(self,
                            k: int,
                            log_probs: torch.Tensor,
                            **kwargs: Dict[str, Any]):

        # Info about the best count number prediction
        # Shape: (k) each
        k = min(k, 10)
        topk_logprobs, topk_counts = torch.topk(log_probs, k, dim=-1)

        topk_logprobs = topk_logprobs.detach().cpu().numpy().tolist()
        topk_counts = topk_counts.detach().cpu().numpy().tolist()
        topk_counts = [str(c) for c in topk_counts]

        topk_answer_dict = {
            'logprobs': topk_logprobs,
            'values': topk_counts
        }
        return topk_answer_dict


