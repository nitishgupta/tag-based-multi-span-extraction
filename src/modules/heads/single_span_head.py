from typing import Dict, Any, Union

import torch

from allennlp.nn.util import replace_masked_values, logsumexp, masked_log_softmax, masked_softmax
from allennlp.modules import FeedForward

from src.modules.heads.head import Head
from src.modules.utils.decoding_utils import decode_token_spans

class SingleSpanHead(Head):
    def __init__(self,
                 start_output_layer: FeedForward,
                 end_output_layer: FeedForward,
                 training_style: str) -> None:
        super().__init__()
        self._start_output_layer = start_output_layer
        self._end_output_layer = end_output_layer
        self._training_style = training_style

    def forward(self,                
                **kwargs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        
        input, mask = self.get_input_and_mask(kwargs)

        # Shape: (batch_size, passage_length)
        start_logits = self._start_output_layer(input).squeeze(-1)

        # Shape: (batch_size, passage_length)
        end_logits = self._end_output_layer(input).squeeze(-1)

        start_log_probs = masked_log_softmax(start_logits, mask)
        end_log_probs = masked_log_softmax(end_logits, mask)

        # Info about the best span prediction
        start_logits = replace_masked_values(start_logits, mask, -1e7)
        end_logits = replace_masked_values(end_logits, mask, -1e7)

        # Shape: (batch_size, 2)
        best_span = get_best_span(start_logits, end_logits)

        output_dict = {
            'start_log_probs': start_log_probs,
            'end_log_probs': end_log_probs,
            'best_span': best_span
        }
        return output_dict

    def _get_logprobs_for_contrastive_training(self, log_probs, answer_as_spans,
                                               contrastive_answer_as_spans, mask):
        """
        Parameters:
        -----------
        start_log_probs: (B, L)
        answer_as_spans: (B, A, 2)
        contrastive_answer_as_spans: (B, C, 2)
        mask: (B, L)

        Returns:
        --------
        renormalized_log_probs: (B, L)
        """
        batch_size, textlen = log_probs.size()

        # Shape: G = (B, A)
        gold_indices = answer_as_spans[:, :, 0]
        gold_mask = (gold_indices > -1).long()
        clamped_gold_indices = replace_masked_values(gold_indices, gold_mask, 0)
        # V = Shape (B, A)
        values = (torch.ones(*gold_indices.size()) * gold_mask).long()
        # Shape: M = (B, A, L); make M[b, a, G[b,a]] = 1 if G[b,a] is not masked using scatter.
        token_mask_gold = mask.new_zeros(*log_probs.size())
        token_mask_gold = token_mask_gold.unsqueeze(1).expand((batch_size, answer_as_spans.size()[1], textlen)).clone()
        token_mask_gold.scatter_(2, clamped_gold_indices.unsqueeze(2), values.unsqueeze(2))
        # Shape: (B, L) token is a candidate if in any gold span
        token_mask_gold = (token_mask_gold.sum(1) >= 1).long()


        # Shape: C = (B, A)
        contrast_indices = contrastive_answer_as_spans[:, :, 0]
        contrast_mask = (contrast_indices > -1).long()
        clamped_contrast_indices = replace_masked_values(contrast_indices, contrast_mask, 0)
        # V = Shape (B, A)
        values = (torch.ones(*contrast_indices.size()) * contrast_mask).long()
        # Shape: M = (B, A, L); make M[b, a, G[b,a]] = 1 if G[b,a] is not masked using scatter.
        token_mask_contrast = mask.new_zeros(*log_probs.size())
        token_mask_contrast = token_mask_contrast.unsqueeze(1).expand((batch_size,
                                                                       contrastive_answer_as_spans.size()[1],
                                                                       textlen)).clone()
        token_mask_contrast.scatter_(2, clamped_contrast_indices.unsqueeze(2), values.unsqueeze(2))
        # Shape: (B, L) token is a candidate if in any contrast span
        token_mask_contrast = (token_mask_contrast.sum(1) >= 1).long()

        # Shape: (B) mask = 1 if instance does NOT have contrastive spans. For such an instance no renormalization
        # should take place
        contrast_mask = (token_mask_contrast.sum(1) == 0).long()
        # If contrast_mask[i] == 1 (instance w/ no contrastive spans), then use original mask instead
        token_mask_contrast = ((token_mask_contrast + (mask * contrast_mask.unsqueeze(1))) >= 1).long()

        # Renormalize log_probs for these indices
        token_candidate_mask = ((token_mask_gold + token_mask_contrast) >= 1).long()

        renormalized_probs = masked_softmax(log_probs, token_candidate_mask)
        renormalized_probs = replace_masked_values(renormalized_probs,
                                                   mask=token_candidate_mask, replace_with=1e-45)
        renormalized_logprobs = torch.log(renormalized_probs)

        return renormalized_logprobs

    def gold_log_marginal_likelihood(self,
                                 gold_answer_representations: Dict[str, torch.LongTensor],
                                 start_log_probs: torch.LongTensor,
                                 end_log_probs: torch.LongTensor,
                                 **kwargs: Any):
        answer_as_spans = self.get_gold_answer_representations(gold_answer_representations)

        if self._training_style == "contrastive":
            input, mask = self.get_input_and_mask(kwargs)
            contrastive_answer_as_spans = self.get_contrastive_answer_representations(gold_answer_representations)
            start_log_probs = self._get_logprobs_for_contrastive_training(start_log_probs, answer_as_spans,
                                                                          contrastive_answer_as_spans, mask)
            end_log_probs = self._get_logprobs_for_contrastive_training(end_log_probs, answer_as_spans,
                                                                        contrastive_answer_as_spans, mask)

        # Shape: (batch_size, # of answer spans)
        gold_span_starts = answer_as_spans[:, :, 0]
        gold_span_ends = answer_as_spans[:, :, 1]
        # Some spans are padded with index -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        gold_span_mask = (gold_span_starts != -1).long()
        clamped_gold_span_starts = \
            replace_masked_values(gold_span_starts, gold_span_mask, 0)
        clamped_gold_span_ends = \
            replace_masked_values(gold_span_ends, gold_span_mask, 0)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_span_starts = \
            torch.gather(start_log_probs, 1, clamped_gold_span_starts)
        log_likelihood_for_span_ends = \
            torch.gather(end_log_probs, 1, clamped_gold_span_ends)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_spans = \
            log_likelihood_for_span_starts + log_likelihood_for_span_ends
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood_for_spans = \
            replace_masked_values(log_likelihood_for_spans, gold_span_mask, -1e7)

        # Shape: (batch_size, )
        if self._training_style == 'soft_em':
            log_marginal_likelihood_for_span = logsumexp(log_likelihood_for_spans)
        elif self._training_style == 'contrastive':
            log_marginal_likelihood_for_span = logsumexp(log_likelihood_for_spans)
        elif self._training_style == 'hard_em':
            most_likely_span_index = log_likelihood_for_spans.argmax(dim=-1)
            log_marginal_likelihood_for_span = log_likelihood_for_spans.gather(dim=1, index=most_likely_span_index.unsqueeze(-1)).squeeze(dim=-1)
        else:
            raise Exception("Illegal training_style")

        return log_marginal_likelihood_for_span

    def decode_answer(self,
                      qp_tokens: torch.LongTensor,
                      best_span: torch.Tensor,
                      p_text: str,
                      q_text: str,
                      **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        
        (predicted_start, predicted_end)  = tuple(best_span.detach().cpu().numpy())
        answer_tokens = qp_tokens[predicted_start:predicted_end + 1]
        spans_text, spans_indices = decode_token_spans([(self.get_context(), answer_tokens)], p_text, q_text)
        predicted_answer = spans_text[0]

        answer_dict = {
            'value': predicted_answer,
            'spans': spans_indices
        }
      
        return answer_dict

    def get_input_and_mask(self, kwargs: Dict[str, Any]) -> torch.LongTensor:
        raise NotImplementedError

    def get_gold_answer_representations(self, gold_answer_representations: Dict[str, torch.LongTensor]) -> torch.LongTensor:
        raise NotImplementedError

    def get_contrastive_answer_representations(self, gold_answer_representations: Dict[str, torch.LongTensor]) -> torch.LongTensor:
        raise NotImplementedError

    def get_context(self) -> str:
        raise NotImplementedError

def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
    """
    This acts the same as the static method ``BidirectionalAttentionFlow.get_best_span()``
    in ``allennlp/models/reading_comprehension/bidaf.py``. We keep it here so that users can
    directly import this function without the class.

    We call the inputs "logits" - they could either be unnormalized logits or normalized log
    probabilities.  A log_softmax operation is a constant shifting of the entire logit
    vector, so taking an argmax over either one gives the same result.
    """
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_log_mask = torch.triu(torch.ones((passage_length, passage_length),
                                          device=device)).log()
    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
    best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
    span_start_indices = best_spans // passage_length
    span_end_indices = best_spans % passage_length
    return torch.stack([span_start_indices, span_end_indices], dim=-1)
