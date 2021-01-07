from typing import Dict, Any, Union, Tuple

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

    def get_span_endpoint_logprobs(self, start_log_probs, end_log_probs, answer_as_spans):
        # Shape: (B, A)
        gold_span_starts = answer_as_spans[:, :, 0]
        gold_span_ends = answer_as_spans[:, :, 1]
        # Some spans are padded with index -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        gold_span_mask = (gold_span_starts != -1).long()
        clamped_gold_span_starts = \
            replace_masked_values(gold_span_starts, gold_span_mask, 0)
        clamped_gold_span_ends = \
            replace_masked_values(gold_span_ends, gold_span_mask, 0)
        # Shape: (B, A)
        log_likelihood_for_span_starts = \
            torch.gather(start_log_probs, 1, clamped_gold_span_starts)
        log_likelihood_for_span_ends = \
            torch.gather(end_log_probs, 1, clamped_gold_span_ends)

        log_likelihood_for_span_starts = \
            replace_masked_values(log_likelihood_for_span_starts, gold_span_mask, -1e7)
        log_likelihood_for_span_ends = \
            replace_masked_values(log_likelihood_for_span_ends, gold_span_mask, -1e7)
            # # Shape: (B, A)
        # log_likelihood_for_spans = \
        #     log_likelihood_for_span_starts + log_likelihood_for_span_ends
        # # For those padded spans, we set their log probabilities to be very small negative value
        # log_likelihood_for_spans = \
        #     replace_masked_values(log_likelihood_for_spans, gold_span_mask, -1e7)
        return (log_likelihood_for_span_starts, log_likelihood_for_span_ends)


    def get_renormalized_logprob(self, gold_logprobs, contrastive_logprobs, contrast_mask):
        """
        gold_logprobs: (B, A)
        contrastive_logprobs: (B, C)
        contrast_mask: (B)

        output: renormalized_gold_logprobs:
        """
        batch_size, num_gold_spans = gold_logprobs.size()
        _, num_contrast_spans = contrastive_logprobs.size()

        # we want to renormalize every element in (:, A) with contrastive labels (:, C),
        # In the output, every element (b, a) should have been normalized with [(b, a) + (b, :)] as the denominator
        # log(p(x)_CE) = log(p) - log(p(x) + p(x'))
        # log(p(x) + p(x')) -- can be computed as log(exp(log(p(x))) + exp(log(p(x'))))
        # That is, run logsumexp on [gold_logprobs, contrastive_logprobs]. Since for ever gold answer in gold_logprobs,
        # we want to normalize over all contrastive_logprobs, we'll create a (B, A, C+1) sized-tensor where logsumexp
        # would be performed on the last dimension.
        # combined_logprob (B, A, C+1) would be concat of G=(B, A, 1) and C=(B, 1_ex, C).

        # Shape: (B, A, C)
        contrastive_logprobs_ex = contrastive_logprobs.unsqueeze(1).expand((batch_size, num_gold_spans,
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


    def _get_contrastive_loss(self, answer_as_spans, contrastive_answer_as_spans, start_log_probs, end_log_probs):
        gold_start_logprobs, gold_end_logprobs = self.get_span_endpoint_logprobs(
            start_log_probs, end_log_probs, answer_as_spans)

        contrast_start_logprobs, contrast_end_logprobs = self.get_span_endpoint_logprobs(
            start_log_probs, end_log_probs, contrastive_answer_as_spans)

        # Shape: (B)
        contrast_mask = self._get_contrast_mask(contrastive_answer_as_spans)

        # (B, A)
        renormalized_start_logprobs = self.get_renormalized_logprob(gold_start_logprobs, contrast_start_logprobs,
                                                                    contrast_mask)
        # (B, A)
        renormalized_end_logprobs = self.get_renormalized_logprob(gold_end_logprobs, contrast_end_logprobs,
                                                                  contrast_mask)

        gold_span_starts = answer_as_spans[:, :, 0]
        # (B, A)
        gold_span_mask = (gold_span_starts != -1).long()

        log_likelihood_for_spans = \
            renormalized_start_logprobs + renormalized_end_logprobs

        log_likelihood_for_spans = \
            replace_masked_values(log_likelihood_for_spans, gold_span_mask, -1e7)

        log_marginal_likelihood_for_span = logsumexp(log_likelihood_for_spans)

        return log_marginal_likelihood_for_span


    def _get_contrast_mask(self, contrastive_answer_as_spans):
        # (B, C)
        span_starts = contrastive_answer_as_spans[:, :, 0]
        # (B, A)
        contrast_mask = (torch.max(span_starts, dim=1)[0] >= 0).long()
        return contrast_mask


    def gold_log_marginal_likelihood(self,
                                 gold_answer_representations: Dict[str, torch.LongTensor],
                                 start_log_probs: torch.LongTensor,
                                 end_log_probs: torch.LongTensor,
                                 **kwargs: Any):
        answer_as_spans = self.get_gold_answer_representations(gold_answer_representations)

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
            contrastive_answer_as_spans = self.get_contrastive_answer_representations(gold_answer_representations)
            c_log_marginal_likelihood = self._get_contrastive_loss(answer_as_spans, contrastive_answer_as_spans,
                                                                   start_log_probs, end_log_probs)
            mle_log_marginal_likelihood = logsumexp(log_likelihood_for_spans)
            contrast_mask = self._get_contrast_mask(contrastive_answer_as_spans)
            contrast_mask = contrast_mask.float()
            # mle + m*ce
            log_marginal_likelihood_for_span = mle_log_marginal_likelihood + (contrast_mask * c_log_marginal_likelihood)
        elif self._training_style == 'only_contrastive':
            contrastive_answer_as_spans = self.get_contrastive_answer_representations(gold_answer_representations)
            c_log_marginal_likelihood = self._get_contrastive_loss(answer_as_spans, contrastive_answer_as_spans,
                                                                   start_log_probs, end_log_probs)
            contrast_mask = self._get_contrast_mask(contrastive_answer_as_spans)
            c_log_marginal_likelihood = replace_masked_values(c_log_marginal_likelihood, contrast_mask, 1e-7)
            log_marginal_likelihood_for_span = c_log_marginal_likelihood
        elif self._training_style == 'topk_contrastive':
            # (B, A)
            gold_span_starts = answer_as_spans[:, :, 0]
            gold_span_ends = answer_as_spans[:, :, 1]
            # (B, K)
            _, topk_start_indices = torch.topk(start_log_probs, k=10, dim=-1)
            _, topk_end_indices = torch.topk(end_log_probs, k=10, dim=-1)

            batch_size, numg = gold_span_starts.size()
            _, numk = topk_start_indices.size()

            # Shape: (B, K, A)
            gold_start_ex = gold_span_starts.unsqueeze(1).expand(batch_size, numk, numg)
            gold_end_ex = gold_span_ends.unsqueeze(1).expand(batch_size, numk, numg)
            topk_starts_ex = topk_start_indices.unsqueeze(2).expand(batch_size, numk, numg)
            topk_ends_ex = topk_end_indices.unsqueeze(2).expand(batch_size, numk, numg)

            # Shape: (B, K)
            topk_start_mask = (gold_start_ex != topk_starts_ex).long().prod(2)
            topk_end_mask = (gold_end_ex != topk_ends_ex).long().prod(2)
            topk_mask = topk_start_mask * topk_end_mask

            topk_start_indices = replace_masked_values(topk_start_indices, topk_mask, -1)
            topk_end_indices = replace_masked_values(topk_end_indices, topk_mask, -1)

            # Shape: (B, K, 2)
            topk_contrastive_as_spans = torch.cat([topk_start_indices.unsqueeze(2), topk_end_indices.unsqueeze(2)],
                                                  dim=2)

            c_log_marginal_likelihood = self._get_contrastive_loss(answer_as_spans, topk_contrastive_as_spans,
                                                                   start_log_probs, end_log_probs)
            mle_log_marginal_likelihood = logsumexp(log_likelihood_for_spans)

            contrast_mask = self._get_contrast_mask(topk_contrastive_as_spans)
            contrast_mask = contrast_mask.float()
            # mle + m*ce
            log_marginal_likelihood_for_span = mle_log_marginal_likelihood + (contrast_mask * c_log_marginal_likelihood)
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
        (predicted_start, predicted_end) = tuple(best_span.detach().cpu().numpy())
        answer_tokens = qp_tokens[predicted_start:predicted_end + 1]
        spans_text, spans_indices = decode_token_spans([(self.get_context(), answer_tokens)], p_text, q_text)
        predicted_answer = spans_text[0]

        answer_dict = {
            'value': predicted_answer,
            'spans': spans_indices
        }
      
        return answer_dict

    def decode_topk_answers(self,
                            start_log_probs: torch.Tensor,
                            end_log_probs: torch.Tensor,
                            k: int,
                            qp_tokens: torch.LongTensor,
                            p_text: str,
                            q_text: str,
                            **kwargs: Dict[str, Any]) -> Dict[str, Any]:

        topk_logprobs, topk_bestspans = get_best_topk_spans(start_log_probs.unsqueeze(0),
                                                            end_log_probs.unsqueeze(0), k)
        # Converted (batch_size=1, k, 2) into a list
        topk_spans = topk_bestspans.detach().cpu().numpy().tolist()
        topk_spans = topk_spans[0]

        predicted_answers = []
        for best_span in topk_spans:
            (predicted_start, predicted_end) = tuple(best_span)
            answer_tokens = qp_tokens[predicted_start:predicted_end + 1]
            spans_text, spans_indices = decode_token_spans([(self.get_context(), answer_tokens)], p_text, q_text)
            predicted_answer = spans_text[0]
            predicted_answers.append(predicted_answer)

        topk_logprobs = topk_logprobs.detach().cpu().numpy().tolist()[0]

        answer_dict = {
            'values': predicted_answers,
            'logprobs': topk_logprobs
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


def get_best_topk_spans(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor,
                        k: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
    # Input: (batch_size, passage_length * passage_length)
    # Output: both are (batch_size, k)
    topk_logprobs, topk_span_indices = torch.topk(valid_span_log_probs.view(batch_size, -1), k, dim=-1)
    # (batch_size, k)
    topk_span_start_indices = topk_span_indices // passage_length
    topk_span_end_indices = topk_span_indices % passage_length

    return topk_logprobs, torch.stack([topk_span_start_indices, topk_span_end_indices], dim=-1)
