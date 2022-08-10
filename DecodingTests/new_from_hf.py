import torch
import torch.nn as nn
import numpy as np
import pickle 
from transformers import (
    LogitsProcessorList,
    StoppingCriteriaList,
    BeamScorer,
    BeamSearchScorer,
    PreTrainedModel,
 )
from transformers.generation_utils import (
    BeamSearchOutput,
    BeamSearchDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSampleOutput,
    BeamSampleDecoderOnlyOutput,
    BeamSampleEncoderDecoderOutput,
)
# from transformers.pytorch_utils import torch_int_div
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

def gmf(x1, x2, p=1, w1=1, w2=1):
    if p is None or p == 0:
        return (x1 * x2) ** 0.5
    elif p == np.inf:
        return torch.maximum(x1, x2)
    elif p == -np.inf:
        return torch.minimum(x1, x2)
    return ((1. / (w1 + w2)) * (((w1 * x1) ** p) + ((w2 * x2) ** p))) ** (1./p)

def beam_search(
    model,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = False,
    model2: Optional[PreTrainedModel] = None,
    model2_kwargs: Optional[dict] = None,
    max_vocab_size: Optional[int] = 1e9,
    gmf_kwargs: Optional[dict] = {'p': 1, 'w1': 1, 'w2': 1},
    **model_kwargs,
) -> Union[BeamSearchOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        beam_scorer (`BeamScorer`):
            An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.
    Return:
        [`generation_utilsBeamSearchDecoderOnlyOutput`], [`~generation_utils.BeamSearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation_utils.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation_utils.BeamSearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    Examples:
    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForSeq2SeqLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     BeamSearchScorer,
    ... )
    >>> import torch
    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    >>> encoder_input_str = "translate English to German: How old are you?"
    >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
    >>> # lets run beam search using 3 beams
    >>> num_beams = 3
    >>> # define decoder start token ids
    >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
    >>> input_ids = input_ids * model.config.decoder_start_token_id
    >>> # add encoder_outputs to model keyword arguments
    >>> model_kwargs = {
    ...     "encoder_outputs": model.get_encoder()(
    ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
    ...     )
    ... }
    >>> # instantiate beam scorer
    >>> beam_scorer = BeamSearchScorer(
    ...     batch_size=1,
    ...     num_beams=num_beams,
    ...     device=model.device,
    ... )
    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
    ...     ]
    ... )
    >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)
    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Wie alt bist du?']
    ```"""
    if model2 is not None:
        assert model2_kwargs is not None # TODO: better error messages
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
    output_scores = output_scores if output_scores is not None else model.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only
    while True:

        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        if model2 is not None:
            model2_inputs = model2.prepare_inputs_for_generation(input_ids, **model2_kwargs)
            outputs2 = model2(
                **model2_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            print(f'Exiting because synced_gpus = {synced_gpus} and this_peer_finished = {this_peer_finished}')
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
        # cannot be generated both before and after the `nn.functional.log_softmax` operation.
        next_token_logits = model.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)

        if model2 is not None:
            next_token_logits2 = outputs2.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits2 = model2.adjust_logits_during_generation(next_token_logits2, cur_len=cur_len)
            # Important !
            trim_size = min(min(next_token_logits.size(-1), next_token_logits2.size(-1)), max_vocab_size)
            #print(f'Trimming to {trim_size}, logits1.size() = {next_token_logits.size()}, logits2.size() = {next_token_logits2.size()}')
            next_token_scores = torch.log(
                    gmf(
                        nn.functional.softmax(
                            next_token_logits[..., :trim_size], dim=-1
                        ), nn.functional.softmax(
                            next_token_logits2[..., :trim_size], dim=-1
                        ), **gmf_kwargs
                    )
                ) # (batch_size * num_beams, vocab_size)
        else:
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)
        #print(f'next_token_scores.size() = {next_token_scores.size()}')
        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        #print(f'next_token_scores_processed.size() = {next_token_scores_processed.size()}')
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)
        #print(f'next_token_scores_added.size() = {next_token_scores.size()}')

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_attentions:
                if model2 is None:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                else:
                    decoder_attentions += (
                        (outputs.decoder_attentions, outputs2.decoder_attentions)
                        if model.config.is_encoder_decoder and model2.config.is_encoder_decoder else
                        (outputs.attentions, outputs2.attentions)
                    )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.size(-1)
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )

        #next_indices = torch_int_div(next_tokens, vocab_size)
        next_indices = next_tokens // vocab_size
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )

        if model2 is not None:
            model2_kwargs = model2._update_model_kwargs_for_generation(
                outputs2, model2_kwargs, is_encoder_decoder=model2.config.is_encoder_decoder
            )

        if model_kwargs["past"] is not None:
            model_kwargs["past"] = model._reorder_cache(model_kwargs["past"], beam_idx)
 
        if model2 is not None and model2_kwargs["past"] is not None:
            # This seems superfluos 
            # Whenever first model "past" becomes non-None, 
            # second model's "past" will also become non-None 
            # TODO: study this properly
            model2_kwargs["past"] = model2._reorder_cache(model2_kwargs["past"], beam_idx)       

        if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True
    
    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None
        else:
            num_return_sequences = beam_scorer.num_beam_hyps_to_keep
            # return only as many indices as sequences
            beam_indices = tuple(
                (beam_indices[i * num_beams : i * num_beams + num_return_sequences] for i in range(batch_size))
            )
            beam_indices = sum(beam_indices, ())

        if model.config.is_encoder_decoder:
            return BeamSearchEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                #beam_indices=beam_indices,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return BeamSearchDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=beam_indices,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return sequence_outputs["sequences"]

def beam_sample(
    self,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = False,
    **model_kwargs,
) -> Union[BeamSampleOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **beam search multinomial
    sampling** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        beam_scorer (`BeamScorer`):
            A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        logits_warper (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.
    Return:
        [`~generation_utils.BeamSampleDecoderOnlyOutput`], [`~generation_utils.BeamSampleEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation_utils.BeamSampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation_utils.BeamSampleEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    Examples:
    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForSeq2SeqLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     TopKLogitsWarper,
    ...     TemperatureLogitsWarper,
    ...     BeamSearchScorer,
    ... )
    >>> import torch
    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    >>> encoder_input_str = "translate English to German: How old are you?"
    >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
    >>> # lets run beam search using 3 beams
    >>> num_beams = 3
    >>> # define decoder start token ids
    >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
    >>> input_ids = input_ids * model.config.decoder_start_token_id
    >>> # add encoder_outputs to model keyword arguments
    >>> model_kwargs = {
    ...     "encoder_outputs": model.get_encoder()(
    ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
    ...     )
    ... }
    >>> # instantiate beam scorer
    >>> beam_scorer = BeamSearchScorer(
    ...     batch_size=1,
    ...     max_length=model.config.max_length,
    ...     num_beams=num_beams,
    ...     device=model.device,
    ... )
    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id)]
    ... )
    >>> # instantiate logits processors
    >>> logits_warper = LogitsProcessorList(
    ...     [
    ...         TopKLogitsWarper(50),
    ...         TemperatureLogitsWarper(0.7),
    ...     ]
    ... )
    >>> outputs = model.beam_sample(
    ...     input_ids, beam_scorer, logits_processor=logits_processor, logits_warper=logits_warper, **model_kwargs
    ... )
    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Wie alt bist du?']
    ```"""
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    output_scores = output_scores if output_scores is not None else self.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only
    while True:

        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
        # cannot be generated both before and after the `nn.functional.log_softmax` operation.
        next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (logits_warper(input_ids, next_token_scores_processed),)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        probs = nn.functional.softmax(next_token_scores, dim=-1)

        next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
        next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

        next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
        next_tokens = torch.gather(next_tokens, -1, _indices)

        next_indices = torch_int_div(next_tokens, vocab_size)
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs["past"] is not None:
            model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

        if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None
        else:
            num_return_sequences = beam_scorer.num_beam_hyps_to_keep
            # return only as many indices as sequences
            beam_indices = tuple(
                (beam_indices[i * num_beams : i * num_beams + num_return_sequences] for i in range(batch_size))
            )
            beam_indices = sum(beam_indices, ())

        if self.config.is_encoder_decoder:
            return BeamSampleEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=beam_indices,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return BeamSampleDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=beam_indices,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return sequence_outputs["sequences"]

if __name__ == '__main__':
    with open('./args_22-06-2022_170008.pkl', 'rb') as f:
        args = pickle.load(f)
        model_kwargs = args.pop('model_kwargs')
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").cuda()
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    out = beam_search(model, **args, **model_kwargs)
    for sequence in tokenizer.batch_decode(out['sequences']):
        print(sequence)
