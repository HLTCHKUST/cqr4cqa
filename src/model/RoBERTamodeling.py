from typing import Optional, Tuple

from transformers import PreTrainedModel, RobertaConfig, RobertaModel
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    QuestionAnsweringModelOutput,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from src.model.convqa_layer import MultiLinearLayer

MAX_FLOAT = 1e30
MIN_FLOAT = -1e30

class ConversationalQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    yes_logits: torch.FloatTensor = None
    no_logits: torch.FloatTensor = None
    unk_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class RobertaForConversationalQuestionAnswering(RobertaPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(
            self,
            config,
            beta=100,
            rational=True,
            put_mask_in_train=False,
    ):
        super(RobertaForConversationalQuestionAnswering, self).__init__(config)
        self.roberta = RobertaModel(config)
        hidden_size = config.hidden_size
        self.rational_l = MultiLinearLayer(config, 1)
        self.logits_l = MultiLinearLayer(config, 2)
        self.unk_l = MultiLinearLayer(config, 1, expand_size=2) if rational else MultiLinearLayer(config, 1)
        self.attention_l = MultiLinearLayer(config, 1)
        # if use_auxiliary_loss:
        #     self.yn_l = MultiLinearLayer(config, 3, expand_size=2) if rational else MultiLinearLayer(config, 3)
        #     self.followup_l = MultiLinearLayer(config, 3, expand_size=2) if rational else MultiLinearLayer(config, 3)
        # else:
        self.yn_l = MultiLinearLayer(config, 2, expand_size=2) if rational else MultiLinearLayer(config, 2)
        
        self.beta = beta
        self.rational = rational
        self.put_mask_in_train = put_mask_in_train

        self.init_weights()
    
    def _generate_masked_data(self, input_data, input_mask):
        """Generate masked data"""
        return input_data * (1 - input_mask) + MIN_FLOAT * input_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            p_mask=None,
            rational_mask=None,
            cls_idx=None,
            yes_no_labels=None,
            followup_labels=None,
            **kwargs,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # last layer hidden-states of sequence, first token:classification token
        final_hidden, pooled_output = outputs['last_hidden_state'], outputs['pooler_output']

        # get span logits
        logits = self.logits_l(final_hidden)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)

        p_mask = p_mask.type(final_hidden.dtype)

        if self.rational:
            # rational_logits
            rational_logits = self.rational_l(final_hidden)
            rational_logits = torch.sigmoid(rational_logits)
            final_hidden = final_hidden * rational_logits

            # attention layer to cal logits
            attention = self.attention_l(final_hidden).squeeze(-1)
            attention.data.masked_fill_(attention_mask.eq(0), -float('inf'))
            attention = F.softmax(attention, dim=-1)
            attention_pooled_output = (attention.unsqueeze(-1) * final_hidden).sum(dim=-2)

            # concate the augmented pooled output with the original one for yes/no/unk prediction
            pooled_output = torch.cat((pooled_output, attention_pooled_output), dim=-1)
            # pooled_output = attention_pooled_output
        
        if p_mask is not None and self.put_mask_in_train: 
            # Here we only apply paragraph mask during the inference time
            start_logits = self._generate_masked_data(start_logits, p_mask)
            end_logits = self._generate_masked_data(end_logits, p_mask)
        
        if self.rational:
            # on to find answer in the article
            rational_logits = rational_logits.squeeze(-1)
            # start_logits, end_logits = start_logits * rational_logits, end_logits * rational_logits 

        # cal unkown/yes/no logits
        unk_logits = self.unk_l(pooled_output)
        yn_logits = self.yn_l(pooled_output)
        yes_logits, no_logits = yn_logits.split(1, dim=-1)

        # start_positions and end_positions is None when evaluate
        # return loss during training
        # return logits during evaluate
        total_loss = None
        if start_positions is not None and end_positions is not None:

            start_positions, end_positions = start_positions + cls_idx, end_positions + cls_idx
    
            new_start_logits = torch.cat((yes_logits, no_logits, unk_logits, start_logits), dim=-1)
            new_end_logits = torch.cat((yes_logits, no_logits, unk_logits, end_logits), dim=-1)
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = new_start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            span_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            start_loss = span_loss_fct(new_start_logits, start_positions)
            end_loss = span_loss_fct(new_end_logits, end_positions)
            qa_loss = (start_loss + end_loss) / 2


            # rational part
            alpha = 0.25
            gamma = 2.

            # use rational span to help calculate loss
            if self.rational:
                rational_mask = rational_mask.type(final_hidden.dtype)
                # rational_loss = -alpha * ((1 - rational_logits) ** gamma) * rational_mask * torch.log(
                #     rational_logits + 1e-7) \
                #                 - (1 - alpha) * (rational_logits ** gamma) * (1 - rational_mask) * \
                #                 torch.log(1 - rational_logits + 1e-7)
                rational_loss = - rational_mask * torch.log(rational_logits + 1e-7) - (1 - rational_mask) * torch.log(1 - rational_logits + 1e-7)
                rational_loss = (rational_loss * (1 - p_mask)).sum() / (1 - p_mask).sum()

                total_loss = qa_loss + rational_loss * self.beta
            else:
                total_loss = qa_loss

        elif not self.put_mask_in_train and p_mask is not None: 
            # Here we only apply paragraph mask during the inference time
            start_logits = self._generate_masked_data(start_logits, p_mask)
            end_logits = self._generate_masked_data(end_logits, p_mask)
        
        if not return_dict:
            output = (start_logits, end_logits, yes_logits, no_logits, unk_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return ConversationalQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            yes_logits=yes_logits, # can also be yes_no logits for QuAC
            no_logits=no_logits, # can also be followup logits for QuAC
            unk_logits=unk_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, class_num):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, class_num)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForConversationalQuestionAnsweringWithClassHead(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config, class_num=1):
        super().__init__(config)
        self.class_num = class_num # quac: 1 class (answerable),

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier = RobertaClassificationHead(config, class_num)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        is_impossible=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        class_logits = self.classifier(sequence_output)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            
            if self.class_num < 2: # quac
                class_loss_fct = BCEWithLogitsLoss()
                class_loss = class_loss_fct(class_logits.squeeze(), is_impossible.squeeze())
            else: # coqa, but coqa doesn't use this model class, coqa use RobertaForConversationalQuestionAnswering instead
                class_loss_fct = CrossEntropyLoss(ignore_index=3)
                class_loss = class_loss_fct(class_logits, is_impossible)
            
            total_loss = (start_loss + end_loss + class_loss) / 3

        if not return_dict:
            output = (start_logits, end_logits, class_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# if __name__ == '__main__':
#     from transformers import AutoConfig
#     config = AutoConfig.from_pretrained('roberta-base')
#     model = RobertaForConversationalQuestionAnswering.from_pretrained('roberta-base', config=config)
#     print(model)