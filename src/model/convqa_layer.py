import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class MultiLinearLayer(nn.Module):
    def __init__(self, config, output_size, expand_size=1):
        super(MultiLinearLayer, self).__init__()
        self.dense1 = nn.Linear(config.hidden_size * expand_size, config.hidden_size)
        self.dense2 = nn.Linear(config.hidden_size, output_size)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = getattr(F, "relu")

    def forward(self, x):
        # x = torch.tanh(self.dropout(self.dense1(x)))
        x = self.activation(self.dense1(x))
        # x = self.dense2(self.dropout(x))
        x = self.dense2(x)
        return x


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, class_num, expand_size=1):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * expand_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, class_num)

    def forward(self, hidden_states, cls_index, **kwargs):
        # x = features[:, cls_index, :]  # take <s> token (equiv. to [CLS])
        hsz = hidden_states.shape[-1]
        cls_index = cls_index[:, None].expand(-1, hsz)  # shape (bsz, hsz)
        x = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, hsz)

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ClsPredLayer(nn.Module):

    def __init__(self, config, expand_size=1):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * expand_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        cls_index: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:

        # No dependency on end_feature so that we can obtain one single `cls_logits` for each sample.
        hsz = hidden_states.shape[-1]

        if cls_index is not None:
            cls_index = cls_index[:, None].expand(-1, hsz)  # shape (bsz, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, -1, :]  # shape (bsz, hsz)

        x = self.dense_0(cls_token_state)
        x = self.activation(x)
        x = self.dense_1(x)

        return x



class PoolerTriAnswerClass(nn.Module):
    """
    Compute SQuAD 2.0 answer class from classification and start tokens hidden states.
    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model, will be used to grab the :obj:`hidden_size` of the model.
    """

    def __init__(self, config, binary_labels=False):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1_unk  = nn.Linear(config.hidden_size, 1, bias=False) 
        self.dense_1_yes = nn.Linear(config.hidden_size, 1, bias=False) if binary_labels else nn.Linear(config.hidden_size, 3, bias=False) 
        self.dense_1_no = nn.Linear(config.hidden_size, 1, bias=False) if binary_labels else nn.Linear(config.hidden_size, 3, bias=False)


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        cls_index: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`, `optional`):
                The hidden states of the first tokens for the labeled span.
            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                The position of the first token for the labeled span.
            cls_index (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Position of the CLS token for each sentence in the batch. If :obj:`None`, takes the last token.
        .. note::
            One of ``start_states`` or ``start_positions`` should be not obj:`None`. If both are set,
            ``start_positions`` overrides ``start_states``.
        Returns:
            :obj:`torch.FloatTensor`: The SQuAD 2.0 answer class.
        """
        # No dependency on end_feature so that we can obtain one single `cls_logits` for each sample.
        hsz = hidden_states.shape[-1]
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2)  # shape (bsz, hsz)

        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, -1, :]  # shape (bsz, hsz)

        x = self.dense_0(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)

        unk_logits = self.dense_1_unk(x).squeeze(-1)
        yes_logits = self.dense_1_yes(x).squeeze(-1)
        no_logits = self.dense_1_no(x).squeeze(-1)

        return unk_logits, yes_logits, no_logits