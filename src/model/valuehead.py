# modified from https://github.com/lvwerra/trl/blob/df428af25faf485851ae67e61974b8f4e48e485d/trl/gpt2.py
from torch import nn
from torch.nn import Identity


class ValueHead(nn.Module):
    """The ValueHead class implements a head for GPT2 that returns a scalar for each output token."""
    def __init__(self, config):
        super().__init__()
        self.detach_head = False

        self.summary = nn.Linear(config.hidden_size, 1)
        self.activation = Identity()

        self.first_dropout = nn.Dropout(0.1)

        self.last_dropout = Identity()

        self.flatten = nn.Flatten()

    def forward(self, hidden_states, cls_index=None):
        if self.detach_head:
            output = hidden_states.detach()
        else:
            output = hidden_states
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output