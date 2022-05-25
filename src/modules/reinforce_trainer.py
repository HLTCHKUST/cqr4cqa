from transformers import AdamW, get_scheduler
import torch
from torch.cuda.amp import GradScaler
from torch.nn import CrossEntropyLoss

from src.utils.ppo_utils import logprobs_from_logits


class REINFORCETrainer:
    """
    The REINFORCETrainer uses REINFORCE to optimise language models.
    """
    def __init__(self, model, args):
        self.args = args
        self.model = model
        self.counter = 0
        self.grad_accum = self.args.grad_accum
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon)
            
        if self.args.use_lr_scheduler:
            self.scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.num_training_steps,
            )
        else:
            self.scheduler = None
        
        # FP16 scaler
        if self.args.fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def step(self, model_input, position_ids, attention_mask, token_type_ids, gen_len, scores, 
             decoder_input_ids=None, decoder_attention_mask=None, num_ce_tokens=0, ce_labels=None):
        """
        Run a REINFORCE optimization step.

        """
        # Calculate logits from the given input
        if decoder_input_ids is None:
            with torch.no_grad():
                output = self.model(model_input[:,:-gen_len-1], position_ids=position_ids[:,:-gen_len-1], 
                            attention_mask=attention_mask[:,:-gen_len-1], token_type_ids=token_type_ids[:,:-gen_len-1], use_cache=True)
            past_key_values = output["past_key_values"]

            output = self.model(
                model_input[:,-gen_len-1:-1], past_key_values=past_key_values, position_ids=position_ids[:,-gen_len-1:-1], 
                attention_mask=attention_mask[:,:-1], token_type_ids=token_type_ids[:,-gen_len-1:-1]
            )
            lm_logits = output["logits"]
        else:
            output = self.model(
                model_input, attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask
            )
            lm_logits = output["logits"]
        
        # Calculate rewards
        rewards = self.compute_rewards(scores)
        
        # Compute Loss
        loss = self.compute_loss(
            rewards, lm_logits, model_input, gen_len, attention_mask, 
            decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, 
            num_ce_tokens=num_ce_tokens, ce_labels=ce_labels
        )

        if self.grad_accum > 1:
            loss = loss / self.grad_accum
        loss.backward()
        
        # Update model
        self.update_model()
        
        # Record statistics
        stats = self.record_step_stats(loss=loss)
        return stats

    def compute_rewards(self, rewards):
        bs = rewards.shape[0]
        if self.args.reinforce_type == "with_baseline":
            rewards =  rewards - rewards.mean(dim=0)
        elif self.args.reinforce_type == "self_critic":
            bs = rewards.shape[0] // 2
            rewards = rewards[:bs] - rewards[bs:]

        rewards = rewards.reshape(bs, 1)
        return rewards

    def compute_loss(self, rewards, lm_logits, model_input, gen_len, attention_mask, 
                decoder_input_ids=None, decoder_attention_mask=None, num_ce_tokens=0, ce_labels=None):
            
        if decoder_input_ids is None:
            logprob = logprobs_from_logits(lm_logits, model_input[:, -gen_len:])
            logprob[attention_mask[:,-gen_len-1:-1] == 0] = 0 # Zero out
            loss = -logprob * rewards
            
            # Handle cross entropy loss
            ce_loss = None
            if num_ce_tokens != 0:
                ce_logits = lm_logits[:,:num_ce_tokens].clone()
                ce_loss_fct = CrossEntropyLoss()
                ce_labels = ce_labels[:,:num_ce_tokens].to(ce_logits.device)

                ce_loss = ce_loss_fct(ce_logits.view(-1, ce_logits.shape[-1]), ce_labels.view(-1))
                
                # Replace prefix loss with ce_loss
                loss[:,:num_ce_tokens] = ce_loss
            loss = torch.sum(loss) / torch.sum(attention_mask)
        else:
            logprob = logprobs_from_logits(lm_logits[:, :-1, :], decoder_input_ids[:, 1:])
            logprob[decoder_attention_mask[:,:-1] == 0] = 0 # Zero out
            loss = -logprob * rewards
            
            # Handle cross entropy loss
            ce_loss = None
            if num_ce_tokens != 0:
                ce_logits = lm_logits[:,:num_ce_tokens].clone()
                ce_loss_fct = CrossEntropyLoss()
                ce_labels = ce_labels[:,:num_ce_tokens].to(ce_logits.device)
                
                ce_loss = ce_loss_fct(ce_logits.view(-1, ce_logits.shape[-1]), ce_labels.view(-1))
                
                # Replace prefix loss with ce_loss
                loss[:,:num_ce_tokens] = ce_loss
            loss = torch.sum(loss) / torch.sum(attention_mask)
            
        return loss

    def record_step_stats(self, loss):
        stats = {
            "reinforce/loss": loss
        }
        return stats

    def update_model(self):
        self.counter += 1
        if self.counter >= self.grad_accum:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
            self.counter = 0
            
    def is_update_step(self):
        return self.counter == self.grad_accum - 1