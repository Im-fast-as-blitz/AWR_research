import torch
import torch.nn as nn


def compute_critic_target_sac(actor, target_critic1, target_critic2, rewards, next_states, is_done, alpha, gamma):
    '''
    Important: use target networks for this method! Do not use "fresh" models except fresh policy in SAC!
    input:
        rewards - PyTorch tensor, (batch_size)
        next_states - PyTorch tensor, (batch_size x features)
        is_done - PyTorch tensor, (batch_size)
    output:
        critic target - PyTorch tensor, (batch_size)
    '''
    with torch.no_grad():
        next_actions, next_log_prob = actor.apply(next_states)
        critic_1_res = target_critic1.get_qvalues(next_states, next_actions)
        critic_2_res = target_critic2.get_qvalues(next_states, next_actions)
        v_ = torch.minimum(critic_1_res, critic_2_res) - alpha * next_log_prob
        critic_target = rewards + gamma * v_ * (1 - is_done)

    assert not critic_target.requires_grad, "target must not require grad."
    assert len(critic_target.shape) == 1, "dangerous extra dimension in target?"

    return critic_target


def compute_actor_loss_sac(actor, critic1, critic2, states, alpha):
    '''
    Returns actor loss on batch of states
    input:
        states - PyTorch tensor, (batch_size x features)
    output:
        actor loss - PyTorch tensor, (batch_size)
    '''
    # make sure you have gradients w.r.t. actor parameters
    actions, log_probs = actor.apply(states)
    q_values_1 = critic1.get_qvalues(states, actions)
    q_values_2 = critic2.get_qvalues(states, actions)

    assert actions.requires_grad, "actions must be differentiable with respect to policy parameters"

    # compute actor loss
    actor_loss = -(torch.minimum(q_values_1, q_values_2) - alpha * log_probs).mean()
    return actor_loss

class SAC_loss(nn.Module):
    def __init__(self, alpha, *args, **kwargs):
        super().__init__()

        self.mse = nn.MSELoss()
        self.alpha = alpha

    def __call__(self, q_values_1, q_values_2, rewards, next_states, is_done):
        critic_target = compute_critic_target_sac(rewards, next_states, is_done, self.alpha)
        critic1_loss = self.mse(critic_target, q_values_1)
        critic2_loss = self.mse(critic_target, q_values_2)
        return critic1_loss, critic2_loss