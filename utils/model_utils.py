from torch import nn


def update_target_networks(model, target_model, tau=0.005):
    for param, target_param in zip(model.parameters(), target_model.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def optimize(env, name, model, optimizer, loss, max_grad_norm=10, n_iterations=0):
    '''
    Makes one step of SGD optimization, clips norm with max_grad_norm and
    logs everything into tensorboard
    '''
    loss = loss.mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    # logging
    env.writer.add_scalar(name, loss.item(), n_iterations)
    env.writer.add_scalar(name + "_grad_norm", grad_norm.item(), n_iterations)