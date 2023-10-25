import torch
import torch.nn as nn
import torchvision.models as visionmodels
from torchray.attribution.common import Probe, get_module, attach_debug_probes
# from torchray.attribution.excitation_backprop import ExcitationBackpropContext
from eb import ExcitationBackpropContext
from torchray.attribution.excitation_backprop import gradient_to_contrastive_excitation_backprop_saliency
from torchray.benchmark import get_example_data, plot_example


def get_param_importance(model: nn.Module, mb_x, mb_y,
                         feature_importance_module=None,
                         top_quantile_importance=0.8):
    assert feature_importance_module is not None
    # Set model to eval mode.
    if model.training:
        orig_is_training = True
        model.eval()
    else:
        orig_is_training = False

    mb_x_requires_grad = mb_x.requires_grad
    mb_x.requires_grad = True

    # get last layer of model
    # model should somehow make the last layer (representation layer) easily accessible.
    last_layer = model.head[2]

    # store last-layer weights to restore later
    last_layer_w = last_layer.weight.clone()

    # Clean any existing gradient
    if mb_x.grad is not None:
        orig_mb_x_grad = mb_x.grad.clone()
    else:
        orig_mb_x_grad = None
    mb_x.grad = torch.zeros_like(mb_x)
    # Store previous gradients and make them zero
    orig_grads = {}
    for name, param in model.named_parameters():
        orig_grads[name] = param.grad.clone()
        if param.grad is not None:
            param.grad.data.zero_()

    # debug_probes = attach_debug_probes(model, debug=True)

    # computing feature importances using the original model as the backward gradient
    with torch.no_grad():
        y = model(mb_x)
    backward_gradient = feature_importance_module(y, mb_y).detach()
    select = backward_gradient.clone()

    # make all last-layer weights positive
    # (If there's a last linear layer, since both positive and negative values in the encoded representation are
    # meaningful and contribute to a parameter's importance)
    last_layer.weight.data.abs_()

    if mb_y.dim() == 4:
        # compute importance only based on new unseen class(es)
        mask = torch.zeros_like(mb_y)
        for label in feature_importance_module.target_labels:
            mask = mask + (mb_y == label)
        mask = mask > 0
        mb_x_curr = mb_x[mask]

        # A feature is good if it's performing well on MOST instances of the new class, hence the mean operation
        new_batch_size = backward_gradient[mask].shape[0]
        n_views = backward_gradient.shape[1]
        # backward_gradient = backward_gradient[mask].mean(dim=(0,1), keepdims=True)
        backward_gradient = backward_gradient.mean(dim=(0, 1), keepdims=True)
        backward_gradient = backward_gradient.repeat(new_batch_size, n_views, 1, 1)
    else:
        mb_x_curr = mb_x
    # else it's peaky, and is ok as is

    # excitation backprop (or contrastive using the second-to-last linear layer)
    with ExcitationBackpropContext(enable=True, debug=False):
        y = model(mb_x_curr)
        y.backward(backward_gradient)

    # extract weight salience (which is implemented to be equal to the gradient)
    param_salience = {}
    for name, param in model.named_parameters():
        param_salience[name] = param.grad.clone()
        assert torch.isnan(param.grad).sum() == 0
        assert (param.grad < 0).sum() == 0

    # restore original gradients
    for name, param in model.named_parameters():
        if name in orig_grads:
            param.grad = orig_grads[name]
    if orig_mb_x_grad is not None:
        mb_x.grad = orig_mb_x_grad

    # restore the sign of last-layer negative weights
    last_layer.weight.data.copy_(last_layer_w.data)

    mb_x.requires_grad = mb_x_requires_grad

    # Restore model's original mode.
    if orig_is_training:
        model.train()

    return param_salience, select

    # Contrastive excitation backprop. (for reference!)
    # input_layer = get_module(model, 'features.9')
    # contrast_layer = get_module(model, 'features.30')
    # classifier_layer = get_module(model, 'classifier.6')
    #
    # input_probe = Probe(input_layer, target='output')
    # contrast_probe = Probe(contrast_layer, target='output')
    #
    # with ExcitationBackpropContext():
    #     y = model(mb_x)
    #     # z = y[0, category_id]
    #     classifier_layer.weight.data.neg_()
    #     # z.backward()
    #
    #     classifier_layer.weight.data.neg_()
    #
    #     contrast_probe.contrast = [contrast_probe.data[0].grad]
    #
    #     # y = model(x)
    #     # z = y[0, category_id]
    #     # z.backward()
    #
    # saliency = gradient_to_contrastive_excitation_backprop_saliency(input_probe.data[0])
    #
    # input_probe.remove()
    # contrast_probe.remove()



# For showing saliency maps during debug
# idx = 18
# view = 0
# img = debug_probes[''].data[0][idx, view, 0].detach().cpu().numpy()
# batch_size = img.shape[0]
# salience = debug_probes['feature_extractor.conv_layers.0'].data[0].grad[view*batch_size + idx]
# salience = salience.detach().cpu().numpy().max(axis=0)
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(img)
# axes[1].imshow(salience)
# plt.show()
