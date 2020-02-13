import torch
import torch.nn as nn
import numpy as np
from mmdet.datasets.transforms import ImageTransform

from collections import OrderedDict
from ..apis.inference import _prepare_data
from mmdet.ops.dcn.modules.deform_conv import DeformConv
from apex.parallel import SyncBatchNorm

custom_operator_list = [DeformConv, SyncBatchNorm]


def is_custom_operator(module):
    return any([isinstance(module, custom_op) for custom_op in custom_operator_list])


def summary(model, cfg):

    def register_hook(name):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            # m_key = "%s-%i" % (class_name, module_idx + 1)
            m_key = name
            new_key = False
            if name not in summary.keys():
                summary[m_key] = OrderedDict()
                new_key = True
                summary[m_key]["input_shape"] = tuple(input[0].size())
                summary[m_key]["output_shape"] = tuple(output.size())
            else:
                if not isinstance(summary[m_key]["input_shape"], list):
                    summary[m_key]["input_shape"] = [summary[m_key]["input_shape"]]
                if not isinstance(summary[m_key]["output_shape"], list):
                    summary[m_key]["output_shape"] = [summary[m_key]["output_shape"]]
                summary[m_key]["input_shape"].append(tuple(input[0].size()))
                summary[m_key]["output_shape"].append(tuple(output.size()))

            params = 0
            flops = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            input_shape = tuple(input[0].size()[1:])
            output_shape = tuple(output.size()[1:])
            batch_size_ = output.size(0)
            if class_name == "ReLU":
                flops = torch.prod(torch.LongTensor(input_shape))
            elif class_name == 'Linear':
                flops = params
            elif class_name == "BatchNorm2d" or class_name == "SyncBatchNorm":
                flops = torch.prod(torch.LongTensor(input_shape)) * 2
                if getattr(module, "affine"):
                    flops *= 2
            elif class_name == "Conv2d" or class_name == "ConvTranspose2d" or class_name == "DeformConv":
                flops = params * torch.prod(torch.LongTensor(output_shape[1:]))
            elif class_name == "MaxPool2d":
                if isinstance(module.kernel_size, tuple):
                    kernel_ops = torch.prod(torch.LongTensor([*module.kernel_size]))
                else:
                    kernel_ops = torch.prod(torch.LongTensor([module.kernel_size ** 2]))
                flops = kernel_ops * torch.prod(torch.LongTensor(output_shape))
            elif class_name == "AdaptiveMaxPool2d":
                kernel_ops = torch.prod(torch.LongTensor(input_shape[1:])//torch.LongTensor(output_shape[1:]))
                flops = kernel_ops * torch.prod(torch.LongTensor(output_shape))
            elif class_name == "AvgPool2d":
                if isinstance(module.kernel_size, tuple):
                    kernel_ops = torch.prod(torch.LongTensor([*module.kernel_size])) + 1
                else:
                    kernel_ops = torch.prod(torch.LongTensor([module.kernel_size ** 2])) + 1
                flops = kernel_ops * torch.prod(torch.LongTensor(output_shape))
            elif class_name == "AdaptiveAvgPool2d":
                kernel_ops = torch.prod(torch.LongTensor(input_shape[1:])//torch.LongTensor(output_shape[1:])) + 1
                flops = kernel_ops * torch.prod(torch.LongTensor(output_shape))
            elif class_name == "Softmax":
                flops = torch.prod(torch.LongTensor(output_shape)) * 3
            elif class_name == "NonLocal2d":
                flops = 2 * (torch.prod(torch.LongTensor(output_shape[1:])) ** 2) * module.planes
                if module.downsample:
                    flops /= 4
            elif class_name == "ContextBlock2d":
                if module.pool == "att":
                    flops = torch.prod(torch.LongTensor(output_shape))

            summary[m_key]["nb_params"] = params
            if new_key:
                summary[m_key]["nb_flops"] = flops * batch_size_
            else:
                if not isinstance(summary[m_key]["nb_flops"], list):
                    summary[m_key]["nb_flops"] = [summary[m_key]["nb_flops"]]
                summary[m_key]["nb_flops"].append(flops * batch_size_)

        return hook
    input_size = [cfg.data.test.img_scale]
    batch_size = 1
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    img = np.random.rand(*reversed(cfg.data.test.img_scale))
    data = _prepare_data(img, img_transform, cfg, 'cuda')
    fake_test_cfg = cfg.test_cfg.copy()
    if hasattr(fake_test_cfg, 'rcnn'):
        fake_test_cfg.rcnn.score_thr = 0
    else:
        fake_test_cfg.score_thr = 0
    # import pdb
    # pdb.set_trace()
    # model = build_detector(
    #     cfg.model, train_cfg=None, test_cfg=fake_test_cfg)
    model.test_cfg = fake_test_cfg
    model = model.to('cuda')
    model.eval()

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    # model.apply(register_hook)
    for m_name, m in model.named_modules():
        # hooks.append(m.register_forward_hook(register_hook(m_name, m)))
        if (
                not isinstance(m, nn.Sequential)
                and not isinstance(m, nn.ModuleList)
                and not (m == model)
                and (hasattr(nn, m.__class__.__name__) or is_custom_operator(m))
        ):
            hooks.append(m.register_forward_hook(register_hook(m_name)))

    # make a forward pass
    with torch.no_grad():
        model(return_loss=False, rescale=True, **data)

    model.test_cfg = cfg.test_cfg
    # remove these hooks
    for h in hooks:
        h.remove()

    line_new_format = "{:<30}   {:>25}  {:>15}  {:>15}"
    line_new = line_new_format.format("Layer (type)", "Output Shape", "Param #", "FLOPS")
    line_length = len(line_new)
    s = "\n"
    s += "-" * line_length + "\n"
    s += line_new + "\n"
    s += "=" * line_length + "\n"
    total_params = 0
    total_flops = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        if isinstance(summary[layer]["output_shape"], list):
            line_new = line_new_format.format(
                layer,
                str(summary[layer]["output_shape"][0]),
                "{0:,}".format(summary[layer]["nb_params"]),
                "{0:,}".format(summary[layer]["nb_flops"][0]),
            )
            total_params += summary[layer]["nb_params"]
            total_flops += summary[layer]["nb_flops"][0]
            total_output += np.prod(summary[layer]["output_shape"][0])
            for i in range(1, len(summary[layer]["output_shape"])):
                line_new += "\n"
                line_new += line_new_format.format(
                    "",
                    str(summary[layer]["output_shape"][i]),
                    "",
                    "{0:,}".format(summary[layer]["nb_flops"][i]),
                )
                total_flops += summary[layer]["nb_flops"][i]
        else:
            line_new = line_new_format.format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
                "{0:,}".format(summary[layer]["nb_flops"]),
            )
            total_params += summary[layer]["nb_params"]
            total_flops += summary[layer]["nb_flops"]
            total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        s += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_flops_size = abs(total_flops.numpy() / (1024 ** 3.))
    total_size = total_params_size + total_output_size + total_input_size

    s += "=" * line_length + "\n"
    s += "Total params: {0:,}".format(total_params) + "\n"
    s += "Trainable params: {0:,}".format(trainable_params) + "\n"
    s += "Non-trainable params: {0:,}".format(total_params - trainable_params)  + "\n"
    s += "-" * line_length + "\n"
    s += "Input size (MB): %0.2f" % total_input_size + "\n"
    s += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    s += "Params size (MB): %0.2f" % total_params_size + "\n"
    s += "Flops size (G): %0.2f" % total_flops_size + "\n"
    s += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    s += "-" * line_length + "\n"
    s += str(model) + "\n"
    # return summary
    return s
