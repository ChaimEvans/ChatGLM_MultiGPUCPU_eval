import torch
from typing import Optional, Tuple


class hook_layer(torch.nn.Module):
    def __init__(self, layer, device, data_temp, tag) -> None:
        super().__init__()
        self.layer = layer.float().to(device) if device == 'cpu' else layer.to(device)
        # print(device)
        self.device = device
        self.device_index = None if device == 'cpu' else int(device.split(':')[1])
        self.data_temp = data_temp
        self.tag = tag

    def ToDevice(self, _nn, hidden_states=False):
        print(self.tag, _nn.device, '->', self.device)
        if(hidden_states):
            if(self.device == 'cpu'):
                return _nn.float().to(self.device)
            else:
                return _nn.half().to(self.device)
        else:
            return _nn.to(self.device)

    def forward(self,
                hidden_states: torch.Tensor,
                position_ids,
                attention_mask: torch.Tensor,
                layer_id,
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                output_attentions: bool = False,):
        if(layer_id == 0 or hidden_states.device.index != self.device_index):
            hidden_states = self.ToDevice(hidden_states, True)
            self.data_temp.position_ids = self.ToDevice(position_ids)
            self.data_temp.attention_mask = self.ToDevice(attention_mask)
        output = self.layer(hidden_states,
                            self.data_temp.position_ids,
                            self.data_temp.attention_mask,
                            layer_id,
                            layer_past,
                            use_cache,
                            output_attentions)
        return output


class layers_data_temp():
    def __init__(self) -> None:
        self.position_ids = None
        self.attention_mask = None


class hook_easy(torch.nn.Module):
    def __init__(self, nn, device, tag) -> None:
        super().__init__()
        self.nn = nn.float().to(device) if device == 'cpu' else nn.to(device)
        # print(device)
        self.device = device
        self.device_index = None if device == 'cpu' else int(device.split(':')[1])
        self.tag = tag

    def ToDevice(self, _nn):
        print(self.tag, _nn.device, '->', self.device)
        if(self.device == 'cpu'):
            return _nn.float().to(self.device)
        else:
            if(self.tag == 'final_layernorm'):
                return _nn.half().to(self.device)
            else:
                return _nn.to(self.device)

    def forward(self, input):
        if(input.device.index != self.device_index):
            input = self.ToDevice(input)
        output = self.nn(input)
        return output


def hook(model, embeddings, layers, final_layernorm):
    print('word_embeddings', end=' -> ')
    model.transformer.word_embeddings = hook_easy(model.transformer.word_embeddings, embeddings, 'word_embeddings')
    data_temp = layers_data_temp()# 创建layers临时数据实例
    for index, _ in enumerate(model.transformer.layers):
        print('layer', index, end=' -> ')
        model.transformer.layers[index] = hook_layer(model.transformer.layers[index], layers[index], data_temp, 'layer:' + str(index))
    print('final_layernorm', end=' -> ')
    model.transformer.final_layernorm = hook_easy(model.transformer.final_layernorm, final_layernorm, 'final_layernorm')
    print('lm_head', end=' -> ')
    model.lm_head = hook_easy(model.lm_head, embeddings, 'lm_head')
    print('hooked.')
    return model


def PickupLayersParameter(layers):
    # 处理layers参数
    if(layers is None or len(layers) < 2):
        raise 'bad layer parameter'
    check_id = set(range(1, 28 + 1))
    layers_num = 0
    for i in layers:
        layer_id = layers[i].split('-')
        layers[i] = set(range(int(layer_id[0]), int(layer_id[1]) + 1))
        if(not(layers[i] <= check_id)):
            raise 'found bad layer id.'
        layers_num += len(layers[i])
    if(layers_num != 28):
        raise 'the layer num is not 28.'
    return layers


def ConfigMultiDevices(model, embeddings:str, layers:object, final_layernorm:str):
    # 处理layers参数{'cuda:1':[1,2,3...]}
    layers = PickupLayersParameter(layers)
    # hook
    new_layers = [None for i in range(28)]  # ['cuda:1','cuda:0'...]
    for i in layers:
        for ii in layers[i]:
            new_layers[ii - 1] = i
    model = hook(model, embeddings, new_layers, final_layernorm)
    return model
