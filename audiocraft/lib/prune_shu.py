import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from .ablate import AblateGPT 
from ..modules.conditioners import ConditioningAttributes, WavCondition,JointEmbedCondition
from transformers import (
         MusicgenConfig,
         MusicgenDecoderConfig,
         T5Config,
         EncodecConfig,
       MusicgenForConditionalGeneration,
    )

text_conditions = {
        "description": "Generate a happy song",
        "genre": "rock"
    }

# 准备 wav 条目
wav_conditions = {
    "background_music": WavCondition(
        wav=torch.randn(3, 32000),
        length=torch.tensor([32000]),
        sample_rate=[32000],
        path=["background_music.wav"]
    )
}

# 准备 joint_embed 条目
joint_embed_conditions = {
    "context_embedding": JointEmbedCondition(
        wav=torch.randn(3, 32000),
        text=["happy"],
        length=torch.tensor([32000]),
        sample_rate=[32000],
        path=["context_embedding.wav"]
    )
}

# 创建 ConditioningAttributes 实例
conditions = ConditioningAttributes(
    text=text_conditions,
    wav=wav_conditions,
    joint_embed=joint_embed_conditions
)



def find_layers(module, layers=[nn.Linear], name=''):
    # 找到 prune layer
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.
    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    # use_cache = model.config.use_cache 
    # model.config.use_cache = False 

    layers = model.transformer.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    #model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    # use_cache = model.config.use_cache
    # model.config.use_cache = False

    #layers = model.transformer.layers
    layers = model.transformer.layers



    #dev = model.hf_device_map["model.embed_tokens"]
    # if "model.embed_tokens" in model.hf_device_map:
    #     device = model.hf_device_map["model.embed_tokens"]
    # dtype = next(iter(model.parameters())).dtype

    dtype = next(iter(model.named_parameters()))[1].dtype
    print("dtype: ", dtype)
    
    #inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    
    inps = torch.zeros((128, model.card, model.dim), dtype=dtype, device=device)      #昨天是二维的
    print("yes")
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            print("batch", batch[0])
            model(batch[0].to(device), [conditions])
        except ValueError:
            pass 
    layers[0] = layers[0].module

    #print("why")

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    # model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    # adjusting the sparsity of a neural network based on a certain threshold
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):    
    layers = model.transformer.layers

    model.transformer.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)
            W[W_mask] = 0

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # use_cache = model.config.use_cache 
    # model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed, seqlen=2048,tokenizer=tokenizer)
    print("good")
    #dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,tokenizer=tokenizer)

    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    

    layers = model.transformer.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
        #     dev = model.hf_device_map[f"model.layers.{i}"]
        #     inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))  #forward is layerwise, 然后每次hook 都有个add_batch(similar to input x)

        for j in range(args.nsamples):
            with torch.no_grad():
                print("attention_mask: ", attention_mask)
                print("position_ids: ", position_ids)
                print("shape: ", inps[j].shape)
                #2 * 24 * 32
                #64 24
                #outs[j] = layer(inps[j].unsqueeze(0).unsqueeze(0))
                #outs[j] = layer(inps[j].unsqueeze(0), src_mask=attention_mask, cross_attention_src=position_ids)[0]

                print('input shape: ', inps[j].shape)
                print('input type: ', position_ids)
                print("outs_j size: ", outs[j].shape)



                #print("layer size: ", layer(torch.reshape(inps[j], (1, 1,1536)), src_mask=attention_mask, cross_attention_src=position_ids)[0].shape)
                #outs[j] = layer(torch.reshape(inps[j], (1, 1,1536)), src_mask=attention_mask, cross_attention_src=position_ids)[0]
                

                print("llll", layer)
                outs[j] = layer(inps[j].unsqueeze(0),src_mask=attention_mask, cross_attention_src=position_ids)[0]
            

                #outs[j] = layer(inps[j].unsqueeze(0))
                # print("input type",inps[j].dtype)
                # print("weight type", layer.weight.dtype)
                #outs[j] = layer(inps[j].unsqueeze(0).float())[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)   # sort by rows

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0        ## set weights to zero 
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), src_mask=attention_mask, cross_attention_src=position_ids)[0]

                #outs[j] = layer(inps[j].unsqueeze(0))[0]
                #outs[j] = layer(torch.reshape(inps[j], (1, 100,1536)), src_mask=attention_mask, cross_attention_src=position_ids)[0]
                
        inps, outs = outs, inps
    # model.config.use_cache = use_cache 
    torch.cuda.empty_cache()    # transformer model usually


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()