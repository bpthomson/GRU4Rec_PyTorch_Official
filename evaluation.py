import torch
import numpy as np

@torch.no_grad()
def batch_eval(gru, test_data, cutoff=[20], batch_size=512, mode='conservative', item_key='ItemId', session_key='SessionId', time_key='Time'):
    if gru.error_during_train: 
        raise Exception('Attempting to evaluate a model that wasn\'t trained properly (error_during_train=True)')
    
    recall = {c: 0 for c in cutoff}
    mrr = {c: 0 for c in cutoff}
    ndcg = {c: 0 for c in cutoff}
    
    H = []
    for i in range(len(gru.layers)):
        H.append(torch.zeros((batch_size, gru.layers[i]), requires_grad=False, device=gru.device, dtype=torch.float32))
    
    n = 0
    reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)
    
    data_iterator = gru.data_iterator.__class__(
        test_data, batch_size, 0, 0, 0, item_key, session_key, time_key, 
        device=gru.device, itemidmap=gru.data_iterator.itemidmap
    )
    
    for in_idxs, out_idxs in data_iterator(enable_neg_samples=False, reset_hook=reset_hook):
        for h in H: h.detach_()
        O = gru.model.forward(in_idxs, H, None, training=False)
        oscores = O.T
        tscores = torch.diag(oscores[out_idxs])
        
        if mode == 'standard': 
            ranks = (oscores > tscores).sum(dim=0) + 1
        elif mode == 'conservative': 
            ranks = (oscores >= tscores).sum(dim=0)
        elif mode == 'median':  
            ranks = (oscores > tscores).sum(dim=0) + 0.5*((oscores == tscores).float().sum(dim=0) - 1) + 1
        else: 
            raise NotImplementedError
        
        ranks = ranks.float()
        for c in cutoff:
            hit_mask = (ranks <= c)
            recall[c] += hit_mask.sum().item()
            mrr[c] += (hit_mask / ranks).sum().item()
            ndcg[c] += (hit_mask / torch.log2(ranks + 1)).sum().item()
            
        n += O.shape[0]
        
    for c in cutoff:
        recall[c] /= n
        mrr[c] /= n
        ndcg[c] /= n
        
    return recall, mrr, ndcg