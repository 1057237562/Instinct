"""Shape-pooled activation checkpointing for the expert dispatch.

Each graph owns scratch inputs, never live model parameters or optimizer state.
Forward retains only the original inputs/weights; backward recomputes expert
activations then differentiates. Results are copied out before a different
layer can reuse scratch. This replaces, rather than nests inside, FFN replay.
"""
from types import SimpleNamespace
from collections import OrderedDict
import os
import torch
from torch.autograd.function import once_differentiable
from model.moe_dispatch import _grouped_expert_forward as _original_dispatch
from model.checkpointing import checkpoint_ffn as _original_checkpoint

_CACHE = OrderedDict()
_LIMIT = 2


def _make_experts(weights, count):
    return [SimpleNamespace(**{name:SimpleNamespace(weight=weights[p*count+e])
        for p,name in enumerate(('gate_proj','up_proj','down_proj'))}) for e in range(count)]


def _key(x,idx,w,weights):
    return (x.device,torch.cuda.current_stream(x.device).cuda_stream,tuple(x.shape),
            tuple(idx.shape),x.dtype,w.dtype,tuple((tuple(p.shape),p.dtype) for p in weights),
            torch.are_deterministic_algorithms_enabled(),
            tuple(os.environ.get(name) for name in ('INSTINCT_MOE_FUSED_ROUTING',
                'INSTINCT_MOE_FUSED_PACK','INSTINCT_MOE_FUSED_SWIGLU',
                'INSTINCT_GROUPED_MM_BACKEND','INSTINCT_MOE_SELECTIVE_CHECKPOINT')))


class _Workspace:
    def __init__(self,x,idx,w,weights,act_fn):
        dispatch = _original_dispatch.__wrapped__
        self.x=torch.empty_like(x,requires_grad=True)
        self.idx=torch.empty_like(idx)
        self.w=torch.empty_like(w,requires_grad=True)
        self.weights=[torch.empty_like(p,requires_grad=True) for p in weights]
        self.dy=torch.empty_like(x)
        experts=_make_experts(self.weights,len(weights)//3)
        self.copy_inputs(x,idx,w,weights)
        self.dy.fill_(0)
        # Existing graph-safe kernels are selected during capture. The same
        # routing/rounding rules run in reference eager and graph execution.
        def forward():
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                return dispatch(self.x,self.idx,self.w,experts,act_fn)
        def backward():
            with torch.enable_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                y=dispatch(self.x,self.idx,self.w,experts,act_fn)
            return torch.autograd.grad(y,(self.x,self.w,*self.weights),self.dy)
        stream=torch.cuda.Stream(device=x.device)
        current=torch.cuda.current_stream(x.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.output=forward()
                self.gradients=backward()
        current.wait_stream(stream)
        self.forward_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.forward_graph,stream=stream):
            self.output=forward()
        self.backward_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.backward_graph,stream=stream):
            self.gradients=backward()

    @torch.no_grad()
    def copy_inputs(self,x,idx,w,weights):
        self.x.copy_(x); self.idx.copy_(idx); self.w.copy_(w)
        torch._foreach_copy_(self.weights,list(weights))


class _CheckpointExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,idx,w,act_fn,*weights):
        key=_key(x,idx,w,weights)
        workspace=_CACHE.get(key)
        if workspace is None:
            if len(_CACHE)>=_LIMIT:
                _CACHE.popitem(last=False)
            workspace=_Workspace(x,idx,w,weights,act_fn)
            _CACHE[key]=workspace
        _CACHE.move_to_end(key)
        workspace.copy_inputs(x,idx,w,weights)
        workspace.forward_graph.replay()
        ctx.workspace=workspace
        ctx.save_for_backward(x,idx,w,*weights)
        return workspace.output.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx,dy):
        x,idx,w,*weights=ctx.saved_tensors
        workspace=ctx.workspace
        workspace.copy_inputs(x,idx,w,weights)
        workspace.dy.copy_(dy)
        workspace.backward_graph.replay()
        dx,dw,*dp=[g.clone(memory_format=torch.contiguous_format) for g in workspace.gradients]
        return dx,None,dw,None,*dp


@torch.compiler.disable
def checkpointed_experts(x,idx,w,experts,act_fn):
    weights=[getattr(e,name).weight for name in ('gate_proj','up_proj','down_proj') for e in experts]
    return _CheckpointExperts.apply(x,idx,w,act_fn,*weights)


def install(enabled):
    import model.moe_dispatch as dispatch
    import model.model_instinct as model
    def checkpoint(ffn,x,*args,**kwargs):
        if hasattr(ffn,'experts'):
            out=ffn(x)
            return out,ffn.aux_loss
        return _original_checkpoint(ffn,x,*args,**kwargs)
    dispatch._grouped_expert_forward=checkpointed_experts if enabled else _original_dispatch
    model.checkpoint_ffn=checkpoint if enabled else _original_checkpoint
