"""Real BF16 momentum storage, precise split weights and FP32 step arithmetic."""
import copy
import io
from types import SimpleNamespace

import pytest
import torch

from trainer.batched_muon import BatchedMuon


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def compare_reference(device, monkeypatch, *, nesterov=True, transposed=False):
    from experiments.audit_muon_precision import BF16MomentumMuon
    torch.manual_seed(151)
    shapes = [(1664,512),(512,1664),(512,512),(128,512),(8,512)] if device == 'cuda' else [(12,8)]*3
    ref_params = [torch.nn.Parameter(torch.randn(s,device=device).bfloat16()*.02) for s in shapes]
    if transposed:
        ref_params = [torch.nn.Parameter(p.detach().T) for p in ref_params]
    params = [torch.nn.Parameter(p.detach().clone()) for p in ref_params]
    # Independent old implementation: FP32 buffers, round/demote AFTER step.
    ref = BF16MomentumMuon(ref_params, nesterov=nesterov, adjust_lr_fn='match_rms_adamw')
    opt = BatchedMuon(params, use_mantissa=True, momentum_dtype='bf16', nesterov=nesterov,
                      adjust_lr_fn='match_rms_adamw')
    pointers = {}
    for step in range(8):
        for i,(p,q) in enumerate(zip(ref_params,params)):
            p.grad = None if (i+step)%5 == 0 else torch.randn_like(p)
            q.grad = None if p.grad is None else p.grad.clone()
        for instance in (ref,opt):
            instance.param_groups[0]['lr'] = [5e-4,0.,1e-6,3e-4][step%4]
            instance.param_groups[0]['momentum'] = [.95,.5,.8,0.][step%4]
        monkeypatch.setenv('INSTINCT_MUON_FUSED_UPDATES','0')
        ref.step()
        monkeypatch.setenv('INSTINCT_MUON_FUSED_UPDATES','1')
        with torch.autocast(device, dtype=torch.bfloat16):
            opt.step()
        for p,q in zip(ref_params,params):
            assert torch.equal(p.view(torch.int16),q.view(torch.int16))
            for key,value in ref.state[p].items():
                assert torch.equal(value,opt.state[q][key])
                assert value.dtype == opt.state[q][key].dtype
            if 'momentum_buffer' in opt.state[q]:
                buf = opt.state[q]['momentum_buffer']
                assert buf.dtype == torch.bfloat16
                assert pointers.setdefault(id(q),buf.data_ptr()) == buf.data_ptr()


@pytest.mark.parametrize('nesterov',[True,False])
@pytest.mark.parametrize('transposed',[True,False])
def test_bf16_momentum_reference_cpu(monkeypatch,nesterov,transposed):
    compare_reference('cpu',monkeypatch,nesterov=nesterov,transposed=transposed)


@pytest.mark.gpu
@pytest.mark.parametrize('nesterov',[True,False])
@pytest.mark.parametrize('transposed',[True,False])
def test_bf16_momentum_reference_cuda(monkeypatch,nesterov,transposed):
    compare_reference('cuda',monkeypatch,nesterov=nesterov,transposed=transposed)


@pytest.mark.gpu
@pytest.mark.parametrize('momentum',[0.,.5,.8,.95,1.])
@pytest.mark.parametrize('nesterov',[True,False])
def test_fused_prepare_rounds_only_persistent_ema(momentum,nesterov):
    from trainer.muon_kernels import prepare_update
    torch.manual_seed(157)
    grads = [torch.randn(19,71,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    buffers = [torch.randn_like(g) for g in grads]
    precise = [b.float() for b in buffers]
    float_grads = [g.float() for g in grads]
    torch._foreach_lerp_(precise,float_grads,1-momentum)
    update = torch.stack(torch._foreach_lerp(float_grads,precise,momentum) if nesterov else precise).bfloat16()
    actual = prepare_update(grads,buffers,momentum=momentum,nesterov=nesterov)
    assert torch.equal(actual,update)
    for b,p in zip(buffers,precise):
        assert torch.equal(b.view(torch.int16),p.bfloat16().view(torch.int16))


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.gpu)])
def test_bf16_momentum_serialized_resume_and_preallocation(device):
    torch.manual_seed(163)
    p = torch.nn.Parameter(torch.randn(128,512,device=device).bfloat16())
    opt = BatchedMuon([p],use_mantissa=True,momentum_dtype='bf16')
    assert opt.reserve_momentum() == p.numel()*4
    assert opt.reserve_momentum() == 0
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()
    stream = io.BytesIO()
    torch.save({'p':p.detach().clone(),'opt':opt.state_dict()},stream)
    stream.seek(0)
    snapshot = torch.load(stream,map_location=device,weights_only=True)
    q = torch.nn.Parameter(snapshot['p'])
    restored = BatchedMuon([q],use_mantissa=True,momentum_dtype='bf16')
    restored.load_state_dict(snapshot['opt'])
    assert restored.reserve_momentum() == 0
    assert restored.state[q]['momentum_buffer'].dtype == torch.bfloat16
    assert restored.state[q]['mantissa'].dtype == torch.uint16
    for step in range(3):
        p.grad = None if step == 1 else torch.randn_like(p)
        q.grad = None if p.grad is None else p.grad.clone()
        opt.step()
        restored.step()
        assert torch.equal(p,q)
        for key,value in opt.state[p].items():
            assert torch.equal(value,restored.state[q][key])


@pytest.mark.parametrize('source,target',[('bf16','fp32'),('fp32','bf16')])
@pytest.mark.parametrize('initialized',[False,True])
def test_resume_rejects_momentum_precision_change(source,target,initialized):
    p = torch.nn.Parameter(torch.randn(8,8).bfloat16())
    opt = BatchedMuon([p],use_mantissa=True,momentum_dtype=source)
    if initialized:
        opt.reserve_momentum()
    new = BatchedMuon([p],use_mantissa=True,momentum_dtype=target)
    with pytest.raises(ValueError,match='momentum dtype changed'):
        new.load_state_dict(copy.deepcopy(opt.state_dict()))


def test_bf16_momentum_factory_and_actual_bytes(monkeypatch):
    from trainer.trainer_utils import build_optimizer, cast_training_parameters, estimate_training_persistent_bytes
    monkeypatch.setenv('INSTINCT_MUON_MANTISSA','1')
    monkeypatch.setenv('INSTINCT_MUON_MOMENTUM_DTYPE','bf16')
    monkeypatch.setenv('INSTINCT_MUON_PREALLOCATE','1')
    model = cast_training_parameters(torch.nn.Linear(8,12),'bf16')
    opt = build_optimizer(model.named_parameters(),lr=1e-6,optimizer='muon')
    assert opt.optimizers[0].momentum_dtype == torch.bfloat16
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    actual = sum(p.numel()*p.element_size()+p.grad.numel()*p.grad.element_size() for p in model.parameters())
    actual += sum(t.numel()*t.element_size() for child in opt.optimizers for s in child.state.values()
                  for k,t in s.items() if isinstance(t,torch.Tensor) and k != 'step')
    assert estimate_training_persistent_bytes(model,SimpleNamespace(optimizer='muon')) == actual
    assert actual == model.weight.numel()*8 + model.bias.numel()*16
    monkeypatch.setenv('INSTINCT_MUON_MANTISSA','0')
    with pytest.raises(ValueError,match='requires INSTINCT_MUON_MANTISSA=1'):
        build_optimizer(model.named_parameters(),lr=5e-4,optimizer='muon')
    monkeypatch.setenv('INSTINCT_MUON_MOMENTUM_DTYPE','int8')
    with pytest.raises(ValueError,match='must be fp32 or bf16'):
        build_optimizer(model.named_parameters(),lr=5e-4,optimizer='muon')


def test_corrupt_bf16_momentum_checkpoint_rejected():
    p = torch.nn.Parameter(torch.randn(8,8).bfloat16())
    opt = BatchedMuon([p],use_mantissa=True,momentum_dtype='bf16')
    opt.reserve_momentum()
    saved = copy.deepcopy(opt.state_dict())
    saved['state'][0]['momentum_buffer'] = saved['state'][0]['momentum_buffer'].float()
    with pytest.raises(ValueError,match='disagrees with its precision format'):
        opt.load_state_dict(saved)


def test_momentum_option_preserves_both_existing_defaults(monkeypatch):
    from trainer.trainer_utils import build_optimizer, cast_training_parameters
    monkeypatch.delenv('INSTINCT_MUON_MANTISSA',raising=False)
    monkeypatch.delenv('INSTINCT_MUON_MOMENTUM_DTYPE',raising=False)
    model = torch.nn.Linear(8,8)
    opt = build_optimizer(model.named_parameters(),lr=5e-4,optimizer='muon')
    assert not opt.optimizers[0].use_mantissa
    assert model.weight.dtype == torch.float32
    monkeypatch.setenv('INSTINCT_MUON_MANTISSA','1')
    model = cast_training_parameters(torch.nn.Linear(8,8),'bf16')
    opt = build_optimizer(model.named_parameters(),lr=5e-4,optimizer='muon')
    assert opt.optimizers[0].use_mantissa
    assert opt.optimizers[0].momentum_dtype == torch.float32


@pytest.mark.parametrize('workspace,batch_size,expected', [(64,16,3), (144,8,8), (272,16,16), (544,32,32)])
def test_low_precision_chunk_controls(workspace,batch_size,expected):
    p = torch.nn.Parameter(torch.empty(1664,512,dtype=torch.bfloat16,device='meta'))
    opt = BatchedMuon([p],use_mantissa=True,momentum_dtype='bf16',
                      workspace_mb=workspace,batch_size=batch_size)
    assert opt._chunk_size(p) == expected


@pytest.mark.gpu
@pytest.mark.parametrize('momentum_dtype',['bf16','fp32'])
def test_split_chunk_change_preserves_updates_and_resume(monkeypatch,momentum_dtype):
    monkeypatch.setenv('INSTINCT_MUON_FUSED_UPDATES','1')
    monkeypatch.setenv('INSTINCT_MUON_NS_BACKEND','triton_graph')
    torch.manual_seed(173)
    shapes = [(1664,512)]*19 + [(512,1664)]*19
    params = [torch.nn.Parameter(torch.randn(s,device='cuda').bfloat16()*.02) for s in shapes]
    copies = [torch.nn.Parameter(p.detach().clone()) for p in params]
    small = BatchedMuon(params,use_mantissa=True,momentum_dtype=momentum_dtype,workspace_mb=64)
    large = BatchedMuon(copies,use_mantissa=True,momentum_dtype=momentum_dtype,workspace_mb=272)
    assert small._chunk_size(params[0]) == 3
    assert large._chunk_size(copies[0]) == 16
    for step in range(5):
        if step == 2:
            large.load_state_dict(copy.deepcopy(small.state_dict()))
            assert large.workspace_bytes == 272*2**20
        for i,(p,q) in enumerate(zip(params,copies)):
            p.grad = None if (i+step)%11 == 0 else torch.randn_like(p)
            q.grad = p.grad
        for opt in (small,large):
            opt.param_groups[0]['lr'] = [5e-4,0.,1e-6,3e-4,5e-5][step]
            opt.param_groups[0]['nesterov'] = step%2 == 0
        small.step()
        large.step()
        for p,q in zip(params,copies):
            assert torch.equal(p.view(torch.int16),q.view(torch.int16))
            for key,value in small.state[p].items():
                assert torch.equal(value,large.state[q][key])
