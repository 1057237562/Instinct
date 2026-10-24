"""Differentiable SFT, KD, DPO, GRPO/CISPO and PPO objectives."""
import numpy as np


def masked_mean(e,x,mask):return (x*mask).sum()/e.binary(mask.sum(),1.,'max')


def sequence_logps(e,logits,labels):
    valid=e.binary(labels,0.,'ge')
    return (e.log_probs(logits,labels)*valid).sum(-1)


def dpo(e,policy,reference,beta=.1):
    half=policy.shape[0]//2
    if policy.shape[0]%2:raise ValueError('DPO expects chosen rows followed by rejected rows')
    margin=(policy[:half]-policy[half:])-(reference[:half]-reference[half:])
    return (-margin*beta).softplus().mean()


def distillation(e,student,teacher,labels,temperature=1.):
    teacher_log=e.softmax(teacher.detach().cast('fp32')/temperature,log=True)
    student_log=e.softmax(student.cast('fp32')/temperature,log=True)
    kl=(teacher_log.exp()*(teacher_log-student_log)).sum(-1)*temperature*temperature
    return masked_mean(e,kl,e.binary(labels,0.,'ge'))


def grpo(e,current,old,reference,advantages,mask,beta=.1,epsilon=.2,epsilon_high=5.,kind='cispo'):
    ratio=(current-old).exp();delta=reference-current;kl=delta.exp()-delta-1.
    advantage=advantages.reshape(-1,1)
    if kind=='cispo':surrogate=ratio.clip(0.,epsilon_high).detach()*advantage*current
    elif kind=='grpo':surrogate=e.binary(ratio*advantage,ratio.clip(1.-epsilon,1.+epsilon)*advantage,'min')
    else:raise ValueError('unknown GRPO objective')
    losses=(-(surrogate-beta*kl)*mask).sum(-1)/e.binary(mask.sum(-1),1.,'max')
    return losses.mean()


def ppo(e,current,old,reference,values,old_values,advantages,returns,mask,
        clip_epsilon=.2,cliprange_value=.2,kl_coef=.02,vf_coef=.5):
    ratio=(current-old).exp();delta=reference-current;kl=delta.exp()-delta-1.
    policy=masked_mean(e,e.binary(-advantages*ratio,-advantages*ratio.clip(1.-clip_epsilon,1.+clip_epsilon),'max')+kl*kl_coef,mask)
    clipped=old_values+(values-old_values).clip(-cliprange_value,cliprange_value)
    vloss=masked_mean(e,e.binary((values-returns)*(values-returns),(clipped-returns)*(clipped-returns),'max'),mask)*.5
    approx_kl=masked_mean(e,(current-old)*(current-old)*.5,mask)
    return policy+vf_coef*vloss,policy,vloss,approx_kl


def group_advantages(rewards,count):
    rewards=np.asarray(rewards,np.float32).reshape(-1,count)
    return ((rewards-rewards.mean(-1,keepdims=True))/(rewards.std(-1,keepdims=True)+1.e-4)).reshape(-1)


def gae(rewards,values,mask,gamma=1.,lam=.95):
    """Terminal rewards on the last generated token; reset over padded positions."""
    rewards,values,mask=np.asarray(rewards),np.asarray(values,np.float32),np.asarray(mask,np.float32)
    advantage=np.zeros_like(values);running=np.zeros(values.shape[0],np.float32)
    last=np.where(mask!=0,np.arange(mask.shape[1])[None,:],-1).max(-1)
    for t in range(values.shape[1]-1,-1,-1):
        valid=mask[:,t]
        next_valid=mask[:,t+1] if t+1<values.shape[1] else np.zeros_like(valid)
        nxt=values[:,t+1] if t+1<values.shape[1] else np.zeros_like(valid)
        reward=rewards*(last==t)
        delta=reward+gamma*nxt*next_valid-values[:,t]
        running=(delta+gamma*lam*running*next_valid)*valid;advantage[:,t]=running
    returns=advantage+values
    selected=advantage[mask.astype(bool)]
    if selected.size:advantage=(advantage-selected.mean())/np.sqrt(selected.var()+1.e-8)*mask
    return advantage.astype(np.float32),returns.astype(np.float32)
