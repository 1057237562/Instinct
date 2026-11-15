"""Torch-free data parallel collectives: NCCL, with portable TCP fallback.

TCP carries JSON/float32 bytes only (never pickle). All ranks must execute the
same collectives. The launcher terminates the whole job if any rank fails.
"""
import ctypes as ct
import ctypes.util
import json
import math
import os
import socket
import struct
import time
import numpy as np


def _read(sock, size):
    data=bytearray(size);view=memoryview(data)
    while view:
        n=sock.recv_into(view)
        if not n:raise ConnectionError('distributed peer disconnected')
        view=view[n:]
    return data


def _send(sock, data):
    sock.sendall(struct.pack('!Q',len(data)));sock.sendall(data)


def _recv(sock, limit=1<<30):
    size=struct.unpack('!Q',_read(sock,8))[0]
    if size>limit:raise ValueError('distributed message exceeds limit')
    return _read(sock,size)


def shard_order(order, world, rank):
    """DistributedSampler-style deterministic padding to equal rank lengths."""
    order=list(order)
    if not order:return []
    total=math.ceil(len(order)/world)*world
    return (order+(order*math.ceil((total-len(order))/len(order)))[:total])[rank:total:world]


class ProcessGroup:
    def __init__(self, backend=None, transport='auto', rank=None, world=None, host=None, port=None, token=None, timeout=180.):
        self.rank=int(os.environ.get('RANK',0)) if rank is None else rank
        self.world=int(os.environ.get('WORLD_SIZE',1)) if world is None else world
        self.host=host or os.environ.get('MASTER_ADDR','127.0.0.1')
        self.port=int(port or os.environ.get('MASTER_PORT',29500))
        self.token=token if token is not None else os.environ.get('INSTINCT_JOB_TOKEN','')
        self.backend,self.transport,self.timeout=backend,transport,timeout
        self.peers={};self.listener=None;self.nccl=None;self.comm=ct.c_void_p()
        if self.world<1 or not 0<=self.rank<self.world:raise ValueError('invalid distributed rank/world')

    def __enter__(self):
        if self.world==1:return self
        try:
            if self.rank==0:
                self.listener=socket.socket();self.listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                self.listener.bind((self.host,self.port));self.listener.listen(self.world);self.listener.settimeout(self.timeout)
                while len(self.peers)<self.world-1:
                    peer,_=self.listener.accept();peer.settimeout(self.timeout)
                    hello=json.loads(_recv(peer,4096))
                    rank=hello.get('rank')
                    if hello.get('token')!=self.token or rank in self.peers or not isinstance(rank,int) or not 0<rank<self.world:
                        peer.close();raise ValueError('invalid distributed rendezvous')
                    self.peers[rank]=peer
            else:
                end=time.monotonic()+self.timeout
                while True:
                    try:
                        peer=socket.create_connection((self.host,self.port),timeout=min(5.,self.timeout));break
                    except OSError:
                        if time.monotonic()>=end:raise
                        time.sleep(.1)
                peer.settimeout(self.timeout);self.peers[0]=peer
                _send(peer,json.dumps(dict(rank=self.rank,token=self.token)).encode())
            library=None
            if self.transport!='tcp' and self.backend is not None:
                try:library=ct.CDLL(os.environ.get('INSTINCT_NCCL_LIBRARY') or ctypes.util.find_library('nccl') or 'libnccl.so.2')
                except OSError:pass
            available=self.sum(np.asarray([library is not None],np.float32))[0]
            if self.transport=='nccl' and available!=self.world:raise RuntimeError('NCCL required but unavailable on a rank')
            if available==self.world:
                self.nccl=library;self._init_nccl()
            return self
        except BaseException:
            self.__exit__(None,None,None);raise

    def broadcast(self, data=None):
        if self.world==1:return data
        if self.rank==0:
            for peer in self.peers.values():_send(peer,data)
            return data
        return bytes(_recv(self.peers[0]))

    def sum(self, values):
        values=np.ascontiguousarray(values,dtype=np.float32)
        if self.world==1:return values.copy()
        if self.rank:
            _send(self.peers[0],values.tobytes());raw=_recv(self.peers[0])
            if len(raw)!=values.nbytes:raise ValueError('collective shape mismatch')
            return np.frombuffer(raw,np.float32).reshape(values.shape).copy()
        result=values.copy()
        for rank in sorted(self.peers):
            raw=_recv(self.peers[rank])
            if len(raw)!=values.nbytes:raise ValueError('collective shape mismatch')
            result+=np.frombuffer(raw,np.float32).reshape(values.shape)
        for peer in self.peers.values():_send(peer,result.tobytes())
        return result

    def any(self, value):return bool(self.sum(np.asarray([value],np.float32))[0])

    def _init_nccl(self):
        class UniqueId(ct.Structure):_fields_=[('internal',ct.c_char*128)]
        self.nccl.ncclGetUniqueId.argtypes=[ct.POINTER(UniqueId)]
        self.nccl.ncclCommInitRank.argtypes=[ct.POINTER(ct.c_void_p),ct.c_int,UniqueId,ct.c_int]
        self.nccl.ncclAllReduce.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p]
        self.nccl.ncclCommDestroy.argtypes=[ct.c_void_p]
        uid=UniqueId()
        if self.rank==0:self._check(self.nccl.ncclGetUniqueId(ct.byref(uid)))
        data=self.broadcast(bytes(uid) if self.rank==0 else None)
        uid=UniqueId.from_buffer_copy(data)
        self._check(self.nccl.ncclCommInitRank(ct.byref(self.comm),self.world,uid,self.rank))

    @staticmethod
    def _check(status):
        if status:raise RuntimeError(f'NCCL error {status}')

    def gradients(self, optimizer, bucket_bytes=16<<20):
        if self.world==1:return
        e=optimizer.e
        present=self.sum(np.asarray([p in optimizer.present for p in optimizer.parameters],np.float32))
        parameters=[p for p,n in zip(optimizer.parameters,present) if n]
        # Missing local experts participate with zero gradients; no rank skips a collective.
        for p in parameters:
            if p not in optimizer.present:
                e.copy(optimizer.gradients[p],e.full(p.shape,0.))
        buckets=[];bucket=[];size=0
        for p in parameters:
            if bucket and size+p.size*4>bucket_bytes:buckets.append(bucket);bucket=[];size=0
            bucket.append(p);size+=p.size*4
        if bucket:buckets.append(bucket)
        with e.no_grad():
            for bucket in buckets:
                mark=e.backend.mark()
                packed=e.cat([optimizer.gradients[p].reshape(-1) for p in bucket])
                if self.nccl:
                    # ncclFloat32=7, ncclSum=0. Same stream orders producer, collective and consumers.
                    self._check(self.nccl.ncclAllReduce(packed.data.storage,packed.data.storage,packed.size,7,0,self.comm,e.backend.main_stream))
                    reduced=packed/self.world
                else:reduced=e.array(self.sum(e.numpy(packed))/self.world)
                offset=0
                for p in bucket:
                    e.copy(optimizer.gradients[p],reduced[offset:offset+p.size].reshape(*p.shape));offset+=p.size
                e.backend.release_since(mark)
        optimizer.present=set(parameters)

    def __exit__(self,*exc):
        if self.nccl and self.comm:
            if self.backend:self.backend.synchronize()
            self.nccl.ncclCommDestroy(self.comm);self.comm=ct.c_void_p()
        for peer in self.peers.values():peer.close()
        self.peers.clear()
        if self.listener:self.listener.close();self.listener=None


def restore_pending(base_path,optimizers,old_world,step):
    """Average saved rank-local partial gradients before changing world size.

    Copying that average to every new rank preserves prior microbatch work when
    the next collective averages new gradients. RNG trajectories can change.
    """
    from contextlib import ExitStack
    from pathlib import Path
    path=Path(base_path)
    with ExitStack() as stack:
        archives=[];metadata=[]
        for rank in range(old_world):
            file=path if rank==0 else path.with_name(path.stem+f'.rank{rank}'+path.suffix)
            archive=stack.enter_context(np.load(file,allow_pickle=False));meta=json.loads(str(archive['__metadata__']))
            if meta['progress']['steps']!=step:raise ValueError('elastic resume requires rank checkpoints at the same step')
            archives.append(archive);metadata.append(meta)
        for name,opt in optimizers.items():
            reference=opt.metadata()
            for meta in metadata:
                info=meta['optimizers'][name]
                for key in ('accumulated','updates','scale','kind'):
                    if info[key]!=reference[key]:raise ValueError('inconsistent saved rank optimizer '+key)
            present=set().union(*(set(meta['optimizers'][name]['present']) for meta in metadata))
            for p in opt.parameters:
                if p.name not in present:continue
                key=f'gradient/{name}/{p.name}'
                values=[np.asarray(a[key],np.float32) for a in archives]
                if any(v.shape!=p.shape for v in values):raise ValueError('elastic resume gradient shape mismatch')
                mark=opt.e.backend.mark()
                with opt.e.no_grad():opt.e.copy(opt.gradients[p],opt.e.array(sum(values)/old_world))
                opt.e.backend.release_since(mark)
            opt.present={p for p in opt.parameters if p.name in present}
