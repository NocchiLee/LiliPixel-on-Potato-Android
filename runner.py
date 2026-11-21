import json, struct, math, re, unicodedata, zlib
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent

class Weights:
    def __init__(self, name):
        self.path = BASE / name
        with self.path.open('rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.start = 8 + n

    def get(self, key, rows=None):
        h = self.header[key]
        dtype = {'F32':'<f4','BF16':'<u2','F16':'<f2'}[h['dtype']]
        a = np.memmap(self.path, dtype, 'r', self.start+h['data_offsets'][0], tuple(h['shape']))
        if rows is not None:
            a = a[rows]
        if h['dtype'] == 'BF16':
            return (a.astype(np.uint32) << 16).view(np.float32)
        return np.asarray(a, dtype=np.float32)

def tokenize(text):
    tok = json.loads((BASE/'tokenizer.json').read_text())
    model = tok['model']
    vocab = model['vocab']
    merges = {tuple(p.split() if isinstance(p,str) else p):i for i,p in enumerate(model['merges'])}
    bs = list(range(33,127))+list(range(161,173))+list(range(174,256))
    cs = bs.copy()
    for i in range(256):
        if i not in bs:
            bs.append(i)
            cs.append(256+len(cs)-188)
    table = dict(zip(bs,map(chr,cs)))
    text = unicodedata.normalize('NFC',text)
    pieces = re.findall((BASE/'split-pattern.txt').read_text('ascii'), text)
    ids = []
    for piece in pieces:
        symbols = [table[x] for x in piece.encode()]
        while len(symbols)>1:
            rank,j = min((merges.get((a,b),10**9),j) for j,(a,b) in enumerate(zip(symbols,symbols[1:])))
            if rank==10**9: break
            symbols[j:j+2] = [''.join(symbols[j:j+2])]
        ids.extend(vocab[s] for s in symbols)
    return ids[:96]

def norm(x,w):
    return x*(1/np.sqrt((x*x).mean(-1,keepdims=True)+1e-6))*w

def silu(x):
    return x/(1+np.exp(-np.clip(x,-80,80)))

def rotate(x,c,s):
    a,b=np.split(x,2,-1)
    return x*c+np.concatenate((-b,a),-1)*s

def attention(q,k,v):
    if q.shape[-2] > 512:
        # Bound peak memory at 32 and 64 pixels by processing one head at a time.
        out = np.empty_like(q)
        for b in range(q.shape[0]):
            for h in range(q.shape[1]):
                out[b,h] = attention_small(q[b,h], k[b,h], v[b,h])
        return out
    return attention_small(q,k,v)

def attention_small(q,k,v):
    p=(q@k.swapaxes(-1,-2))/np.float32(math.sqrt(q.shape[-1]))
    p=np.exp(p-p.max(-1,keepdims=True))
    return (p/p.sum(-1,keepdims=True))@v

def encode(prompt):
    w=Weights('enc.st').get
    ids=tokenize(prompt)
    x=w('model.embed_tokens.weight',ids)
    angles=np.outer(np.arange(len(ids)),1e6**(-np.arange(0,128,2)/128))
    ang=np.tile(angles,(1,2)).astype('f')
    for i in range(12):
        base=f'model.layers.{i}.'
        weight=lambda k:w(base+k+'.weight')
        h=norm(x,weight('input_layernorm'))
        def project(key,heads):
            return (h@weight('self_attn.'+key+'_proj').T).reshape(len(ids),heads,128).transpose(1,0,2)
        q,k,v=project('q',16),project('k',8),project('v',8)
        q=rotate(norm(q,weight('self_attn.q_norm')),np.cos(ang),np.sin(ang))
        k=rotate(norm(k,weight('self_attn.k_norm')),np.cos(ang),np.sin(ang))
        o=attention(q,k.repeat(2,0),v.repeat(2,0)).transpose(1,0,2).reshape(len(ids),-1)
        x=x+o@weight('self_attn.o_proj').T
        h=norm(x,weight('post_attention_layernorm'))
        x=x+(silu(h@weight('mlp.gate_proj').T)*(h@weight('mlp.up_proj').T))@weight('mlp.down_proj').T
    return norm(x,w('model.norm.weight'))@w('linear.weight').T

def generate(prompt,size,steps,seed):
    feats=encode(prompt)
    weights=Weights('dit.st')
    w=lambda k:weights.get('inner.'+k)
    p=2 if size==64 else 1
    n=size//p
    txt=np.tile(w('pad_emb'),(96,1))
    txt[:len(feats)]=norm(feats,w('txt_norm.weight'))@w('txt_proj.weight').T
    pooled=np.stack((txt[:len(feats)].mean(0),w('null_txt').mean(0)))+w('res_emb.weight')[[16,32,64].index(size)]
    txt=np.stack((txt,w('null_txt')))
    y,x=np.indices((n,n))*p+(p-1)/2
    freq=10000**(-np.arange(12)/12)
    ang=np.tile(np.c_[np.outer(np.r_[-np.arange(1,97),y.ravel()],freq),np.outer(np.r_[np.zeros(96),x.ravel()],freq)],(1,2)).astype('f')
    c,s=np.cos(ang),np.sin(ang)
    img=np.random.default_rng(seed).standard_normal((n*n,4*p*p),dtype=np.float32)
    def linear(x,key):
        return x@w(key+'.weight').T+w(key+'.bias')
    for step in range(steps):
        sigma = 1 - step * 0.999 / max(1, steps-1)
        next_sigma = 0 if step+1 == steps else 1 - (step+1)*0.999/max(1,steps-1)
        t=sigma*1000*10000**(-np.arange(384)/384)
        tm=linear(silu(linear(np.r_[np.cos(t),np.sin(t)].astype('f'),'time_mlp.0')),'time_mlp.2')
        mod=linear(silu(tm+pooled),'mod').reshape(2,6,768)
        h=np.concatenate((txt,np.broadcast_to(img@w(f'x_projs.{p}.weight').T,(2,n*n,768))),1)
        for i in range(24):
            get=lambda k:w(f'blocks.{i}.'+k)
            a,b,g,d,e,f=(mod+get('table')).transpose(1,0,2)[:,:,None,:]
            u=norm(h,get('norm1.weight'))*(1+b)+a
            q,k,v=(u@get('qkv.weight').T).reshape(2,-1,3,16,48).transpose(2,0,3,1,4)
            o=attention(rotate(q,c,s),rotate(k,c,s),v).transpose(0,2,1,3).reshape(h.shape)
            h=h+(o@get('proj_out.weight').T)*g
            u=norm(h,get('norm2.weight'))*(1+e)+d
            h=h+((silu(u@get('fc1.weight').T)*(u@get('fc3.weight').T))@get('fc2.weight').T)*f
        vc,vu=linear(norm(h[:,96:],w('norm_out.weight')),f'outs.{p}')
        img+=(next_sigma-sigma)*(vu+3*(vc-vu))
        print(f'{step+1}/{steps}',flush=True)
    pixels=img.reshape(n,n,4,p,p).transpose(0,3,1,4,2).reshape(size,size,4)
    if not np.isfinite(pixels).all(): raise ValueError('Nonfinite output')
    return np.rint((pixels.clip(-1,1)+1)*127.5).astype('u1')

def save_png(path,pixels):
    def chunk(t,b):
        return struct.pack('>I',len(b))+t+b+struct.pack('>I',zlib.crc32(t+b))
    n=len(pixels)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',n,n,8,6,0,0,0))+chunk(b'IDAT',zlib.compress(b''.join(b'\0'+row.tobytes() for row in pixels)))+chunk(b'IEND',b''))

if __name__=='__main__':
    import sys
    prompt,size,steps,seed,out=sys.argv[1:]
    save_png(Path(out),generate(prompt,int(size),int(steps),int(seed)))
