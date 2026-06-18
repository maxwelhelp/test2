#!/usr/bin/env python3
"""v4.5 Central Projection Council v2.

Not report-only: builds a small projected program state Z[T,L,d], simulates
several whole-program alternatives with tiny matrices, scores current->target
context movement, and writes runtime feedback targets/biases for controllers.

Still safe:
- no model weight mutation
- no auto-deploy
- no actor/critic
- output is detached feedback for future aux-loss / tiny bias
"""
from __future__ import annotations
import argparse, glob, json, math, os
from pathlib import Path
from typing import Any, Dict, List, Tuple

LANES = ["detail","state","abstract","memory"]
PRIMS = ["channel","block","low_rank","ctx_matrix","product_gate","diff","gated_contrast","memory_keep"]
TRANS = ["identity","detail_to_state","state_to_abstract","state_to_memory","memory_to_state","fork_detail_to_state_memory","join_detail_state_to_abstract","head_prepare"]
FANOUT = ["one","two","three","all_soft"]
EDGE = {"detail":0,"state":1,"abstract":2,"memory":3}


def read_json(p: Path, default=None):
    try:
        return json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        return {} if default is None else default

def write_json(p: Path, x: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(x, f, indent=2, ensure_ascii=False, sort_keys=True); f.write("\n")

def fnum(x, d=0.0):
    try: return float(x)
    except Exception: return d

def latest_report(root: Path) -> Path:
    xs = sorted((root/"simple_butterfly_matrix_v4_tape_lane/agent_reports").glob("v4_4_context_controllers_*"), key=lambda p:p.stat().st_mtime)
    if not xs: raise FileNotFoundError("no v4_4_context_controllers_* reports")
    return xs[-1]

def latest_trace(rd: Path) -> Tuple[int, Path]:
    xs = sorted(rd.glob("trace_feedback_epoch_*.json"))
    if not xs: raise FileNotFoundError(f"no trace_feedback in {rd}")
    def ep(p):
        try: return int(p.stem.split("_")[-1])
        except Exception: return -1
    p=max(xs,key=ep); return ep(p),p

def get(d:Dict[str,Any], *ks, default=None):
    c=d
    for k in ks:
        if isinstance(c,dict) and k in c: c=c[k]
        else: return default
    return c

def series(x,T):
    x=x if isinstance(x,list) else []
    y=[fnum(v) for v in x[:T]]
    return y+[0.0]*(T-len(y))

def topk(xs,k,gap=1):
    order=sorted(range(len(xs)),key=lambda i:xs[i],reverse=True); out=[]
    for i in order:
        if len(out)>=k: break
        if all(abs(i-j)>=gap for j in out): out.append(i)
    return sorted(out)

def softmax(xs,temp=1.0):
    if not xs: return []
    m=max(xs); es=[math.exp((x-m)/max(temp,1e-6)) for x in xs]; s=sum(es) or 1
    return [e/s for e in es]

# Tiny deterministic matrix ops over Python lists, d <= 32.
def mat_vec(M,v): return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]
def add(a,b): return [x+y for x,y in zip(a,b)]
def scale(a,s): return [x*s for x in a]
def norm(a): return math.sqrt(sum(x*x for x in a)/max(1,len(a)))
def tanh_vec(a): return [math.tanh(x) for x in a]

def eye(d,s=1.0): return [[s if i==j else 0.0 for j in range(d)] for i in range(d)]
def shift(d,k=1,s=0.7): return [[s if j==(i-k)%d else 0.0 for j in range(d)] for i in range(d)]
def diag(d,phase=0.0,s=0.35): return [[(1+s*math.sin(i*0.7+phase)) if i==j else 0.0 for j in range(d)] for i in range(d)]

def primitive_mats(d):
    return {
        "channel": eye(d,1.02),
        "block": shift(d,1,0.45),
        "low_rank": [[0.04*math.sin((i+1)*(j+1)) for j in range(d)] for i in range(d)],
        "ctx_matrix": diag(d,0.5,0.25),
        "product_gate": diag(d,1.3,0.35),
        "diff": [[(1.0 if i==j else (-0.55 if j==(i-1)%d else 0.0)) for j in range(d)] for i in range(d)],
        "gated_contrast": diag(d,2.0,0.45),
        "memory_keep": eye(d,0.88),
    }

def route_basis(name:str)->List[List[float]]:
    L=4; R=[[1.0 if i==j else 0.0 for j in range(L)] for i in range(L)]
    def boost(a,b,w): R[a][b]+=w
    if name=="detail_to_state": boost(0,1,1.2)
    elif name=="state_to_abstract": boost(1,2,1.2)
    elif name=="state_to_memory": boost(1,3,1.1)
    elif name=="memory_to_state": boost(3,1,1.1)
    elif name=="fork_detail_to_state_memory": boost(0,1,0.9); boost(0,3,0.9)
    elif name=="join_detail_state_to_abstract": boost(0,2,0.7); boost(1,2,0.9)
    elif name=="head_prepare":
        for i in range(L): boost(i,2,0.35); boost(i,3,0.2)
    # row normalize
    out=[]
    for row in R:
        s=sum(row) or 1; out.append([x/s for x in row])
    return out

def route_apply(Z, R):
    L=len(Z); d=len(Z[0]); Y=[[0.0]*d for _ in range(L)]
    for fr in range(L):
        for to in range(L):
            w=R[fr][to]
            for j in range(d): Y[to][j]+=w*Z[fr][j]
    return Y

def build_context(trace:Dict[str,Any])->Dict[str,Any]:
    b = get(trace,"boundary_by_step", default=get(trace,"boundary","boundary_by_step", default=[]))
    seq = get(trace,"sequence_change_by_step", default=get(trace,"sequence","sequence_change_by_step", default=[]))
    prim = get(trace,"primitive_by_step", default=get(trace,"primitive_weights_by_step", default=[]))
    route = get(trace,"route_matrix_by_step", default=get(trace,"route_by_step", default=[]))
    T=max(12,len(b) if isinstance(b,list) else 0,len(seq) if isinstance(seq,list) else 0,len(prim) if isinstance(prim,list) else 0,len(route) if isinstance(route,list) else 0)
    b=series(b,T); seq=series(seq,T)
    return {"T":T,"boundary":b,"seq":seq,"primitive_by_step":prim if isinstance(prim,list) else [],"route_by_step":route if isinstance(route,list) else [],"seq_peaks":topk(seq,min(4,T),2),"boundary_mean":sum(b)/T,"boundary_std":(sum((x-sum(b)/T)**2 for x in b)/T)**0.5}

def initial_Z(ctx,d=24):
    T=ctx["T"]; Z=[]
    for t in range(T):
        step=[]
        for l in range(4):
            v=[]
            for j in range(d):
                val=0.15*math.sin((t+1)*(j+1)*0.13)+0.12*math.cos((l+1)*(j+1)*0.19)
                val+=0.9*ctx["seq"][t]*(1 if l in (0,1) else 0.5)
                val+=0.25*ctx["boundary"][t]*(1 if l in (2,3) else 0.3)
                v.append(val)
            step.append(v)
        Z.append(step)
    return Z

def make_alts(ctx):
    T=ctx["T"]; ps=ctx["seq_peaks"] or [max(1,T//4),max(2,T//2),max(3,3*T//4)]
    while len(ps)<3: ps.append(min(T-1, ps[-1]+2))
    a,b,c=ps[0],ps[1],ps[-1]
    return [
        {"id":"global_fork_memory_recall","boundary":[a,b,c],"trans":{a:"fork_detail_to_state_memory",b:"state_to_abstract",c:"memory_to_state"},"fanout":{a:"two",b:"one",c:"one"},"prim":{a:{"detail":"memory_keep"},b:{"state":"ctx_matrix"},c:{"memory":"ctx_matrix"}},"target":"early fork + mid abstract + late memory recall"},
        {"id":"abstract_head_prepare","boundary":[b,c],"trans":{b:"state_to_abstract",c:"head_prepare"},"fanout":{b:"one",c:"two"},"prim":{b:{"state":"low_rank"},c:{"abstract":"product_gate"}},"target":"make abstract lane useful for head"},
        {"id":"route_specialize_no_boundary","boundary":[],"trans":{a:"detail_to_state",b:"state_to_abstract",c:"memory_to_state"},"fanout":{a:"one",b:"one",c:"one"},"prim":{a:{"detail":"diff"}},"target":"sharpen allowed routes without new stage boundary"},
        {"id":"dormant_primitive_wakeup","boundary":[a],"trans":{a:"detail_to_state"},"fanout":{a:"two"},"prim":{a:{"detail":"diff","state":"block"}},"target":"wake weak primitives in high-change region"},
        {"id":"memory_write_read_pair","boundary":[a,c],"trans":{a:"state_to_memory",c:"memory_to_state"},"fanout":{a:"one",c:"one"},"prim":{a:{"state":"memory_keep"},c:{"memory":"ctx_matrix"}},"target":"make memory write causal by adding later recall"},
    ]

def simulate(ctx, alt, d=24):
    Z=initial_Z(ctx,d); mats=primitive_mats(d)
    before=[x[:] for x in Z[-1]]
    boundary=set(alt.get("boundary",[])); trans=alt.get("trans",{}); prim=alt.get("prim",{})
    for t in range(ctx["T"]):
        step=[v[:] for v in Z[t]]
        # primitive edits at target context places
        for lane, p in prim.get(t,{}).items():
            li=EDGE.get(lane,0); M=mats.get(p, mats["channel"])
            step[li]=tanh_vec(mat_vec(M,step[li]))
        R=route_basis(trans.get(t,"identity"))
        if t not in boundary:
            # still allow transition but weaker if no boundary: this models context desire without hard deploy.
            I=route_basis("identity"); R=[[0.65*I[i][j]+0.35*R[i][j] for j in range(4)] for i in range(4)]
        step=route_apply(step,R)
        if t+1<ctx["T"]:
            Z[t+1]=[[0.75*Z[t+1][l][j]+0.25*step[l][j] for j in range(d)] for l in range(4)]
        Z[t]=step
    after=Z[-1]
    move=sum(norm([after[l][j]-before[l][j] for j in range(d)]) for l in range(4))/4
    # target context score: boundaries should sit on sequence-change peaks, useful transitions should move state/memory/abstract.
    seq_match=sum(ctx["seq"][t] for t in boundary if 0<=t<ctx["T"])/max(1,len(boundary)) if boundary else 0
    useful=sum(1 for x in trans.values() if x!="identity")/max(1,len(trans))
    complexity=0.015*(len(boundary)+len(trans)+sum(len(v) for v in prim.values()))
    pqs=0.35*min(1,20*seq_match)+0.25*useful+0.25*min(1,move)-complexity
    gain=0.04*pqs
    return round(gain,6), round(pqs,6), round(move,6)

def compile_feedback(ctx, scored, max_bias=0.03):
    T=ctx["T"]; fb={"boundary_target":[0.0]*T,"transition_target":{},"fanout_target":{},"route_bias":{},"primitive_bias":{},"max_bias":max_bias,"deploy":False}
    for rank,a in enumerate(scored[:2]):
        w=1/(rank+1)
        for t in a["boundary"]:
            if 0<=t<T: fb["boundary_target"][t]=max(fb["boundary_target"][t], round(w,4))
        for t,k in a["trans"].items(): fb["transition_target"].setdefault(str(t),{})[k]=max(fb["transition_target"].get(str(t),{}).get(k,0),round(w,4))
        for t,k in a["fanout"].items(): fb["fanout_target"].setdefault(str(t),{})[k]=max(fb["fanout_target"].get(str(t),{}).get(k,0),round(w,4))
        for t,mp in a["prim"].items():
            fb["primitive_bias"].setdefault(str(t),{})
            for lane,p in mp.items(): fb["primitive_bias"][str(t)].setdefault(lane,{})[p]=round(max_bias*w,4)
        for t,k in a["trans"].items():
            rb=fb["route_bias"].setdefault(str(t),{})
            if k=="detail_to_state": rb["detail->state"]=round(max_bias*w,4)
            if k=="state_to_abstract": rb["state->abstract"]=round(max_bias*w,4)
            if k=="state_to_memory": rb["state->memory"]=round(max_bias*w,4)
            if k=="memory_to_state": rb["memory->state"]=round(max_bias*w,4)
            if k=="fork_detail_to_state_memory": rb["detail->state"]=round(max_bias*w,4); rb["detail->memory"]=round(max_bias*w,4)
    return fb

def run(args):
    root=Path(args.repo_root).resolve(); rd=Path(args.report_dir).resolve() if args.report_dir else latest_report(root)
    ep,tp=latest_trace(rd); trace=read_json(tp,{})
    ctx=build_context(trace); ctx["epoch"]=ep
    alts=make_alts(ctx); scored=[]
    for a in alts:
        gain,pqs,move=simulate(ctx,a,args.dim); a.update({"predicted_gain":gain,"program_quality_score":pqs,"projected_move":move,"deploy":False,"target_context":a.pop("target")}); scored.append(a)
    scored=sorted(scored,key=lambda x:(x["predicted_gain"],x["program_quality_score"]),reverse=True)
    fb=compile_feedback(ctx,scored,args.max_bias)
    out={"version":"v4.5_central_projection_council_v2","mode":"projected_matrix_simulation_plus_controller_feedback","report_dir":str(rd),"epoch":ep,"small_projection":{"T":ctx["T"],"lanes":LANES,"dim":args.dim,"state":"Z[T,L,d] deterministic projection from trace context"},"current_context":ctx,"alternatives":scored,"controller_feedback":fb,"notes":["This is not just a report: it compiles projected simulations into controller-ready feedback.","Runtime must still load feedback with tiny aux-loss or detached bias; no auto-deploy here."]}
    op=rd/f"projection_council_v2_epoch_{ep:03d}.json"; write_json(op,out)
    fp=rd/f"scout_runtime_feedback_epoch_{ep:03d}.json"; write_json(fp,{"version":"v4.5_scout_runtime_feedback_v1","source":str(op),"epoch":ep,"feedback":fb,"deploy":False})
    print(f"[council_v2] report={rd}")
    print(f"[council_v2] wrote {op}")
    print(f"[council_v2] wrote {fp}")
    for a in scored[:5]: print(f"  {a['id']}: gain={a['predicted_gain']} pqs={a['program_quality_score']} move={a['projected_move']} target={a['target_context']}")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--repo-root",default="."); p.add_argument("--report-dir",default=None); p.add_argument("--dim",type=int,default=24); p.add_argument("--max-bias",type=float,default=0.03)
    run(p.parse_args())
if __name__=="__main__": main()
