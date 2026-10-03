
from dataclasses import dataclass, field
from itertools import combinations
import random

RANKS="23456789TJQKA"
SUITS="cdhs"
DECK=[r+s for r in RANKS for s in SUITS]
RV={r:i+2 for i,r in enumerate(RANKS)}
HAND_NAMES=["高牌","一对","两对","三条","顺子","同花","葫芦","四条","同花顺"]

def parse_cards(text):
    cards=[x.strip().capitalize() for x in text.replace(","," ").split() if x.strip()]
    for c in cards:
        if len(c)!=2 or c[0] not in RANKS or c[1] not in SUITS:
            raise ValueError(f"非法牌：{c}")
    if len(set(cards)) != len(cards):
        raise ValueError("存在重复牌")
    return cards

def eval5(cards):
    ranks=sorted((RV[c[0]] for c in cards), reverse=True)
    counts={v:ranks.count(v) for v in set(ranks)}
    flush=len({c[1] for c in cards})==1
    uniq=sorted(set(ranks), reverse=True)
    if 14 in uniq: uniq.append(1)
    straight_high=None
    for i in range(len(uniq)-4):
        if uniq[i]-uniq[i+4]==4:
            straight_high=uniq[i]; break
    if flush and straight_high:
        return (8,straight_high)
    quads=sorted([v for v,n in counts.items() if n==4], reverse=True)
    if quads:
        q=quads[0]; k=max(v for v in ranks if v!=q)
        return (7,q,k)
    trips=sorted([v for v,n in counts.items() if n==3], reverse=True)
    pairs=sorted([v for v,n in counts.items() if n>=2], reverse=True)
    if trips and len(pairs)>=2:
        t=trips[0]; p=max(v for v in pairs if v!=t)
        return (6,t,p)
    if flush:
        return (5,*ranks)
    if straight_high:
        return (4,straight_high)
    if trips:
        t=trips[0]; ks=sorted([v for v in ranks if v!=t], reverse=True)[:2]
        return (3,t,*ks)
    if len(pairs)>=2:
        p1,p2=pairs[:2]; k=max(v for v in ranks if v!=p1 and v!=p2)
        return (2,p1,p2,k)
    if len(pairs)==1:
        p=pairs[0]; ks=sorted([v for v in ranks if v!=p], reverse=True)[:3]
        return (1,p,*ks)
    return (0,*ranks)

def evaluate(cards):
    if len(cards)<5: return None
    return max(eval5(list(c)) for c in combinations(cards,5))

def hand_label(score):
    if score is None: return "牌力不足（需要至少 5 张牌）"
    return HAND_NAMES[score[0]]

def combo_score(combo):
    a,b=combo
    va,vb=RV[a[0]],RV[b[0]]
    hi,lo=max(va,vb),min(va,vb)
    pair=va==vb
    suited=a[1]==b[1]
    gap=hi-lo
    score=0
    if pair: score += 70 + hi*2
    else: score += hi*2 + lo
    if suited: score += 8
    if gap<=1: score += 6
    elif gap==2: score += 3
    if hi>=14 and lo>=10: score += 10
    return score

def baseline_range(position):
    # Simplified training ranges, not GTO.
    pos=position.upper()
    thresholds={"UTG":82,"HJ":78,"CO":72,"BTN":66,"SB":70,"BB":58}
    t=thresholds.get(pos,72)
    return [c for c in combinations(DECK,2) if combo_score(c)>=t]

def remove_known(combos, known):
    ks=set(known)
    return [c for c in combos if not (set(c)&ks)]

def filter_range(combos, action):
    a=action.upper()
    if a in ("","NONE","无"): return list(combos)
    cuts={"BET":0.72,"RAISE":0.62,"3BET":0.50,"ALL-IN":0.35}
    cut=cuts.get(a,0.72)
    ranked=sorted(combos,key=combo_score,reverse=True)
    n=max(1,int(len(ranked)*cut))
    return ranked[:n]

def equity(hero, board, villain_combos, trials=1000, seed=7):
    known=set(hero+board)
    candidates=[c for c in villain_combos if not (set(c)&known)]
    if not candidates: return 0.0
    rng=random.Random(seed)
    wins=ties=total=0
    base=[c for c in DECK if c not in known]
    for _ in range(min(trials,2000)):
        v=rng.choice(candidates)
        if set(v)&known: continue
        rem=[c for c in base if c not in v]
        need=5-len(board)
        if len(rem)<need: continue
        runout=rng.sample(rem,need)
        hs=evaluate(hero+board+runout)
        vs=evaluate(list(v)+board+runout)
        if hs is None or vs is None: continue
        total+=1
        if hs>vs: wins+=1
        elif hs==vs: ties+=1
    return (wins+ties*0.5)/total if total else 0.0

def pot_odds(pot, call):
    if call<=0: return 0.0
    return call/(pot+call)

def call_ev(equity_value,pot,call):
    # Net EV relative to decision point.
    return equity_value*(pot+call)-call

def bet_ev(equity_value,pot,bet,fold_rate=0.25):
    # Training approximation: fold equity + showdown equity.
    if bet<=0: return 0.0
    return fold_rate*pot + (1-fold_rate)*(equity_value*(pot+bet)-bet)

def board_texture(board):
    if len(board)<3: return "信息不足"
    ranks=[RV[c[0]] for c in board]
    suits=[c[1] for c in board]
    paired=len(ranks)!=len(set(ranks))
    suit_counts={s:suits.count(s) for s in set(suits)}
    mono=max(suit_counts.values(),default=0)>=3
    two_tone=max(suit_counts.values(),default=0)==2
    sr=sorted(set(ranks))
    connected=any(sr[i+2]-sr[i]<=2 for i in range(max(0,len(sr)-2)))
    parts=[]
    if paired: parts.append("配对牌面")
    if mono: parts.append("三同花倾向")
    elif two_tone: parts.append("两同花倾向")
    if connected: parts.append("顺子连接度较高")
    return "、".join(parts) if parts else "相对干燥"

def approximate_outs(hero, board, villain_combos, seed=11):
    known=set(hero+board)
    remain=[c for c in DECK if c not in known]
    current=equity(hero,board,villain_combos,400,seed)
    outs=[]
    for card in remain:
        e=equity(hero,board+[card],villain_combos,160,seed+hash(card)%1000)
        if e>current+0.10:
            outs.append(card)
    return len(outs), outs

@dataclass
class GameState:
    hero: list
    board: list
    pot: float
    call: float
    effective_stack: float
    villain_position: str="BTN"
    villain_action: str="无"

@dataclass
class ActionComparison:
    action: str
    ev: float

@dataclass
class AnalysisResult:
    hand_strength: str
    range_count: int
    equity: float
    outs: int
    outs_cards: list
    pot_odds: float
    breakeven_equity: float
    spr: float
    call_ev: float
    actions: list
    texture: str
    assumptions: list

class SituationAnalyzer:
    def analyze(self,state:GameState):
        if len(state.hero)!=2: raise ValueError("Hero 必须有 2 张手牌")
        if len(state.board)>5: raise ValueError("公共牌不能超过 5 张")
        known=state.hero+state.board
        if len(set(known))!=len(known): raise ValueError("Hero 与公共牌存在重复牌")
        if len(state.board)<3: raise ValueError("至少需要 Flop 才能进行局面分析")
        score=evaluate(state.hero+state.board)
        ranges=remove_known(baseline_range(state.villain_position),known)
        ranges=filter_range(ranges,state.villain_action)
        eq=equity(state.hero,state.board,ranges,1200)
        po=pot_odds(state.pot,state.call)
        spr=state.effective_stack/state.pot if state.pot else float("inf")
        cev=call_ev(eq,state.pot,state.call)
        outs, cards=approximate_outs(state.hero,state.board,ranges)
        actions=[
            ActionComparison("过牌",0.0),
            ActionComparison("跟注",cev),
            ActionComparison("下注 33%",bet_ev(eq,state.pot,state.pot*0.33)),
            ActionComparison("下注 50%",bet_ev(eq,state.pot,state.pot*0.50)),
            ActionComparison("下注 75%",bet_ev(eq,state.pot,state.pot*0.75)),
            ActionComparison("下注 100%",bet_ev(eq,state.pot,state.pot)),
        ]
        assumptions=[
            "对手范围为简化训练范围，不等同于 GTO Solver。",
            "胜率采用 Monte Carlo 估算，会存在随机误差。",
            "Outs 当前为估算 Outs，尚未区分 Clean / Dirty Outs。",
            "下注 EV 使用简化 Fold Equity 假设，仅用于训练比较。",
        ]
        return AnalysisResult(hand_label(score),len(ranges),eq,outs,cards,po,po,spr,cev,actions,board_texture(state.board),assumptions)
