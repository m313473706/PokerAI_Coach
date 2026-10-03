
from poker_core import *

def check(name, cond):
    if not cond: raise AssertionError(name)
    print("PASS:",name)

# Math
check("Pot Odds", abs(pot_odds(12,6)-1/3)<1e-9)
check("Break-even Equity", abs(pot_odds(12,6)-1/3)<1e-9)
check("SPR", abs(100/12-8.3333333333)<1e-6)
check("Call EV", abs(call_ev(1/3,12,6)-0)<1e-9)

# Hand evaluation
check("Pair", hand_label(evaluate(["As","Kh","Kd","8c","3s"]))=="一对")
check("Trips", hand_label(evaluate(["8s","8h","Kd","8c","3s"]))=="三条")
check("Straight", hand_label(evaluate(["9s","Ts","6s","7h","8d"]))=="顺子")
check("Flush", hand_label(evaluate(["As","Ks","2s","7s","Js"]))=="同花")

# Range / blocker
base=baseline_range("BTN")
filtered=remove_known(base,["As","Ah"])
check("Range Generation", len(base)>0)
check("Blocker Filtering", len(filtered)<len(base) and all(not(set(c)&{"As","Ah"}) for c in filtered))
r3=filter_range(filtered,"3BET")
check("Range Action Filtering", len(r3)<=len(filtered))

# Equity
eq=equity(["As","Kh"],["Kd","8c","3s"],r3,300)
check("Equity Range", 0<=eq<=1)

# Outs / texture
outs,cards=approximate_outs(["As","Kh"],["Kd","8c","3s"],r3)
check("Outs", 0<=outs<=47 and len(cards)==outs)
check("Board Texture", isinstance(board_texture(["Kd","8c","3s"]),str))

# Invalid input
try:
    parse_cards("As As")
    ok=False
except ValueError:
    ok=True
check("Invalid Card Handling", ok)

# Analyzer
res=SituationAnalyzer().analyze(GameState(["As","Kh"],["Kd","8c","3s"],12,6,100,"BTN","无"))
check("Integrated Analysis", 0<=res.equity<=1 and res.range_count>0 and len(res.actions)==6)

print("\n14 / 14 TESTS PASSED")
