
import tkinter as tk
from tkinter import ttk, messagebox
from poker_core import parse_cards, GameState, SituationAnalyzer

def f(x):
    return f"{x:.2f}"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Poker AI Coach 0.3.2")
        self.geometry("980x760")
        self.analyzer=SituationAnalyzer()
        self.build()
    def build(self):
        frm=ttk.Frame(self,padding=12); frm.pack(fill="both",expand=True)
        fields=ttk.Frame(frm); fields.pack(fill="x")
        labels=[("Hero 手牌","As Kh"),("公共牌","Kd 8c 3s"),("底池 BB","12"),("跟注 BB","6"),("有效筹码 BB","100")]
        self.vars={}
        for i,(name,default) in enumerate(labels):
            ttk.Label(fields,text=name).grid(row=0,column=i,padx=4,pady=3)
            v=tk.StringVar(value=default); self.vars[name]=v
            ttk.Entry(fields,textvariable=v,width=16).grid(row=1,column=i,padx=4,pady=3)
        ttk.Label(fields,text="对手位置").grid(row=2,column=0,padx=4,pady=3)
        self.pos=tk.StringVar(value="BTN"); ttk.Combobox(fields,textvariable=self.pos,values=["UTG","HJ","CO","BTN","SB","BB"],state="readonly",width=13).grid(row=3,column=0,padx=4)
        ttk.Label(fields,text="对手行动").grid(row=2,column=1,padx=4,pady=3)
        self.act=tk.StringVar(value="无"); ttk.Combobox(fields,textvariable=self.act,values=["无","BET","RAISE","3BET","ALL-IN"],state="readonly",width=13).grid(row=3,column=1,padx=4)
        ttk.Button(fields,text="开始分析",command=self.analyze).grid(row=3,column=2,padx=8)
        ttk.Button(fields,text="重置",command=self.reset).grid(row=3,column=3,padx=8)
        self.out=tk.Text(frm,height=34,wrap="word"); self.out.pack(fill="both",expand=True,pady=12)
    def analyze(self):
        try:
            hero=parse_cards(self.vars["Hero 手牌"].get())
            board=parse_cards(self.vars["公共牌"].get())
            st=GameState(hero,board,float(self.vars["底池 BB"].get()),float(self.vars["跟注 BB"].get()),float(self.vars["有效筹码 BB"].get()),self.pos.get(),self.act.get())
            r=self.analyzer.analyze(st)
            a="\n".join(f"  {x.action}: {x.ev:+.2f} BB" for x in r.actions)
            assumptions="\n".join("  • "+x for x in r.assumptions)
            text=f"""【当前牌力】
{r.hand_strength}

【对手范围】
位置：{st.villain_position}
行动：{st.villain_action}
剩余估算组合：{r.range_count}

【胜率】
Hero vs Range：{r.equity*100:.2f}%

【补牌】
估算 Outs：{r.outs}
补牌牌面：{", ".join(r.outs_cards) if r.outs_cards else "无明显改善牌"}

【底池赔率】
Pot Odds：{r.pot_odds*100:.2f}%
盈亏平衡胜率：{r.breakeven_equity*100:.2f}%

【SPR】
{r.spr:.2f}

【EV】
跟注 EV：{r.call_ev:+.2f} BB

【牌面结构】
{r.texture}

【行动对比】
{a}

【模型说明】
{assumptions}
"""
            self.out.delete("1.0","end"); self.out.insert("1.0",text)
        except Exception as e:
            messagebox.showerror("输入或分析错误",str(e))
    def reset(self):
        self.vars["Hero 手牌"].set("As Kh"); self.vars["公共牌"].set("Kd 8c 3s")
        self.vars["底池 BB"].set("12"); self.vars["跟注 BB"].set("6"); self.vars["有效筹码 BB"].set("100")
        self.pos.set("BTN"); self.act.set("无"); self.out.delete("1.0","end")

if __name__=="__main__":
    App().mainloop()
