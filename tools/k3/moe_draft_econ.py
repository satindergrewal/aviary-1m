#!/usr/bin/env python3
"""MoE draft-length economics: how fast does a batch saturate the expert set?

Fable-DSpark's hypothesis for why MTP wins at np1 and loses at np4: decode cost on
a big MoE is bound by DISTINCT EXPERTS TOUCHED, and a verify batch of np*(1+n_max)
tokens touches far more of them than np tokens does.

Under uniform independent routing, B tokens each picking k of N experts touch
    D(B) = N * (1 - (1 - k/N)^B)
distinct experts in expectation. That saturates, which is the whole story: early
tokens each drag in nearly k new experts, later ones are almost free.

Expected accepted tokens per sequence for draft length n with per-position
acceptance a[] is the chain 1 + a0 + a0*a1 + ... (a rejection ends the chain).
"""

def D(B, N, k):
    return N * (1.0 - (1.0 - k / N) ** B)

def accepted(n, a):
    tot, chain = 1.0, 1.0
    for i in range(n):
        chain *= a[i]
        tot += chain
    return tot

ACC = [0.829, 0.674, 0.465]          # measured per-position acceptance, GLM-5.2 MTP

MODELS = {
    "GLM-5.2 (glm-dsa)": dict(N=256, k=8),
    "Kimi-K3 REAP80 (ours)": dict(N=179, k=16),
}

print("=" * 74)
print("EXPERT-SET SATURATION  D(B) = N*(1-(1-k/N)^B)")
print("=" * 74)
print(f"{'batch B':>8} " + "".join(f"{name:>26}" for name in MODELS))
for B in (1, 2, 4, 8, 12, 16, 24, 32, 64):
    row = f"{B:>8} "
    for name, p in MODELS.items():
        d = D(B, **p)
        row += f"{d:>18.1f} ({d/p['N']*100:4.1f}%)"
    print(row)

print()
print("=" * 74)
print("MARGINAL COST OF THE NEXT DRAFT POSITION  (verify batch = np*(1+n))")
print("=" * 74)
for name, p in MODELS.items():
    print(f"\n{name}   N={p['N']} experts, top-{p['k']}  "
          f"({p['k']/p['N']*100:.1f}% of experts per token)")
    for np_ in (1, 4):
        print(f"  np={np_}")
        base_d = D(np_, **p)
        base_t = np_ * accepted(0, ACC)
        print(f"    {'n':>2} {'batch':>6} {'D(B)':>7} {'tok/step':>9} "
              f"{'tok per expert':>15} {'vs n=0':>8}")
        base_eff = base_t / base_d
        for n in range(0, 4):
            B = np_ * (1 + n)
            d = D(B, **p)
            t = np_ * accepted(n, ACC)
            eff = t / d
            print(f"    {n:>2} {B:>6} {d:>7.1f} {t:>9.2f} {eff:>15.4f} "
                  f"{(eff/base_eff-1)*100:>+7.1f}%")

print()
print("=" * 74)
print("THE CLUE IN FABLE'S OWN NUMBERS")
print("=" * 74)
# t_step = (np * accepted) / throughput
for label, np_, n, tput in (("np1 + MTP n=3", 1, 3, 63.50),
                            ("np4, no MTP  ", 4, 0, 101.40)):
    B = np_ * (1 + n)
    t = np_ * accepted(n, ACC)
    print(f"  {label}: verify batch = {B} tokens, "
          f"step time = {t/tput*1000:.1f} ms")
print("  -> same batch size, step times within ~5%.")
print("  -> cost tracks TOTAL VERIFY-BATCH TOKENS, not slot count.")
print("     So the adaptive rule is a function of np*(1+n), one variable.")
