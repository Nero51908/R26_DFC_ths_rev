#!/usr/bin/env python3
"""
test_reward.py — unit tests for the firm-capacity reward (helper_fns.dfc_reward).

Dependency-light (numpy only): does NOT import torch/gymnasium/sb3, so it runs anywhere.
Verifies the design properties that motivated the redesign:
  1. honouring the commitment beats any breach;
  2. shortfall penalty is CONVEX (deep breaches punished super-linearly);
  3. surplus (over-delivery) is nearly free vs an equal-size shortfall;
  4. the "met" band is the ABSOLUTE tolerance atol;
  5. the honour bonus scales down with spilled PV (actual_c);
  6. config.reward_params carries the expected knobs;
  7. the curtailment penalty (k_curtail) punishes spilled PV / daytime zeroing (off by default).

    python tests/test_reward.py        # exit 0 = all pass
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Ch5_DFC_Code/
import helper_fns as hlp

P = dict(atol=0.02, k_short=15.0, k_surplus=0.5)
checks = []


def ok(name, cond):
    checks.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


# 1. honour beats deep breach
r_met = hlp.dfc_reward(0.50, 0.50, 0.0, **P)
r_deep = hlp.dfc_reward(0.20, 0.50, 0.0, **P)
ok("1 honour (1.0) beats deep breach", r_met > r_deep and abs(r_met - 1.0) < 1e-9)

# 2. convex shortfall: doubling depth more-than-doubles the penalty
r_s10 = hlp.dfc_reward(0.40, 0.50, 0.0, **P)   # shortfall 0.10 -> -k*0.01
r_s20 = hlp.dfc_reward(0.30, 0.50, 0.0, **P)   # shortfall 0.20 -> -k*0.04
ok("2 shortfall penalty convex (r(0.2) < 2*r(0.1))", r_s20 < 2 * r_s10)
ok("2b matches -k_short*shortfall^2", abs(r_s20 - (-15.0 * 0.20**2)) < 1e-6)

# 3. surplus nearly free vs equal shortfall. Over-delivery has shortfall 0 -> still "met",
# so it keeps the +1 bonus minus only the light surplus nudge: 1 - k_surplus*surplus = 0.9.
r_surp = hlp.dfc_reward(0.70, 0.50, 0.0, **P)  # surplus 0.20, shortfall 0 -> met -> 1 - 0.5*0.2
r_short = hlp.dfc_reward(0.30, 0.50, 0.0, **P)  # shortfall 0.20 -> -k_short*0.04 = -0.60
ok("3 surplus much cheaper than equal shortfall", r_surp > r_short)
ok("3b surplus keeps met-bonus minus nudge (0.9)", abs(r_surp - 0.9) < 1e-9)

# 4. absolute met-band: just inside atol is met (reward ~1, minus tiny convex term),
# just outside loses the bonus and goes negative.
r_in = hlp.dfc_reward(0.50 - 0.01, 0.50, 0.0, **P)   # shortfall 0.01 <= atol -> met
r_out = hlp.dfc_reward(0.50 - 0.03, 0.50, 0.0, **P)  # shortfall 0.03 > atol -> breach
ok("4 within atol ~ met (1 - k_short*shortfall^2)",
   abs(r_in - (1 - 15.0 * 0.01**2)) < 1e-9 and r_in > 0.99)
ok("4b just outside atol is penalised", r_out < 0.0)

# 5. honour bonus scales with spill (actual_c)
ok("5 spill reduces honour bonus", hlp.dfc_reward(0.50, 0.50, 0.0, **P) >
                                    hlp.dfc_reward(0.50, 0.50, 0.2, **P))
ok("5b bonus = 1 - actual_c when met", abs(hlp.dfc_reward(0.50, 0.50, 0.2, **P) - 0.8) < 1e-9)

# 6. config wiring
try:
    import config
    ok("6 config.reward_params has atol/k_short/k_surplus/k_curtail",
       set(config.reward_params) == {"atol", "k_short", "k_surplus", "k_curtail"})
    ok("6b config env_id is the curtailment variant",
       config.sb3_config["env_id"] == "dfc_gymnasium/UtilityScalePVBESS-v0")
except Exception as e:                       # config import shouldn't need torch now
    ok(f"6 config import ({e})", False)

# 7. OLD baseline reward fn n3 (helper_fns.dfc_reward_n3), kept for the new-vs-old comparison.
#    It is SYMMETRIC (over/under-delivery penalised equally) with a relative close-tolerance —
#    the flaw the firm reward fixes. Verify the faithful port and contrast with dfc_reward.
ok("7 n3 honour == 1.0 (exact match, no spill)", abs(hlp.dfc_reward_n3(0.50, 0.50, 0.0) - 1.0) < 1e-9)
ok("7b n3 not-close breach = -|err|", abs(hlp.dfc_reward_n3(0.20, 0.50, 0.0) - (-0.30)) < 1e-9)
ok("7c n3 is SYMMETRIC: equal over/under-delivery cost the same",
   abs(hlp.dfc_reward_n3(0.30, 0.50, 0.0) - hlp.dfc_reward_n3(0.70, 0.50, 0.0)) < 1e-9)
ok("7d n3 honour beats deep breach", hlp.dfc_reward_n3(0.50, 0.50, 0.0) > hlp.dfc_reward_n3(0.20, 0.50, 0.0))
ok("7e firm reward is ASYMMETRIC where n3 is not (surplus >> shortfall only under firm)",
   (hlp.dfc_reward(0.70, 0.50, 0.0, **P) > hlp.dfc_reward(0.30, 0.50, 0.0, **P)) and
   abs(hlp.dfc_reward_n3(0.70, 0.50, 0.0) - hlp.dfc_reward_n3(0.30, 0.50, 0.0)) < 1e-9)

# 8. curtailment penalty (k_curtail): committing 0 / curtailing 100% is NO LONGER free, restoring
#    n3's anti-zeroing pressure while keeping firm's asymmetric shortfall. Off by default (k_curtail=0).
ok("8 default k_curtail=0 -> full curtail reward-neutral (firm reward unchanged)",
   abs(hlp.dfc_reward(0.0, 0.0, 1.0, **P) - 0.0) < 1e-9)
ok("8b k_curtail>0 penalises full daytime zeroing (commit 0, curtail 100%)",
   hlp.dfc_reward(0.0, 0.0, 1.0, **P, k_curtail=0.3) < 0.0 and
   abs(hlp.dfc_reward(0.0, 0.0, 1.0, **P, k_curtail=0.3) - (-0.3)) < 1e-9)
ok("8c curtailment penalty monotone (more spill -> lower reward)",
   hlp.dfc_reward(0.50, 0.50, 0.1, **P, k_curtail=0.3) >
   hlp.dfc_reward(0.50, 0.50, 0.5, **P, k_curtail=0.3))
ok("8d penalty leaves met-bonus structure intact (met, c=0.2, k=0.5 -> 0.7)",
   abs(hlp.dfc_reward(0.50, 0.50, 0.2, **P, k_curtail=0.5) - 0.7) < 1e-9)

n_fail = sum(1 for _, c in checks if not c)
print(f"\nSUMMARY: {len(checks) - n_fail} pass, {n_fail} fail")
sys.exit(1 if n_fail else 0)
