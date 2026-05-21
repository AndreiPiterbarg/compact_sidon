"""
Step 1 numerical verification: the PERIOD-1 reconciliation of hEq2. NO repo imports.

KEY INSIGHT: autocorr(f) and K_ms BOTH fit a length-1 period (supports in (-1/2,1/2)),
so the period-1 Parseval split is EXACT (integer lattice r in Z, prefactor 1):

  LHS2 = int autocorr(f)*K_ms = sum_{r in Z} |Ff(r)|^2 * (FK_ms(r)).re
       = 1 + sum_{r != 0} |Ff(r)|^2 * Khat1(r)            [r=0 term = 1*1 = 1]

  where Ff(r) = 𝓕f(r), Khat1(r) = (𝓕K_ms(r)).re   (INTEGER lattice, NOT j/u).

This matches the bundle's h_parseval_split CONSTANT = 1 exactly.

Then Cauchy-Schwarz on the tail:
  sum_{r!=0}|Ff(r)|^2 Khat1(r) <= sqrt(sum_{r!=0}|Ff(r)|^4) sqrt(sum_{r!=0}Khat1(r)^2)
with
  h_K_bound: sum_{r!=0}Khat1(r)^2 <= K2 - 1   (period-1 Plancherel, K_ms fits)
  h_F_bound: sum_{r!=0}|Ff(r)|^4 <= M - 1
       since sum_r|Ff(r)|^4 = sum_r|𝓕(f*f)(r)|^2 = int_{-1/2}^{1/2}(f*f)^2 = int_R(f*f)^2
             (f*f fits period 1 EXACTLY) <= ||f*f||_inf * int(f*f) = M*1 = M,
             minus r=0 term |Ff(0)|^4 = 1.
=> LHS2 <= 1 + sqrt(M-1) sqrt(K2-1).  EXACT bundle bound.
"""
import mpmath as mp
mp.mp.dps = 30

delta1=mp.mpf('0.138'); delta2=mp.mpf('0.055'); delta3=mp.mpf('0.025')
lam1=mp.mpf('0.85'); lam2=mp.mpf('0.10'); lam3=mp.mpf('0.05')
pi=mp.pi
K2 = mp.mpf('4.7889269384783')   # from verify_K2_fast.py; matches certifier [4.7888,4.7889]

def Kms_hat(xi):  # 𝓕K_ms(xi) (real, even)
    return (lam1*mp.besselj(0,pi*delta1*xi)**2 + lam2*mp.besselj(0,pi*delta2*xi)**2
            + lam3*mp.besselj(0,pi*delta3*xi)**2)

def run(name, f, w):
    print("\n===== test f:", name, " supp (-%s,%s) ====="%(w,w))
    intf = mp.quad(f,[-w,0,w]); print("int f =", mp.nstr(intf,12))
    def Ff(xi): return mp.mpf(mp.quad(lambda x: f(x)*mp.cos(2*pi*x*xi), [-w,0,w]))
    # LHS2 via R-Parseval (exact reference):
    LHS2 = 2*mp.quad(lambda xi: Ff(xi)**2*Kms_hat(xi), [0,1,3,8,20,mp.inf])
    print("LHS2 (int autocorr*K_ms) =", mp.nstr(LHS2,14))
    # M = ||f*f||_inf
    def cff(x):
        lo=max(-w,x-w); hi=min(w,x+w)
        return mp.mpf('0') if hi<=lo else mp.quad(lambda t:f(t)*f(x-t),[lo,hi])
    M = max(cff(mp.mpf(k)/1500) for k in range(0,int(2*w*1500)+1))
    print("M = ||f*f||_inf ~", mp.nstr(M,12))
    # int (f*f)^2  (f*f fits (-1/2,1/2))
    iff2 = 2*mp.quad(lambda x: cff(x)**2, [0, w, 2*w])
    print("int (f*f)^2 =", mp.nstr(iff2,12), " | <= M*1 ? ", iff2<=M, " (M-(int)=",mp.nstr(M-iff2,8),")")
    # ----- PERIOD-1 lattice sums (integer r) -----
    RMAX=200
    split_const_check = mp.mpf('0')   # = sum_r |Ff(r)|^2 Khat1(r)
    sumF4_tail = mp.mpf('0'); sumK2_tail = mp.mpf('0'); tail_inner = mp.mpf('0')
    for r in range(1, RMAX+1):
        fr = Ff(mp.mpf(r))**2          # |𝓕f(r)|^2  (real f even => Ff real)
        kr = Kms_hat(mp.mpf(r))        # (𝓕K_ms(r)).re
        tail_inner += 2*fr*kr
        sumF4_tail += 2*fr**2
        sumK2_tail += 2*kr**2
    # r=0 terms: |Ff(0)|^2=1, Khat1(0)=Kms_hat(0)=1
    print("\n[PERIOD-1 split]  1 + sum_{r!=0}|Ff(r)|^2 Khat1(r) =",
          mp.nstr(1+tail_inner,14), " | matches LHS2? diff", mp.nstr(abs(1+tail_inner-LHS2),8))
    print("  h_K_bound: sum_{r!=0}Khat1(r)^2 =", mp.nstr(sumK2_tail,12),
          " <= K2-1 =", mp.nstr(K2-1,12), "?", sumK2_tail<=K2-1, " slack", mp.nstr(K2-1-sumK2_tail,8))
    print("  (full sum_r Khat1^2 = 1+tail =", mp.nstr(1+sumK2_tail,12), " should = K2 =", mp.nstr(K2,12),
          " diff", mp.nstr(abs(1+sumK2_tail-K2),8), ")")
    print("  h_F_bound: sum_{r!=0}|Ff(r)|^4 =", mp.nstr(sumF4_tail,12),
          " <= M-1 =", mp.nstr(M-1,12), "?", sumF4_tail<=M-1, " slack", mp.nstr(M-1-sumF4_tail,8))
    print("  (full sum_r|Ff(r)|^4 = 1+tail =", mp.nstr(1+sumF4_tail,12), " should = int(f*f)^2 =",
          mp.nstr(iff2,12), " diff", mp.nstr(abs(1+sumF4_tail-iff2),8), ")")
    CS = mp.sqrt(sumF4_tail)*mp.sqrt(sumK2_tail)
    bound = 1 + mp.sqrt(M-1)*mp.sqrt(K2-1)
    print("\n  tail_inner =", mp.nstr(tail_inner,12), " <= CS =", mp.nstr(CS,12), "?", tail_inner<=CS)
    print("  CS <= sqrt(M-1)sqrt(K2-1) =", mp.nstr(mp.sqrt(M-1)*mp.sqrt(K2-1),12), "?",
          CS<=mp.sqrt(M-1)*mp.sqrt(K2-1))
    print("  ==> LHS2 =", mp.nstr(LHS2,12), " <= bundle bound 1+sqrt(M-1)sqrt(K2-1) =",
          mp.nstr(bound,12), "?", LHS2<=bound, " margin", mp.nstr(bound-LHS2,8))

w1=mp.mpf('0.2')
b1=lambda x:(1-(x/w1)**2)**2 if abs(x)<w1 else mp.mpf('0')
c1=1/mp.quad(b1,[-w1,0,w1]); run("triangle^2 bump", lambda x:c1*b1(x), w1)

w2=mp.mpf('0.24')
c2=1/(2*w2); run("box", lambda x:(c2 if abs(x)<w2 else mp.mpf('0')), w2)
