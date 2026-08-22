import math, json

# ---------- conversions ----------
def srgb_to_lin(c):
    return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
def lin_to_srgb(c):
    return 12.92*c if c <= 0.0031308 else 1.055*(c**(1/2.4))-0.055

M1 = [[0.4122214708,0.5363325363,0.0514459929],
      [0.2119034982,0.6806995451,0.1073969566],
      [0.0883024619,0.2817188376,0.6299787005]]
M2 = [[0.2104542553, 0.7936177850,-0.0040720468],
      [1.9779984951,-2.4285922050, 0.4505937099],
      [0.0259040371, 0.7827717662,-0.8086757660]]
M1i= [[ 4.0767416621,-3.3077115913, 0.2309699292],
      [-1.2684380046, 2.6097574011,-0.3413193965],
      [-0.0041960863,-0.7034186147, 1.7076147010]]
M2i= [[1, 0.3963377774, 0.2158037573],
      [1,-0.1055613458,-0.0638541728],
      [1,-0.0894841775,-1.2914855480]]
def mv(M,v): return [sum(M[i][j]*v[j] for j in range(3)) for i in range(3)]

def oklch_to_srgb(L,C,H):
    a = C*math.cos(math.radians(H)); b = C*math.sin(math.radians(H))
    lms = mv(M2i,[L,a,b]); lms = [x**3 for x in lms]
    rgb = mv(M1i,lms)
    return [lin_to_srgb(x) for x in rgb]
def in_gamut(rgb): return all(-1e-4 <= c <= 1+1e-4 for c in rgb)
def clamp(rgb): return [min(1,max(0,c)) for c in rgb]
def hexof(L,C,H):
    r,g,b = clamp(oklch_to_srgb(L,C,H))
    return '#%02x%02x%02x' % tuple(round(x*255) for x in (r,g,b))
def hex_to_rgb(h):
    h=h.lstrip('#'); return [int(h[i:i+2],16)/255 for i in (0,2,4)]
def rel_lum(rgb): 
    r,g,b=[srgb_to_lin(c) for c in rgb]; return 0.2126*r+0.7152*g+0.0722*b
def contrast(h1,h2):
    a,b = rel_lum(hex_to_rgb(h1)), rel_lum(hex_to_rgb(h2))
    hi,lo = max(a,b), min(a,b)
    return (hi+0.05)/(lo+0.05)
def srgb_to_oklab(rgb):
    lin=[srgb_to_lin(c) for c in rgb]; lms=mv(M1,lin)
    lms=[math.copysign(abs(x)**(1/3),x) for x in lms]
    return mv(M2,lms)
def hex_to_lab(h):
    L,a,b = srgb_to_oklab(hex_to_rgb(h))
    # oklab -> approximate CIELab for dE2000 via XYZ path is overkill; use CIELAB proper
    return None

# proper CIELab for dE2000
def hex_to_cielab(h):
    r,g,b=[srgb_to_lin(c) for c in hex_to_rgb(h)]
    X = 0.4124564*r+0.3575761*g+0.1804375*b
    Y = 0.2126729*r+0.7151522*g+0.0721750*b
    Z = 0.0193339*r+0.1191920*g+0.9503041*b
    Xn,Yn,Zn = 0.95047,1.0,1.08883
    def f(t): return t**(1/3) if t>216/24389 else (841/108)*t+4/29
    fx,fy,fz=f(X/Xn),f(Y/Yn),f(Z/Zn)
    return (116*fy-16, 500*(fx-fy), 200*(fy-fz))

def de2000(h1,h2):
    L1,a1,b1=hex_to_cielab(h1); L2,a2,b2=hex_to_cielab(h2)
    C1=math.hypot(a1,b1); C2=math.hypot(a2,b2); Cb=(C1+C2)/2
    G=0.5*(1-math.sqrt(Cb**7/(Cb**7+25**7))) if Cb>0 else 0
    a1p,a2p=(1+G)*a1,(1+G)*a2
    C1p,C2p=math.hypot(a1p,b1),math.hypot(a2p,b2)
    h1p=math.degrees(math.atan2(b1,a1p))%360 if (a1p or b1) else 0
    h2p=math.degrees(math.atan2(b2,a2p))%360 if (a2p or b2) else 0
    dLp=L2-L1; dCp=C2p-C1p
    if C1p*C2p==0: dhp=0
    elif abs(h2p-h1p)<=180: dhp=h2p-h1p
    elif h2p-h1p>180: dhp=h2p-h1p-360
    else: dhp=h2p-h1p+360
    dHp=2*math.sqrt(C1p*C2p)*math.sin(math.radians(dhp)/2)
    Lbp=(L1+L2)/2; Cbp=(C1p+C2p)/2
    if C1p*C2p==0: hbp=h1p+h2p
    elif abs(h1p-h2p)<=180: hbp=(h1p+h2p)/2
    elif h1p+h2p<360: hbp=(h1p+h2p+360)/2
    else: hbp=(h1p+h2p-360)/2
    T=1-0.17*math.cos(math.radians(hbp-30))+0.24*math.cos(math.radians(2*hbp))+\
      0.32*math.cos(math.radians(3*hbp+6))-0.20*math.cos(math.radians(4*hbp-63))
    dth=30*math.exp(-((hbp-275)/25)**2)
    Rc=2*math.sqrt(Cbp**7/(Cbp**7+25**7)) if Cbp>0 else 0
    Sl=1+(0.015*(Lbp-50)**2)/math.sqrt(20+(Lbp-50)**2)
    Sc=1+0.045*Cbp; Sh=1+0.015*Cbp*T
    Rt=-math.sin(math.radians(2*dth))*Rc
    return math.sqrt((dLp/Sl)**2+(dCp/Sc)**2+(dHp/Sh)**2+Rt*(dCp/Sc)*(dHp/Sh))

def deuter(h):
    r,g,b=[srgb_to_lin(c) for c in hex_to_rgb(h)]
    L =  0.31399022*r+0.63951294*g+0.04649755*b
    M =  0.15537241*r+0.75789446*g+0.08670142*b
    S =  0.01775239*r+0.10944209*g+0.87256922*b
    Md = 0.9513092*L + 0.04866992*S
    r2 =  5.47221206*L -4.6419601*Md +0.16963708*S
    g2 = -1.1252419*L +2.29317094*Md -0.1678952*S
    b2 =  0.02980165*L -0.19318073*Md +1.16364789*S
    return '#%02x%02x%02x'%tuple(round(min(1,max(0,lin_to_srgb(x)))*255) for x in (r2,g2,b2))

# ---------- themes ----------

def fit_gamut(L,C,H):
    """reduce chroma until in gamut"""
    c=C
    while c>0.0005 and not in_gamut(oklch_to_srgb(L,c,H)):
        c-=0.002
    return max(c,0.0)

def hexg(L,C,H):
    return hexof(L,fit_gamut(L,C,H),H)

def nudge(L0,C,H,bgs,target,mode,step=0.004,lim=60):
    """keep the designed lightness; move only if the target is missed"""
    L=L0
    for _ in range(lim):
        h=hexg(L,C,H)
        if min(contrast(h,b) for b in bgs)>=target:
            return round(L,3),h,fit_gamut(L,C,H)
        L = L+step if mode=="dark" else L-step
        if L>0.995 or L<0.06: break
    h=hexg(L,C,H)
    return round(L,3),h,fit_gamut(L,C,H)

THEMES = [
 dict(id="graphite", nom="Graphite & Ambre", mode="dark", ordre=1,
      cv=(0.145,0.008,60), sf=(0.192,0.009,60), rs=(0.235,0.010,60),
      tx=(0.960,0.004,80), mu=(0.705,0.013,70), bd=(0.315,0.012,60), bs=(0.430,0.014,60),
      ac=(0.790,0.150,78), a2=(0.700,0.150,62),
      pitch="Chaleur minérale. Ambre sur graphite — la direction qui vous distingue le plus : vos concurrents sont tous bleus."),
 dict(id="ivoire", nom="Ivoire & Encre", mode="light", ordre=2,
      cv=(0.972,0.006,80), sf=(1.000,0.000,80), rs=(0.988,0.004,80),
      tx=(0.235,0.012,60), mu=(0.520,0.012,70), bd=(0.895,0.008,75), bs=(0.780,0.010,75),
      ac=(0.480,0.105,175), a2=(0.420,0.110,168),
      pitch="Clair, sobre, imprimable. Vert-sarcelle profond sur ivoire — pour les captures d'écran en rapport et les salles éclairées."),
 dict(id="minuit", nom="Minuit & Iris", mode="dark", ordre=3,
      cv=(0.135,0.014,285), sf=(0.198,0.018,285), rs=(0.240,0.020,285),
      tx=(0.955,0.006,285), mu=(0.700,0.014,285), bd=(0.320,0.018,285), bs=(0.435,0.022,285),
      ac=(0.780,0.130,290), a2=(0.700,0.150,300),
      pitch="Nuit froide, accent iris. La plus familière aux profils techniques sans tomber dans le bleu générique."),
 dict(id="ardoise", nom="Ardoise & Chaux", mode="dark", ordre=4,
      cv=(0.125,0.008,240), sf=(0.190,0.010,240), rs=(0.232,0.011,240),
      tx=(0.955,0.004,240), mu=(0.700,0.012,240), bd=(0.315,0.011,240), bs=(0.430,0.013,240),
      ac=(0.880,0.170,124), a2=(0.800,0.160,132),
      pitch="Ardoise et vert chaux. La plus contrastée des cinq — un accent qu'on ne rate pas, y compris sur vidéoprojecteur."),
 dict(id="porcelaine", nom="Porcelaine & Cobalt", mode="light", ordre=5,
      cv=(0.962,0.005,240), sf=(1.000,0.000,240), rs=(0.980,0.004,240),
      tx=(0.215,0.020,255), mu=(0.510,0.014,250), bd=(0.890,0.008,245), bs=(0.775,0.011,245),
      ac=(0.450,0.190,265), a2=(0.390,0.185,270),
      pitch="Porcelaine et cobalt. La plus institutionnelle — celle qui passe sans discussion dans un grand compte."),
]
# lightness de départ des sémantiques, par mode
SEM = {
 'ok' : dict(H=158, C=0.140, dark=0.740, light=0.520),
 'wa' : dict(H=92,  C=0.135, dark=0.810, light=0.560),
 'da' : dict(H=27,  C=0.190, dark=0.680, light=0.545),
 'inf': dict(H=222, C=0.100, dark=0.750, light=0.520),
}
TARGET_TEXT, TARGET_UI = 4.5, 3.0

res={}
print(f"{'thème':<21}{'jeton':<9}{'hex':<9}{'oklch':<26}{'cv':>6}{'sf':>7}{'rs':>7}")
print("="*82)
for T in THEMES:
    cv,sf,rs = hexg(*T['cv']), hexg(*T['sf']), hexg(*T['rs'])
    bgs=[cv,sf,rs]
    r={'nom':T['nom'],'mode':T['mode'],'ordre':T['ordre'],'pitch':T['pitch'],
       'canvas':cv,'surface':sf,'raised':rs,
       'okl':{'canvas':T['cv'],'surface':T['sf'],'raised':T['rs']}}
    def add(name,L,C,H,target,keepdesign=True):
        Lf,h,Cf = nudge(L,C,H,bgs,target,T['mode'])
        r[name]=h; r['okl'][name]=(Lf,round(Cf,3),H)
        mn=min(contrast(h,b) for b in bgs)
        flag='' if mn>=target else '  ⚠ ÉCHEC'
        print(f"{T['nom'][:19]:<21}{name:<9}{h:<9}{'%.3f %.3f %g'%(Lf,Cf,H):<26}"
              f"{contrast(h,cv):>6.2f}{contrast(h,sf):>7.2f}{contrast(h,rs):>7.2f}{flag}")
        return mn
    mins=[]
    mins.append(add('text',*T['tx'],TARGET_TEXT))
    mins.append(add('muted',*T['mu'],TARGET_TEXT))
    mins.append(add('accent',*T['ac'],TARGET_TEXT))
    add('accentStrong',*T['a2'],1.0)
    for k,s in SEM.items():
        mins.append(add(k,s[T['mode']],s['C'],s['H'],TARGET_TEXT))
    # bordures : décorative (>=1.5 vs surface) et forte (>=3 vs surface, pour champs et focus)
    Lb,bd,Cb = nudge(T['bd'][0],T['bd'][1],T['bd'][2],[sf],1.5,T['mode'])
    r['border']=bd; r['okl']['border']=(Lb,round(Cb,3),T['bd'][2])
    Lb2,bs2,Cb2 = nudge(T['bs'][0],T['bs'][1],T['bs'][2],[sf],TARGET_UI,T['mode'])
    r['borderStrong']=bs2; r['okl']['borderStrong']=(Lb2,round(Cb2,3),T['bs'][2])
    print(f"{'':21}{'border':<9}{bd:<9}{'%.3f %.3f %g'%(Lb,Cb,T['bd'][2]):<26}{contrast(bd,sf):>20.2f}")
    print(f"{'':21}{'bordFort':<9}{bs2:<9}{'%.3f %.3f %g'%(Lb2,Cb2,T['bs'][2]):<26}{contrast(bs2,sf):>20.2f}"
          + ('' if contrast(bs2,sf)>=3 else '  ⚠'))
    r['min']=round(min(mins),2)
    print(f"{'':21}→ contraste texte minimum de la palette : {r['min']}:1")
    print("-"*82)
    res[T['id']]=r
json.dump(res,open('themes.json','w'),indent=1,ensure_ascii=False)
print("\nRÉCAP")
for k,v in res.items(): print(f"  {v['nom']:<22}{v['mode']:<7}min {v['min']}:1")
