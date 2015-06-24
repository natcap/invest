import sys, os, string, time, datetime, json
from datetime import datetime
from math import *
import warnings

import numpy as num
from scipy import optimize
from scipy import stats
from scipy import interpolate
from pylab import *
from pylab import find
from matplotlib import *
import fpformat, operator

import CPf_SignalSmooth as SignalSmooth

#warnings.filterwarnings('error', "invalid value encountered in divide")

g=9.81

def Fast_k(T,h):
    if not (h >= 0.05).all():
        too_shallow = np.where(h >= 0.05)
        print('Some depths are too shallow in h of size', h.size)
        for i in too_shallow:
            print(i, h[i])

    assert (h >= 0.05).all(), \
        'Detected depths that are too shallow'
    g=9.81;

    if type(h) is list:
        h=array(h)

    # Doesn't allow mu0 to be negative
    muo=4.0*pi**2*h/(g*T**2)

    expt=1.55+1.3*muo+0.216*muo**2
    Term=1.0+muo**1.09*num.exp(-expt)
    mu=muo*Term/num.sqrt(num.tanh(muo))

    k=mu/h
    n=.5*(1.0+(2.0*k*h/sinh(2.0*k*h)))
    C=2*pi/T/k
    Cg=C*n ; #Group velocity
    
    if type(n) is numpy.ndarray:
        out=h<.05
        k[out]=nan;C[out]=nan;Cg[out]=nan
        
    return k,C,Cg

def Indexed(x,value): # locates index of point in vector x (type = num.array) that has closest value as variable value
    if isinstance(x,list):
        x=num.array(x)        
    mylist=abs(x-value);    
    if isinstance(x,num.ndarray):
        mylist=mylist.tolist()
    minval=min(mylist)
    ind=[i for i,v in enumerate(mylist) if v==minval]
    ind=ind[0]
    return ind
#End of Indexed

def MaxIndex(x): 
    ''' Return list of position(s) of largest element '''
    m = max(x)
    ind=[i for i, j in enumerate(x) if j == m]
    return m,ind
#End of MaxIndex

def FindRootKD(fun,a,b,BetaKD,tol=1e-16):
    a=float(a);b=float(b)
    assert(sign(fun(a)) != sign(fun(b)))
    c=(a+b)/2
    while math.fabs(fun(c))>tol:
        if a==c or b==c:
            break
        if sign(fun(c))==sign(fun(b)):
            b=c
        else:
            a=c
        c=(a+b)/2
    return c
#End of FindRootKD

def Erosion_Quick(Ho,To,S,B,D,W,Dur,m,A,z):
    g=9.81;rho=1024;dx=1;
    TD=Dur;BD=B+D;
    Lo=g*To**2*1.0/(2*pi);
    Gam=0.78;
    Co=g*To/(2.0*pi);#deep water phase speed with period
    hb=1.0/(g**(1*1.0/5)*Gam**(4*1.0/5))*(((Ho**2)*Co)/2)**(2*1.0/5);#breaking wave depth
    xb=(hb/A)**1.5;#surf zone width
    Hb=Gam*hb;
    
    Coef=0.5*(1.0-num.cos(2.0*z)) # final erosion distance

    Term1=xb-hb/m
    if Term1<=0:
        m=floor(xb/hb)
        Term1=xb-hb/m
        
    Rinf1=(S*Term1)/(B+hb-S/2.) # erosion without taking width into account   
    if D>0:
        Rinf=(S*Term1)/(BD+hb-S/2.)-W*(B+hb-0.5*S)/(BD+hb-0.5*S) # potential erosion distance
    if Rinf < 0:
        Rinf=(S*Term1)/(B+hb-S/2.); 
    else:
        Rinf=Rinf1
    
    R=Rinf*Coef;    # final erosion distance    
    return R,Rinf,m

def ErosionFunction(A,m,xb,hb,Hb,S,B,D,W,Dur,z):
    g=9.81;xb=float(xb);hb=float(hb);Hb=float(Hb);
    S=float(S);B=float(B);D=float(D);BD=B+D;
    Coef=0.5*(1.0-num.cos(2.0*z)) # final erosion distance
    
    Term1=xb-hb/m
    if Term1<=0:
        m=floor(xb/hb)
        Term1=xb-hb/m
        
    Rinf1=(S*Term1)/(B+hb-S/2.) # erosion without taking width into account   
    if D>0:
        Rinf=(S*Term1)/(BD+hb-S/2.)-W*(B+hb-0.5*S)/(BD+hb-0.5*S) # potential erosion distance
    if Rinf < 0:
        Rinf=(S*Term1)/(B+hb-S/2.); 
    else:
        Rinf=Rinf1
    R=Rinf*Coef;
    if R<0:
        R=0.0
    
    if m>.5:
        R=0;
        
    return R,Rinf,m
#End of ErosionFunction

# calculates the wave number given an angular frequency (sigma) and local depth (dh)
def iterativek(sigma,dh):
    g=9.81;rho=1024;
    kestimated=(sigma**2)/(g*(num.sqrt(num.tanh((sigma**2)*dh/g)))) # intial guess for the wave number
    tol = 1 # initialize the tolerence that will determine if the iteration should continue
    count=0
    while (tol>0.00005) and (count<1000): # iterate over the dispersion relationship
        count += 1
        kh=kestimated*dh
        tol = kestimated-(sigma**2)/(num.tanh(kh)*g) # check the num.difference between the previous and the current iterations
        kcalculated=(sigma**2)/(num.tanh(kh)*g) # k values for current iteration
        kestimated=kcalculated # set the current value for k as the previous value for the subsequent iteration
    qk=kcalculated # return the value of k when accuracy tolerance is achieved
    return qk
#End of iterativek

# wind-wave generation
def WindWave(U,F,d):
    # calculates wind waves which are a function of Wind Speed (U), Fetch length (F), and local depth (d) unum.sing emnum.pirical equations
    g=9.81;rho=1024;
    ds=g*d/U**2.0;Fs=g*F/U**2.0
    A=num.tanh(0.343*ds**1.14)
    B=num.tanh(4.41e-4*Fs**0.79/A)
    H=0.24*U**2*1.0/g*(A*B)**0.572 # wave height
    A=num.tanh(0.1*ds**2.01)
    B=num.tanh(2.77e-7*Fs**1.45*1.0/A)
    T=7.69*U/g*(A*B)**0.18 # wave period
    return H,T
#End of WindWave

def MudErosion(Uc,Uw,h,To,me,Cm):
    rho=1024.0;nu=1.36e-6;d50=0.00003
    ks=2.5*d50;kap=0.4
    if isinstance(Uc,list):
        Uc=num.array(Uc)        
        Uw=num.array(Uw)        

    # current
    if max(abs(Uc))>0:
        us1=0.01;zo1=0.01;dif=10 # initial value for u* and zo  
        Tc=h*0; # shear stress
        while dif>1e-4:
            zo2=ks/30*(1-num.exp(-us1*ks/(27*nu)))+nu/(9*us1)
            us2=kap*Uc/(num.log(h/zo2)-1)
            dif1=abs(us1-us2);dif2=abs(zo1-zo2);dif=mean(dif1+dif2);
            zo1=zo2;us1=us2;
        Tc=rho*us1**2 # shear stress due to current
    else:    Tc=h*0

    # waves
    Rw=1.0*Uw**2*To/(2*num.pi)/nu
    fw=0.0521*Rw**(-0.187) # smooth turbulent flow
    Tw=0.5*rho*fw*Uw**2

    # combined wave and current
    temp=Tc*(1+1.2*(Tw/(Tc+Tw))**3.2)
    Trms=(temp**2+0.5*Tw**2)**0.5

    # erosion 
    Te=h*0+0.0012*Cm**1.2 # erosion threshold
    dmdt=me*(Trms-Te) # erosion rate
    dmdt[find(dmdt<=0)]=0
    Erosion=3600*dmdt/Cm*100 # rate of bed erosion [cm/hr]

    # erosion: bed erosion rate, Trms: wave and current shear stress, Tc: current shear stress, Tw: wave shear stress, Te: erosion threshold
    return Erosion,Trms,Tc,Tw,Te
#End of MudErosion

def ParseSegment(X):
    # this function returns location of beg. and end of segments that have same value in a vector
    # input: X vector 
    # output: 
    # - Beg and End_ar: indices of beginning and end of segments that have same value
    # - L_seg, Values: length of segment and value of terms in that segment

    X=num.array(X)    
    End_ar=num.nonzero(num.diff(X));End_ar=End_ar[0] # ID locations BEFORE number changes

    Beg_ar=num.append([0],End_ar+1) # index of bef. of segments
    End_ar=num.append(End_ar,len(X)-1) # index of end of segments
    L_seg=End_ar-Beg_ar+1 # length of the segments
    Values=X[Beg_ar]

    return Values,Beg_ar,End_ar,L_seg
#End of ParseSegment

def SeawallOvertop(Hs,Tp,htoe,Hwall):

    #Input:
    # Hs: significant wave height at the toe of the structure
    # Tp: peak wave period
    #- htoe: depth @toe
    #Hwall: height of the wall
    
    #Output:
    # Q_det: deterministic overtopping
    # Q_prob: probabilistic overtopping
    
    g=9.81;Hs=float(Hs);Tp=float(Tp);htoe=float(htoe);Hwall=float(Hwall)
    Tmo=Tp/1.1;
    hstar=1.35*htoe/Hs*(2*pi*htoe)/(g*Tmo**2);
    Rc=Hwall-htoe;
    #Case if hstar>.3 - Non-impulsive
    if Rc/Hs<.1 or Rc/Hs>3.5:
        qp1=NaN;qd1=NaN;
    else:
        qp1=0.04*exp(-2.6*Rc/Hs)*sqrt(g*Hs**3);
        qd1=0.04*exp(-1.8*Rc/Hs)*sqrt(g*Hs**3);
    
    #Case if hstar<.2 - Impulsive
    qp2=1.5e-4*(hstar*Rc/Hs)**(-3.1)*(hstar**2*sqrt(g*htoe**3));
    qd2=2.8e-4*(hstar*Rc/Hs)**(-3.1)*(hstar**2*sqrt(g*htoe**3));
    if hstar*Rc/Hs>1:
        qp2=NaN;qd2=NaN;

    if hstar*Rc/Hs<0.02:
        qp2=2.7e-4*(hstar*Rc/Hs)**(-2.7)*(hstar**2*sqrt(g*htoe**3));
        qd2=3.8e-4*(hstar*Rc/Hs)**(-2.7)*(hstar**2*sqrt(g*htoe**3));
    
    if hstar>.3:
        qd=qd1;qp=qp1;
    elif hstar<.2:
        qd=qd2;qp=qp2;
    elif hstar<=.3 and hstar>=.2:
        qd=max(qd1,qd2);qp=max(qp1,qp2);

    Q_det=qd*1.0e3;Q_prob=qp*1.0e3;#Outputs in l/m/s
    if isnan(Q_prob):
        msg='Non Damage';
        Q_det=-1;Q_prob=-1;
    elif Q_prob<2:
        msg='Non Damage';
    elif Q_prob<10:
        msg='Som Damage';
    elif Q_prob<50:
        msg='Mor Damage';
    elif Q_prob>=50:
        msg='Yes Damage';
    return Q_det,Q_prob,msg
#End seawall overtop

def LeveeOvertop(Hmo,Tp,Slope,CrestHeight,ToeDepth,Berm,Veg):
    #Hmo=Wave[-1];Tp=To;Slope=tanalph;CrestHeight=dikeHeight;Berm=0;Veg=0
    #Computes average overtopping rate cubic meters per second per unit length of the levvee (m**3*1.0/s/m).

    #Inputs
    #Hmo: Significant Wave Height at the Toe of the Levee/Dike
    #Tp: Peak Wave Period
    #Slope: The slope of the face of the Levee/Dike (Tangent of the angle of the Levee from horizontal)
    #CrestHeight: The Height of the Crest of the Levee/Dike relative to the ground elevation at the toe of the levee.
    #ToeDepth: The water depth at the toe of the Levee/Dike (SWL-Storm and Tide only, no wave effects)
    #Berm: Is a berm present on the face of the Levee/Dike? 0, if no. 1, if yes.
    #Veg: Is the berm smooth (0) or vegetated with grass (1).

    g=9.81
    Tmo = Tp/1.1 #The conversion from peak period to spectural period used in Eurotop
    Lo = g*Tmo**2.0/(2.0*num.pi) #Eurotop uses the deep water wave length
    IrNum = Slope/num.sqrt(Hmo/Lo)
    Rc = CrestHeight-ToeDepth #Freeboard
    term1 = num.sqrt(g*Hmo**3.0) #A scaling term used in all wave overtopnum.ping formulae

    #Adjustment factors
    gambeta = 1.0 #Angle of attack. Assume normally indcident waves.
    gamvert = 1.0 #Vertical wall. Assume that there is not a vertical barrier along the crest of the Levee

    if (Veg == 1.0 and Hmo <= 0.75):
        gamf = 1.15*Hmo**0.5 #The friction factor. If the Levee is vegetated and the wave height is small enough, the increased roughness due to grass has an impact.
    else:
        gamf = 1.0

    #Berm Factor
    if Berm == 0.0: #There is no berm present
        gamberm = 1.0 
    else:
        width = .25*Lo #The greatest berm width according to the Eurotop manual.
        BermHeight = 0.5*CrestHeight #Simply place the flat berm halfway up the face of the levee.    
        if(ToeDepth - Hmo) >= BermHeight and (ToeDepth + Hmo) <= BermHeight: #The berm is outside the zone of influence
            gamberm = 1.0
        else:
            rb = width/(2.0*Hmo/Slope+width) #Ratio of the width of the berm relative to the horizontal length of the levee between +/- 1 Hmo  
            if ToeDepth >= BermHeight: #If the berm is inundation
                rdb = 0.5-0.5*num.cos(num.pi*((ToeDepth-BermHeight)/(2*Hmo)))
            else: #If the berm is above the still water elevation
                #rdb (a component of the berm factor) is a function of the 2#Runup value . . . 
                R2 = num.min([1.75*gambeta*gamf*0.8*IrNum*Hmo,gambeta*gamf*0.8*(4.3-(1.6/num.sqrt(IrNum)))*Hmo]) #...which is a function of the berm factor, therefore an iteration with need to be performed. 0.8 is the first guess of for the berm factor in the iteration
                rdb = 0.5-0.5*num.cos(num.pi*((BermHeight-ToeDepth)/R2init))
                gamberm = 1.0-rb*(1.0-rdb) #Inital berm factor for iteration
                num.diff = num.absolute(0.8-gamberm)
                while num.diff > 0.01: #Tolerance for iteration 
                    gamberminit = gamberm
                    R2 = num.min([1.75*gambeta*gamf*gamberm*IrNum*Hmo,gambeta*gamf*gamberm*(4.3-(1.6/num.sqrt(IrNum)))*Hmo])
                    rdb = 0.5-0.5*num.cos(num.pi*((BermHeight-ToeDepth)/R2init))
                    gamberm = 1.0-rb*(1.0-rdb) #Inital berm factor for iteration
                    num.diff = num.absolute(gamberminit-gamberm)
            gamberm = 1.0-rb*(1.0-rdb)
        if gamberm < 0.6:
            gamberm = 0.6 #The minimum value for the berm factor is 0.6

    #Now that the adjustment factors have been computed, we can compute overtopnum.ping.

    #Wave overtopping only (positive freeboard):
    if Rc > 0 and IrNum<=5.0: #num.different overtopnum.ping formula for num.different ranges if Iribarren Number
        q = term1*(0.067/num.sqrt(Slope))*gamberm*IrNum*num.exp(-4.3*Rc/(IrNum*Hmo*gamberm*gamf*gambeta*gamvert))
        debug=1
    elif Rc > 0 and IrNum>=7.0:
        q = term1*0.21*num.exp(-Rc/(gamf*gambeta*Hmo*(0.33+.022*IrNum)))
        debug=2
    elif Rc > 0 and IrNum>5.0 and IrNum<7.0:
        #There is no equation for Iribarren Number between 5 and 7 so an interpolation must be applied
        q5 = term1*(0.067/num.sqrt(Slope))*gamberm*IrNum*num.exp(-4.3*Rc/(IrNum*Hmo*gamberm*gamf*gambeta*gamvert))
        q7 = term1*0.21*num.exp(-Rc/(gamf*gambeta*Hmo*(0.33+.022*IrNum)))
        q = (IrNum - 5.0)/2.0*(q7-q5)+q5
        debug=3
    #Wave overtopnum.ping and overflow (negative freeboard or zero freeboard)
    if Rc <= 0:
        qoverflow = 0.6*num.sqrt(g*num.absolute(Rc**3))
        if IrNum<=2.0:
            qwave = 0.0537*IrNum*term1
        else:
            qwave = term1*(.136-(.226/IrNum**3.0))
        q = qwave+qoverflow
        debug=4

    #Calculated the minimum height required to achieve a safe ovetopping rate:
    if IrNum <=5.0:
        RcReq = (IrNum*Hmo*gamf/-4.3) * num.log(1e-6*num.sqrt(Slope)/(term1*0.067*IrNum))
    elif IrNum >=7.0:
        RcReq = (-Hmo*gamf*(0.33+0.022*IrNum))*num.log(1e-6/(0.21*term1))
    else:
        RcReq5 = (IrNum*Hmo*gamf/-4.3) * num.log(1e-6*num.sqrt(Slope)/(term1*0.067*IrNum))
        RcReq7 = (-Hmo*gamf*(0.33+0.022*IrNum))*num.log(1e-6/(0.21*term1))
        RcReq = (IrNum - 5.0)/2.0*(RcReq7-RcReq5)+RcReq5
    RcReq=RcReq-Rc
    
    if RcReq<0:
        RcReq=0

    return q, RcReq
#End of OvertoppingLevee

def  SafetyMessage(struc,q):
    if struc=="Vehicles":
        if q>5e-4:
            msg="Unsafe at any speed"
        elif q>1e-4:
            msg="Unsafe parking on horizontal composit breakwaters"
        elif q>2.5e-5:
            msg="Unsafe parking on vertical wall breakwaters"
        elif q>1e-6:
            msg="Unsafe driving at high speed"
        else:
            msg="Safe driving at any speed"
    elif struc=="Pedestrian":               
        if q>1e-3:
            msg="Very dangerous amount of overtopping"
        elif q>5e-4:
            msg="Large amount of overtopping"
        elif q>5e-5:
            msg="Significant amount of overtopping"
        elif q>5e-6:
            msg="Some amount of overtopping"
        elif q>1e-7:
            msg="Small amount of overtopping"
        else:
            msg="No overtopping"
    elif struc=="Buildings":
        if q>1e-5:
            msg="Structural damage"
        elif q>1e-6:
            msg="Minor damage to fittings, signposts, etc."
        else:
            msg="No damage"
    elif struc=="Embankment & Seawalls":
        if q>5e-2:
            msg="Damage even if fully protected"
        elif q>1e-2:
            msg="Damage if back-slope not protected"
        elif q>5e-3:
            msg="Damage if crest not protected"
        else:
            msg="No damage"
    elif struc=="Grass & Sea-Dikes":
        if q>1e-2:
            msg="Damage"
        elif q>1e-1:
            msg="Start of damage"
        else:
            msg="No damage"
    elif struc=="Revetments":
        if q>.5:
            msg="Damage even for paved promenade"
        elif q>.05:
            msg="Damage if promenade not paved"
        else:
            msg="No damage"

    return msg
#End of SafetyMessage

def WindWave_Deep(U_a,F): 
    ii=1;g=9.81
    while ii<5:
        #tim=68.8*(U_a/g)*(g*F/U_a**2)**.666;
        #if tim>3600: #Duration limited conditions
            #F=(g*3600/(U_a*68.8))**1.5*U_a**2*1.0/g;
        #else:
            #ii=5;

        ii=5
        Fs=g*F/U_a**2;
        if F>=23123.0*(1.0*U_a**2.0/g): #Fully arisen sea
            Hdeep=0.243*(1.0*U_a**2.0/g);
            Tdeep=8.134*(1.0*U_a/g);
        else:    
            Hdeep=0.0016*(U_a**2.0/g)*Fs**.5;
            Tdeep=0.2857*(U_a/g)*Fs**.333;
        ii=ii+1;
    return Hdeep,Tdeep
#End of WindWave_Deep

def WindWave_Holjuisten(U,F,d):
    ds=g*d/U**2;Fs=g*F/U**2;
    A=num.tanh(0.343*ds**1.14);B=num.tanh(4.14e-4*Fs**0.79/A);
    H=0.24*U**2.0/g*(A*B)**0.572;
    A=num.tanh(0.1*ds**2.01);B=num.tanh(2.77e-7*Fs**1.45*1.0/A);
    T=7.69*U/g*(A*B)**0.187; 
    return H,T
#End of WindWave_Holjuisten

def WindWave_Shallow(U,F,d):
    ds=g*d/U**2;Fs=g*F/U**2;
    A=num.tanh(0.530*ds**(3.0/4));B=num.tanh((0.00565*Fs**.5)/A);
    H=0.283*U**2.0/g*(A*B);
    A=num.tanh(0.833*ds**(3.0/8));B=num.tanh((0.0379*Fs**.333)/A);
    T=7.54*U/g*(A*B);
    return H,T
#End of WindWave_Shallow

def TimeAdjust(time,U3600): #Adjusts the velocity w.r.t. U3600 and duration of wind field
    if time>3600:
        U=U3600*(-0.15*num.log10(time)+1.5334);
    elif time<3600:
        U=U3600*(1.277+0.296*num.tanh(0.9*num.log10(45*1.0/time)));
    return U
#End of TimeAdjust

def gradient2(U,z):
    #dU=gradient2(U,z)    
    lz=len(z);dU=U*0;
    dU[0]=(U[1]-U[0])/(z[1]-z[0]);
    for uu in range(1,lz-1,1):
        dU[uu]=0.5*(U[uu+1]-U[uu])/(z[uu+1]-z[uu])+0.5*(U[uu]-U[uu-1])/(z[uu]-z[uu-1])
    dU[lz-1]=(U[lz-1]-U[lz-2])/(z[lz-1]-z[lz-2]);
    return dU
#End of Gradient2

def LocateBreak(TS):
    enew=TS;s=len(enew);
    b=[0.0]*(s)
    # Find the Crossing Points:
    for i in range(0,s-1):
        temp=enew[i]*enew[i+1];
        if temp<0:
            if enew[i]<0:
                b[i+1]=1 #Upcrossing location
            else:
                b[i+1]=0
        else:
            b[i+1]=0;
        
    j=0;wave=[];
    for k in range(0,s-1):
        if b[k]==1:
            j=j+1;
            wave.append(k);

    zeroloc=wave;
    return zeroloc
#End of LocateBreak

def FindBreaker(x,h,H,T,Sr):
    g=9.81;Lo=g*T**2.0/(2.0*pi);

    keep=find(h>.1)
    x=x[keep];h=h[keep];    lx=len(x)
    H=H[keep];Sr=Sr[keep];

    Cr=find(Sr==11);
    if len(Cr)>0:
        Deep=Cr[-1]+10
    else:
        Deep=0;
    
    ## Breaker
    a=H[Deep:lx]-0.78*h[Deep:lx];
    #a=SignalSmooth.smooth(num.array(a),len(a)*0.01,'hanning')     
    l1=LocateBreak(a);ix=x[Deep:lx];
    
    if len(l1)==0:
        xb=1.0;Hb=1;
        l1=len(h)-1;
    else:
        l1=l1[-1]+Deep-1;
        Hb=H[l1];
    loc=l1;
    return loc,Hb

def Runup_ErCoral(x,h,H,Ho,Eta0,Eta1,T,A,m,Sr):
    keep=find(h>.1)
    x=x[keep];h=h[keep];
    H=H[keep];Eta0=Eta0[keep];
    Eta1=Eta1[keep];Sr=Sr[keep];
    
    g=9.81;Lo=g*T**2.0/(2.0*pi);
    lx=len(x)
    Cr=find(Sr==11);
    if len(Cr)>0:
        Deep=Cr[-1]+10
    else:
        Deep=0;
    
    ## Breaker
    a=H[Deep:lx]-0.78*h[Deep:lx];
    #a=SignalSmooth.smooth(num.array(a),len(a)*0.01,'hanning')     
    l1=LocateBreak(a);ix=x[Deep:lx];
    
    if len(l1)==0:
        xb=1.0;hb=A*xb**(2.0/3.0);Hb=0.78*hb;
        l1=len(h)-1;
    else:
        l1=l1[-1]+Deep-1;
        xb=abs(x[-1]-x[l1]);hb=A*xb**(2.0/3);Hb=0.78*hb;
        Hb=H[l1];hb=Hb/.78;xb=(hb/A)**(3.0/2);

    l=(l1-1);
    while H[l]>H[l+1]:
        l=l-1;

    if len(Cr)>0:
        l=Cr[-1];
    else:
        l=0;
    ## Runup
    if Eta0[-1]<0:
        Eta0[-1]=0;
    if Eta1[-1]<0:
        Eta1[-1]=0;
    temp=H[l:lx];
    Hm,temp=MaxIndex(temp)
    
    temp=temp[0]+l;
    sig=2.0*pi/T;
    ki=iterativek(sig,h[temp]);Li=2.0*pi/ki;
    n=.5*(1.0+(2.0*ki*h[temp]*1.0/sinh(2.0*ki*h[temp]))); #to compute Cg
    C=Li/T;  Cg=C*n; Co=Lo/T;Cgo=0.5*Co;
    H_o=sqrt(Hm**2.0*Cg/Cgo);H_o=round(H_o,2);
    Rs0=1.1*(0.35*m*(H_o*Lo)**0.5+((H_o*Lo*0.563*m**2.0+Ho*Lo*0.004)**0.5)/2.0);
    Rs0=round(Rs0,2);

    if Eta0[-1]==0 or Eta1[-1]==0:
        Etap=0.0
    else:
        coef0=Eta0[-1]/(0.35*m*sqrt(H_o*Lo)); # To correct eta with vegetation
        Etap=Eta1[-1]/coef0; #Corrected MWL
    Hp=(Etap*1.0/(0.35*m))**2.0/Lo; #Hprime to estimate runup
    Hp=round(Hp,2);Eta=round(Etap,2)
    Rs1=1.1*(Etap+sqrt(Lo*(Hp*0.563*m**2.0+0.004*Ho))/2.0);# Runup with vegetation
    Rs1=round(Rs1,2);
   
    return Rs0,Rs1,xb,hb,Hb,Etap,Hp
#End Runup_ErCoral

def   WaveRegenWindCD(Xnew,bath_sm,Surge,Ho,To,Uo,Cf,Sr,PlantsPhysChar):
    # x: Vector of consecutive cross-shore positions going shoreward
    # h: Vector of water depths at the corresponding cross-shore position
    # Ho: Initial wave height to be applied at the first cross-shore position
    # To: Wave Period
    # Roots: An num.array of physical properties (density, diameter, and height) of mangrove roots at the corresponding cross-shore position (all zeros if mangroves don't exist at that location)
    # Trunk: An num.array of physical properties (density, diameter, and height) of mangrove trunks or the marsh or seagrass plants at the corresponding cross-shore position (all zeros if vegetation don't exist at that location)
    # Canop: An num.array of physical properties (density, diameter, and height) of the mangrove canopy at the corresponding cross-shore position (all zeros if mangroves don't exist at that location)
    # ReefLof: Location of reef
    
    
    # constants
    g=9.81;rho=1024.0;B=1.0;Beta=0.05;
    lxo=len(Xnew);dx=num.diff(Xnew);dx=abs(dx)
    factor=3.0**(-2);
    
    #Compute wind reduction factor and reduce surge
    zo=num.zeros(lxo)
    S=num.zeros(lxo)
    temp=find(Sr==7)#Marshes
    if temp.any():
        zo[temp]=0.11
        S[temp]=SurgeReduction(SurgeRed[temp])
    temp=find(Sr==8)#mangroves
    if temp.any():
        zo[temp]=0.55
        S[temp]=SurgeReduction(SurgeRed[temp])
    
    #bathymetry
    ho=num.array(-bath_sm+S)
    out=find(ho<.05);
    if out.any():
        out=out[0]
    else:
        out=len(ho)
    h=ho[0:out-1]
    Xnew=num.array(Xnew)
    x=Xnew[0:out-1];
    lx=len(x)
    
    #Create wind vector
    if Uo<>0:
        Cd_airsea=(1.1+0.035*Uo)*1e-3;
        Zo_marine=0.018*Cd_airsea*Uo**2*1.0/g;#Roughness water
        Zo=[max(Zo_marine,zo[ii]-h[ii]/30) for ii in range(lx)] #Reduction in roughness b/c veg. underwater
        Zo=num.array(Zo)
        fr=(Zo/Zo_marine)**0.0706*log(10*1.0/Zo)/log(10*1.0/Zo_marine);#Reduction factor
        U=fr*Uo;
    else:
        U=num.zeros(lx)+.0001;
    Ua=0.71*U**1.23;
    
    # Vegetation characteristics
    Roots=PlantsPhysChar['Roots']
    hRoots=Roots['RootHeight'];
    NRoots=Roots['RootDens']
    dRoots=Roots['RootDiam']
    CdR=Roots['RootCd']
    
    Trunks=PlantsPhysChar['Trunks']
    hTrunk=Trunks['TrunkHeight'];
    NTrunk=Trunks['TrunkDens']
    dTrunk=Trunks['TrunkDiam']
    CdT=Trunks['TrunkCd']
    
    Canop=PlantsPhysChar['Canops']
    hCanop=Canop['CanopHeight'];
    NCanop=Canop['CanopDens']
    dCanop=Canop['CanopDiam']
    CdC=Canop['CanopCd']
    
    # create relative depth values for roots, trunk and canopy
    alphr=hRoots/ho;alpht=hTrunk/ho;alphc=hCanop/ho
    for kk in range(lx): 
        if alphr[kk]>1:
            alphr[kk]=1;alpht[kk]=0.000000001;alphc[kk]=0.00000001 # roots only
        elif alphr[kk]+alpht[kk]>1:
            alpht[kk]=1-alphr[kk];alphc[kk]=0.000000001 # roots and trunk
        elif alphr[kk]+alpht[kk]+alphc[kk]>1:
            alphc[kk]=1-alphr[kk]-alpht[kk] # roots, trunk and canopy
    
    #Read Oyster reef Characteristics
    if Sr[Sr==3].any(): #Oyster reef
        oyster=PlantsPhysChar['Oyster']
        ReefLoc=ArtReefP[0]
        hc=ArtReefP[1]
        Bw=ArtReefP[2]
        Cw=ArtReefP[3]
        ReefType=ArtReefP[4]
        hi=mean(h[ReefLoc])
        case='main'
    
    #----------------------------------------------------------------------------------------
    #Initialize the model
    #----------------------------------------------------------------------------------------
    
    # Constants 
    H=lx*[0.];Db=lx*[0.];Df=lx*[0.];Diss=lx*[0.];H2=lx*[0.];Dveg=lx*[0.];
    C=lx*[0.];n=lx*[0.];Cg=lx*[0.];k=lx*[0.];L=lx*[0.];T=lx*[0.];Hmx=lx*[0.]
    Er=lx*[0.];Br=lx*[0.]
    
    #Forcing
    Ho=float(Ho);To=float(To);Uo=float(Uo);Surge=float(Surge);Etao=0.0
    sig=2*num.pi/To;fp=1*1.0/To; # Wave period, frequency etc.
    ki,C[0],Cg[0]=Fast_k(To,h[0]) #Wave number, length, etc
    Li=2*num.pi/ki;Lo=g*To**2*1.0/(2*num.pi);
    Kk=[];dd=gradient2(h,x);
    
    #Wave param
    Co=g*To/(2*num.pi);Cgo=Co/2;#Deep water phase speed
    k[0]=ki;
    H[0]=Ho
    T[0]=To;
    
    #Rms wave height
    temp1=2.0*pi/ki
    Hmx[0]=0.1*temp1*tanh(h[0]*ki);#Max wave height - Miche criterion
    if H[0]>Hmx[0]:
        H[0]=Hmx[0];
    
    #Wave and roller energy
    Db[0]=0.00001;Df[0]=0.00001;Diss[0]=0.00001; #Dissipation due to brkg,bottom friction and vegetation
    Dveg[0]=0.00001;Er[0]=0.00001;Br[0]=0.00001;
    
    #Whafis terms
    CorrFact=0.8; #Correction for estimating num.tanh as exp.
    Sin12=lx*[0.];t=lx*[0.];T=lx*[0.];Sin=lx*[0.];Inet=lx*[0.];Term=lx*[0.];L=lx*[0.]
    T2=lx*[0.];T3=lx*[0.];T4=lx*[0.];T5=lx*[0.];T6=lx*[0.];
    H2[0]=H[0]**2;d1=h[0];
    T[0]=To;t[0]=T[0]**3;
    
    at=7.54;gt=0.833;mt=1.0/3.0;sigt=0.0379;#Coeff. for wind wave period
    ah=0.283;gh=0.530;mh=0.5;sigh=0.00565;#Coeff. for wind wave height
    bt1=num.tanh(gt*(g*d1*1.0/Ua[0]**2.0)**0.375);
    bh1=num.tanh(gh*(g*d1*1.0/Ua[0]**2.0)**0.75);
    nut1=(bh1*1.0/sigh)**2.0*(sigt/bt1)**3.0;
    H_inf1=ah*bh1*Ua[0]**2*1.0/g;t_inf1=(at*bt1*Ua[0]/g)**3;
    
    D=d1*1.0/Lo;lam=2*k[0]*d1;
    T2[0]=num.sqrt(Lo*D/(num.sinh(2*num.pi*D)*num.cosh(2*num.pi*D)**3));
    T3[0]=num.tanh(2*num.pi*D)**.5*(1-2*num.pi*D/num.sinh(4*num.pi*D));
    T4[0]=2*num.pi*(1-lam*1*1.0/num.tanh(lam))/num.sinh(lam);
    T5[0]=num.pi/2*(1+lam**2*1*1.0/num.tanh(lam)/num.sinh(lam))*T2[0];
    T6[0]=g/(6*num.pi*T[0])*(1+lam**2*1*1.0/num.tanh(lam)/num.sinh(lam))*T3[0];
    if H[0]<=H_inf1 and t[0]<=t_inf1:
        Sin[0]=CorrFact*(at*sigt)**3.0/g*(Ua[0]/g)**factor*(1-(H[0]/H_inf1)**2.0)**nut1
    else:
        Sin[0]=0.00001
    
    if H_inf1<>0:
        Inet[0]=Cg[0]*T[0]*CorrFact*(sigh*ah*Ua[0])**2*1.0/g*(1-(H[0]/H_inf1)**2)+H[0]**2*T6[0]*Sin[0];
    else:
        Inet[0]=0.00001;
    if h[0]>10:
        Inet[0]=0.00001
    
    Term[0]=(T4[0]+T5[0]/num.sqrt(d1))*dd[0]+T6[0]*Sin[0];# Constants 
    kd=lx*[0.]
    ping1=0;ping2=0;ping3=0;
    
    #----------------------------------------------------------------------------------------
    # Begin wave model 
    #----------------------------------------------------------------------------------------
    for xx in range(lx-1) :#Transform waves, take MWL into account
        if h[xx]>.05: #make sure we don't compute waves in water deep enough
            
            #Determine wave period
            Uxx=Ua[xx] #wind speed
            Uxx1=Ua[xx+1]
            kd[xx]=k[xx]*h[xx]
            if h[xx]>10:
                Uxx=0.00001
                Uxx1=0.00001
        
            d1=h[xx+1];d2=h[xx];
            bt1=num.tanh(gt*(g*d1*1.0/Uxx1**2)**0.375);
            bh1=num.tanh(gh*(g*d1*1.0/Uxx1**2)**0.75);
            nut1=(bh1*1.0/sigh)**2*(sigt/bt1)**3;
            H_inf1=ah*bh1*Uxx1**2*1.0/g;t_inf1=(at*bt1*Uxx1*1.0/g)**3;
            bt2=num.tanh(gt*(g*d2*1.0/Uxx**2)**0.375);
            bh2=num.tanh(gh*(g*d2*1.0/Uxx**2)**0.75);
            nut2=(bh2*1.0/sigh)**2*(sigt/bt2)**3;
            H_inf2=ah*bh2*Uxx**2*1.0/g;t_inf2=(at*bt2*Uxx/g)**3;
        
            #Averages
            H_inf12=mean([H_inf1,H_inf2]);
            nut_12=mean([nut1,nut2]);
            t_inf12=mean([t_inf1,t_inf2]);
        
            #Solve for Period T
            if H[xx]<=H_inf12 and t[xx]<=t_inf12:
                Sin12[xx+1]=CorrFact*(at*sigt)**3*1.0/g*(Uxx1*1.0/g)**factor*(1-(H[xx]/H_inf12)**2)**nut_12;
            else:
                Sin12[xx+1]=0.00001;
        
            t[xx+1]=t[xx]+dx[xx]*Sin12[xx+1];
            T[xx+1]=t[xx+1]**.3333;
            T[xx+1]=To;
            fp=1*1.0/T[xx+1];   
            k[xx+1],C[xx+1],Cg[xx+1]=Fast_k(To,h[xx+1])
        
            D=d1*1.0/Lo;lam=2*k[xx+1]*d1;
            T2[xx+1]=num.sqrt(Lo*D/(num.sinh(2*num.pi*D)*num.cosh(2*num.pi*D)**3));
            T3[xx+1]=num.tanh(2*num.pi*D)**.5*(1-2*num.pi*D/num.sinh(4*num.pi*D));
            T4[xx+1]=2*num.pi*(1-lam*1*1.0/num.tanh(lam))/num.sinh(lam);
            T5[xx+1]=num.pi/2*(1+lam**2*1*1.0/num.tanh(lam)/num.sinh(lam))*T2[xx+1];
            T6[xx+1]=g/(6.0*num.pi*T[xx+1])*(1+lam**2*1.0/num.tanh(lam)/num.sinh(lam))*T3[xx+1];
        
            if H[xx]<=H_inf1 and t[xx+1]<=t_inf1:
                Inet[xx+1]=(Cg[xx+1]*T[xx+1])*CorrFact*(sigh*ah*Uxx1)**2*1.0/g*(1-(H[xx]/H_inf1)**2)+H[xx]**2*T6[xx+1]*Sin12[xx+1];
            else:
                Inet[xx+1]=0.00001;
        
            Term[xx+1]=(T4[xx+1]+T5[xx+1]/num.sqrt(d1))*dd[xx+1]+T6[xx+1]*Sin12[xx+1];# Constants 
        
            #Other Diss. Terms    
            Coral=find(Cf>0.01)
            Gam=0.78;B=1;
            Db[xx]=(3.0/16)*num.sqrt(num.pi)*rho*g*(B**3)*fp*((H[xx]/num.sqrt(2))**7)/ ((Gam**4)*(h[xx]**5)); #Dissipation due to brkg    
            Df[xx]=1.0*rho*Cf[xx]/(16.0*num.sqrt(num.pi))*(2*num.pi*fp*(H[xx]/num.sqrt(2.0))/num.sinh(k[xx]*h[xx]))**3;#Diss due to bot friction 
        
            # dissipation due to vegetation
            V1=3.0*num.sinh(k[xx]*alphr[xx]*h[xx])+num.sinh(k[xx]*alphr[xx]*h[xx])**3.0 # roots
            V2=(3.0*num.sinh(k[xx]*(alphr[xx]+alpht[xx])*h[xx])-3.0*num.sinh(k[xx]*alphr[xx]*h[xx])+
                num.sinh(k[xx]*(alphr[xx]+alpht[xx])*h[xx])**3.0-
                num.sinh(k[xx]*alphr[xx]*h[xx])**3) # trunk
            V3=(3.0*num.sinh(k[xx]*(alphr[xx]+alpht[xx]+alphc[xx])*h[xx])
                -3.0*num.sinh(k[xx]*(alphr[xx]+alpht[xx])*h[xx])+
                num.sinh(k[xx]*(alphr[xx]+alpht[xx]+alphc[xx])*h[xx])**3.0-
                num.sinh(k[xx]*(alphr[xx]+alpht[xx])*h[xx])**3.0) # canopy
        
            CdDN=CdR[xx]*dRoots[xx]*NRoots[xx]*V1+CdT[xx]*dTrunk[xx]*NTrunk[xx]*V2+CdC[xx]*dCanop[xx]*NCanop[xx]*V3
            temp1=rho*CdDN*(k[xx]*g/(2.0*sig))**3.0/(2.0*num.sqrt(num.pi))
            temp3=(3.0*k[xx]*num.cosh(k[xx]*h[xx])**3)
            Dveg[xx]=temp1*1.0/temp3*(H[xx]/num.sqrt(2.0))**3 # dissipation due to vegetation
        
            Fact=16.0/(rho*g)*T[xx+1];
            Diss[xx+1]=Fact*(Db[xx]+Df[xx]+Dveg[xx]);
        
            Inet12=mean([Inet[xx],Inet[xx+1]]);
        
            Term12=mean([Term[xx],Term[xx+1]]);
            if Uo==0: 
                Term12=0;
                Inet12=0;
                
            H2[xx+1]=H2[xx]+dx[xx]/(Cg[xx]*T[xx])*(Inet12-Diss[xx]-H2[xx]*Term12);
            if H2[xx+1]<0:
                H2[xx+1]=1e-4;
            H[xx+1]=num.sqrt(H2[xx+1]);
            Hmx[xx+1]=0.1*(2.0*pi/k[xx+1])*tanh(h[xx+1]*k[xx+1]);#Max wave height - Miche criterion
            #if H[xx+1]>Hmx[xx+1]:
                #H[xx+1]=Hmx[xx+1]
    
            if Sr[xx+1]==3:
                if xx+1 in ReefLoc:
                    Rloc=ReefLoc[0]-1
                    Kt,wavepass,msgO,msgOf,ping1,ping2,ping3=BreakwaterKt(H[Rloc],To,hi,hc,Cw,Bw,case,ping1,ping2,ping3)
                    Kk.append(float(Kt))
                    H[xx+1]=Kt*H[Rloc]
                    H2[xx+1]=H[xx+1]**2.0
            
            Br[xx+1]=Br[xx]-dx[xx]*(g*Er[xx]*sin(Beta)/C[xx]-0.5*Db[xx]) # roller flux
            Er[xx+1]=Br[xx+1]/(C[xx+1]) # roller energy
        
        
        #Art. reef transmission coefficient
        if len(Kk)>0:
            Kk=array(Kk)
            Kt=mean(Kk)
            if Kt>0.98:
                Kt=1.0
        else:
            Kt=1.0;
            
        #Interpolate profile of wave height over reef
        if Sr[xx+1]==3 and Kt<0.95:
            Hrf=array(H)
            Hrf[-1]=Hrf[-2];
            temp1=array(x);temp2=temp1.tolist()
            temp1= [ item for i,item in enumerate(temp1) if i not in ReefLoc ]
            Hrf=[ item for i,item in enumerate(Hrf) if i not in ReefLoc ]
            F=interp1d(temp1,Hrf);
            Hrf=F(temp2)
            H=array(Hrf)
        
    H=SignalSmooth.smooth(num.array(H),len(H)*0.01,'hanning')             
    Ew=lx*[0.0];Ew=[0.125*rho*g*(H[i]**2.0) for i in range(lx)] # energy density
    ash=array(h)
    
    #-------------------------------------------------------------------------------------------------
    #Mean Water Level
    #-------------------------------------------------------------------------------------------------
    # force on plants if they were emergent; take a portion if plants occupy only portion of wc
    Fxgr=[rho*g*CdR[i]*dRoots[i]*NRoots[i]*H[i]**3.0*k[i]/(12.0*num.pi*num.tanh(k[i]*ash[i])) for i in range(lx)]
    Fxgt=[rho*g*CdT[i]*dTrunk[i]*NTrunk[i]*H[i]**3.0*k[i]/(12.0*num.pi*num.tanh(k[i]*ash[i])) for i in range(lx)]
    Fxgc=[rho*g*CdC[i]*dCanop[i]*NCanop[i]*H[i]**3.0*k[i]/(12.0*num.pi*num.tanh(k[i]*ash[i])) for i in range(lx)]
    fx=[-alphr[i]*Fxgr[i]-alpht[i]*Fxgt[i]-alphc[i]*Fxgc[i] for i in range(lx)] # scale by height of indiv. elements
    fx=SignalSmooth.smooth(num.array(fx),len(fx)*0.01,'hanning')     
    
    # estimate MWS
    X=x;
    dx=1;Xi=num.arange(X[0],X[-1]+dx,dx);lxi=len(Xi) # use smaller dx to get smoother result
    F=interpolate.interp1d(X,ash);ashi=F(Xi);
    F=interpolate.interp1d(X,k);ki=F(Xi);
    F=interpolate.interp1d(X,Ew);Ewi=F(Xi);
    F=interpolate.interp1d(X,Er);Eri=F(Xi);
    F=interpolate.interp1d(X,fx);fxi=F(Xi);
    Sxx=lxi*[0.0];Rxx=lxi*[0.0];Eta=lxi*[0.0];O=0;
    
    while O<8: # iterate until convergence of water level
        hi=[ashi[i]+Eta[i] for i in range(lxi)] # water depth        
        
        # Arbitrarily constrain hi to a minimum of -1 meter:
        hi = numpy.maximum(hi, -1.)

        Rxx=[2.0*Eri[i] for i in range(lxi)] # roller radiation stress
        # estimate MWL along Xshore transect
        temp1=[Sxx[i]+Rxx[i] for i in range(lxi)]
        temp2=num.gradient(num.array(temp1),dx)
    
        Integr=[(-temp2[i]+fxi[i])/(rho*g*hi[i]) for i in range(lxi)]
        Eta[0]=Etao
        Eta[1]=Eta[0]+Integr[0]*dx
        for i in range(1,lxi-2):
            Eta[i+1]=Eta[i-1]+Integr[i]*2*dx
        Eta[lxi-1]=Eta[lxi-2]+Integr[lxi-1]*dx
        O=O+1
    F=interpolate.interp1d(Xi,Eta);Eta=F(X);
    
    #Rerun without the vegetation
    Sxx=lxi*[0.0];Rxx=lxi*[0.0];Eta_nv=lxi*[0.0];O=0;
    while O<8: # iterate until convergence of water level
        hi=[ashi[i]+Eta_nv[i] for i in range(lxi)] # water depth        

        # Arbitrarily constrain hi to a minimum of -1 meter:
        hi = numpy.maximum(hi, -1.)

        Sxx=[0.5*Ewi[i]*(4.0*ki[i]*hi[i]/num.sinh(2.0*ki[i]*hi[i])+1.0) for i in range(lxi)] # wave radiation stress
        Rxx=[2.0*Eri[i] for i in range(lxi)] # roller radiation stress
        # estimate MWL along Xshore transect
        temp1=[Sxx[i]+Rxx[i] for i in range(lxi)]
        temp2=num.gradient(num.array(temp1),dx)
    
        Integr=[(-temp2[i])/(rho*g*hi[i]) for i in range(lxi)]
        Eta_nv[0]=Etao
        Eta_nv[1]=Eta[0]+Integr[0]*dx
        for i in range(1,lxi-2):
            Eta_nv[i+1]=Eta_nv[i-1]+Integr[i]*2*dx
        Eta_nv[lxi-1]=Eta_nv[lxi-2]+Integr[lxi-1]*dx
        O=O+1
    F=interpolate.interp1d(Xi,Eta_nv);Eta_nv=F(X);
    
    Ubot=[num.pi*H[ii]/(To*num.sinh(k[ii]*h[ii])) for ii in range(lx)] # bottom velocity
    Ur=[(Ew[ii]+2.0*Er[ii])/(1024.0*h[ii]*C[ii])for ii in range(lx)] 
    Ic=[Ew[ii]*Cg[ii]*Ur[ii]/Ubot[ii] for ii in range(lx)]
    
    H_=num.zeros(lxo)+nan;Eta_=num.zeros(lxo)+nan
    Eta_nv_=num.zeros(lxo)+nan
    Ur_=num.zeros(lxo)+nan;Ic_=num.zeros(lxo)+nan
    Ubot_=num.zeros(lxo)+nan;Kt_=num.zeros(lxo)+nan
    H_[0:lx]=H;Eta_[0:lx]=Eta;Eta_nv_[0:lx]=Eta_nv;
    Ur_[0:lx]=Ur;Ic_[0:lx]=Ic
    Ubot_[0:lx]=Ubot;Kt_[0:lx]=Kt
    other=[0.1*(2.0*pi/k[ii])*tanh((h[ii])*k[ii]) for ii in range(len(k))]
    
    return H_,Eta_,Eta_nv_,Ubot_,Ur_,Kt_,Ic_,Hmx,other # returns: wave height, wave setup, wave height w/o veg., wave setup w/o veg, wave dissipation, bottom wave orbital velocity over the cross-shore domain
        #End of WaveRegen

def OvertopSens(H,To,tanalph,CrestHeight,ToeDepth):
    Q=[];RC=[];
    [q, Rc]=LeveeOvertop(H,To,tanalph,CrestHeight,ToeDepth,0,0)
    Q.append(q);RC.append(Rc)
    [q, Rc]=LeveeOvertop(max(H+.1,H*1.2),To,tanalph,CrestHeight,ToeDepth,0,0)
    Q.append(q);RC.append(Rc)
    temp=min(H-.1,H*.8)
    if temp<0:
        temp=H*.8;    
    [q, Rc]=LeveeOvertop(temp,To,tanalph,CrestHeight,ToeDepth,0,0)
    Q.append(q);RC.append(Rc)
    [q, Rc]=LeveeOvertop(H,To-1,tanalph,CrestHeight,ToeDepth,0,0)
    Q.append(q);RC.append(Rc)
    [q, Rc]=LeveeOvertop(H,To+1,tanalph,CrestHeight,ToeDepth,0,0)
    Q.append(q);RC.append(Rc)
    [q, Rc]=LeveeOvertop(H*1.2,To-1,tanalph,CrestHeight,ToeDepth,0,0)
    Q.append(q);RC.append(Rc)
    [q, Rc]=LeveeOvertop(H*1.2,To+1,tanalph,CrestHeight,ToeDepth,0,0)
    Q.append(q);RC.append(Rc)
    [q, Rc]=LeveeOvertop(H*.8,To-1,tanalph,CrestHeight,ToeDepth,0,0)
    Q.append(q);RC.append(Rc)
    [q, Rc]=LeveeOvertop(H*.8,To+1,tanalph,CrestHeight,ToeDepth,0,0)
    Q.append(q);RC.append(Rc)
    return Q,RC
#End of OvertopSens

def BreakwaterKt(Hi,To,hi,hc,Cwidth,Bwidth,case,ping1,ping2,ping3):
    hi=round(hi,2);hc=round(hc,2);hco=hc;
    Lo=9.81*To**2.0/(2.0*num.pi)
    Rc=hc-hi # depth of submergence
    difht=abs(hc-hi);difht=str(difht)

    if Cwidth<>0:
        ReefType='Trapez'
    else:
        ReefType='Dome'
        
    if Rc>0 and ReefType=="Dome":
        print("The artificial structure is emerged by "+difht+" m. It blocks all incoming waves.We make it smaller so it reaches the water surface")
        hc=hi-.01
        Rc=hc-hi

    
    msgO="";msgOf=""
    wavepass=abs(0.095*Lo*num.tanh(2.0*num.pi*Rc/Lo))
    if Hi<wavepass and hi>hc:
        Kt=1;
        wavepass=1#wave doesn't break
        if case=='main' and ping1==0:
            print("Under current wave conditions, the reef is small enough that it doesn't break the waves. Energy is dissipated via bottom friction.  You can try to increase your reef height to see a larger effect on waves.") #BREAK
            msgO=msgO+"Under current wave conditions, the reef is small enough that it doesn't break the waves. Energy is dissipated via bottom friction.  You can try to increase your reef height to see a larger effect on waves."
            ping1=1
    
    elif ReefType=="Trapez": # it's not a reef ball
        wavepass=0 #wave breaks
        if hc/hi<0.5: #The reef is too submerged
            hc=round(0.5*hi,2);hcf=round(3.28*hc,2);hcof=round(3.28*hco,2);
            if case=='main' and ping3==0:
                    msgO=msgO+"Under current wave conditions, the reef affects waves, but its size is outside the validity range of our simple model. We increase it by "+str(hc-hco)+" m to continue our computation."
                    msgOf=msgOf+"Under current wave conditions, the reef affects waves, but its size is outside the validity range of our simple model. We increase it by "+str(round(3.28*(hc-hco),2))+" ft to continue our computation."
                    ping3=1
        if abs(Rc/Hi)>6:
            hc=round(6*Hi+hi,2)
            if case=='main' and ping2==0:
                    msgO=msgO+"Under current wave conditions, the reef height is above the range of validity of our simple model. We change it to "+str(hc)+" m from "+str(hco)+" m to continue our computation."
                    msgOf=msgOf+"Under current wave conditions, the reef height is above the range of validity of our simple model. We change it to "+str(hcf)+" ft from "+str(hcof)+" ft to continue our computation."
                    ping2=1
            
        Rc=hc-hi # depth of submergence
        Boff=(Bwidth-Cwidth)/2.0 # base dif on each side
        ksi=(hc/Boff)/num.sqrt(Hi/Lo)
        
        # van der Meer (2005)
        Kt1=-0.4*Rc/Hi+0.64*(Cwidth/Hi)**(-.31)*(1.0-num.exp(-0.5*ksi)) # transmission coeff: d'Angremond
        Kt1=max(Kt1,0.075);Kt1=min(0.8,Kt1);
    
        Kt2=-0.35*Rc/Hi+0.51*(Cwidth/Hi)**(-.65)*(1.0-num.exp(-0.41*ksi)) # transmission coeff: van der Meer
        Kt2=max(Kt2,0.05);Kt2=min(Kt2,-0.006*Cwidth/Hi+0.93)
    
        if Cwidth/Hi<8.0: # d'Angremond
            Kt=Kt1
        elif Cwidth/Hi>12.0: # van der Meer
            Kt=Kt2
        else: # linear interp
            temp1=(Kt2-Kt1)/4.0;temp2=Kt2-temp1*12.0
            Kt=temp1*Cwidth/Hi+temp2
            
    else: # it's a reef ball
        Rc=hi-hc
        wavepass=0 #wave breaks
        Kto=1.616-31.322*Hi/(9.81*To**2)-1.099*hc/hi+0.265*hc/Bwidth #D'Armono and Hall

        #New Formula
        Bt=0.6*Bwidth
        KtLow=(-.2496*min(4.0,Bt/sqrt(Hi*Lo))+.9474)**2.0
        KtHigh=1.0/(1+.3*(Hi/Rc)**1.5*Bt/sqrt(Hi*Lo))
        if Rc/Hi<=0.4:
            Kt=KtLow
        elif Rc/Hi>=.71:
            Kt=KtHigh
        else:
            a=(KtLow-KtHigh)/(0.4-0.71)
            b=KtLow-a*0.4
            Kt=a*Rc/Hi+b
        if Kt>1:
            Kt=1
        Kt=min(Kt,Kto)
        print(str(Kt))
        
        if hc>hi:
            Kt=1
            difht=abs(hc-hi);difht=str(difht)
            if case=='main':
                msgO=msgO+"The reef balls are emerged by "+difht+" m. They completely block all incoming waves."
        elif Kt>1:
            Kt=1
            if case=='main':
                msgO=msgO+"Your layout is outside of the range of validity of our simple equation, but it is likely that it doesn't have an effect on waves."
          
    if Kt<0:
        Kt=1  #No transmission - breakwater fully emerged
        if case=='main':
            msgO=msgO+"Your reef is fully emerged and blocks the incoming wave."
    return Kt,wavepass,msgO,msgOf,ping1,ping2,ping3
#End of BreakwaterKt


def Wind(rx,ry,Po,Vfm,Rmax,Zlat):

    rhoa=1.15;#Density of air in kg/m**3
    Om=2.0*pi/86400;#Rate of rotation of earth 
    fc=2.0*Om*sin(Zlat*pi/180)*10;#Coriolis param
    
    Pn=1013.0; #Pressure at periph and center of hur. in mbars
    inflow=25.0*pi/180.0;#Inflow angle
    dP=abs(Po-Pn);
    
    B=2.0-(Po-900.0)/160.0; A=Rmax**B;
    
    # Wind speed
    Ra=num.sqrt(rx**2+ry**2);
    Wa=num.sqrt(A*B*(dP*100)*num.exp(-A*1.0/(Ra**B))*1.0/(rhoa*(Ra**B))+Ra**2.0*fc**2.0/4)-1.0*Ra*fc/2;#Wind speed at that point; convert pres
    
    alpha=[atan2(ry[ii],rx[ii])+2*pi for ii in range(len(rx))];alpha=array(alpha)
    alpha[alpha>=2*pi]=alpha[alpha>=2*pi]-2*pi;#Angles b/n 0 and 2pi
    delta=alpha+pi/2; 
    Thet=delta+inflow; Thet[Thet>2*pi]=Thet[Thet>2.0*pi]-2*pi;#Angle b/n 0 and 2pi   
    Wv=0.8*abs(Wa+Vfm*cos(delta));#Add forward velo to make field asymmetric
    
    # Wind stress
    Wc=5.6;#Use equation of VanDorn for wind stress coef k
    k=1.2e-6+2.25e-6*(1-Wc*1.0/abs(Wv))**2.0;
    k[abs(Wv)<=Wc]=1.2e-6;
    Twx=k*Wv**2.0*cos(Thet);
    Twy=k*Wv**2.0*sin(Thet);
    return Twx,Twy,Wv
#End of wind fn

def Surge1D(X,h,Time,Tide,Sdp,Xh,rx,n,Twx,Twy,fc):

    g=9.81;
    Lt=len(Time);
    Lh=len(h);
    
    #Initialize all vectors that will be used in the model.
    Cd=5e-3+h*0.;  #Drag coef w/t vegetation;
    S_t=[0.]*len(X);S_t1=[0.]*len(X);V_t1=[0.]*len(X);
    S=[[0.]*Lt]*Lh;V=[[0.]*Lt]*Lh;D=[[0.]*Lt]*Lh;bl=[[0.]*Lt]*Lh;
    S=array(S);V=array(V);D=array(D);bl=array(bl)
    for tt in range(0,Lt-1):
        dt=(Time[tt+1]-Time[tt])*3600;    
        RX=X-Xh[tt+1];#Position of each point wrt hurricane eye

        F=interp1d(rx,Twx);Twx_t1=F(RX)
        Twx_t1=[n[ii]*Twx_t1[ii] for ii in range(len(n))];Twx_t1=array(Twx_t1)#values of wind stress along that axis
        F=interp1d(rx,Twy);Twy_t1=F(RX)
        Twy_t1=[n[ii]*Twy_t1[ii] for ii in range(len(n))];Twy_t1=array(Twy_t1)
        F=interp1d(rx,Sdp);Sdp_t1=F(RX)
        Sdp_t1=[n[ii]*Sdp_t1[ii] for ii in range(len(n))];Sdp_t1=array(Sdp_t1)#values of wind stress along that axis
        
        if tt==0:
            Twy_t=Twy_t1;Sdp_t=Sdp_t1;V_t=V_t1
    
        for xx in range(0,Lh-1):
            dx=abs(X[xx+1]-X[xx]);
            Twx_xt1=Twx_t1[xx];Twy_xt1=Twy_t1[xx];Sdp_xt1=Sdp_t1[xx];
            Twx_x1t1=Twx_t1[xx+1];Twy_x1t1=Twy_t1[xx+1];Sdp_x1t1=Sdp_t1[xx+1];
            
            Twy_xt=Twy_t[xx];Sdp_xt=Sdp_t[xx];
            Twy_x1t=Twy_t1[xx+1];Sdp_x1t=Sdp_t[xx+1];
    
            #Det. longshore velo V
            Dh=0.5*(h[xx+1]+h[xx]);#Depth
            Dsp=0.25*(Sdp_xt1+Sdp_x1t1+Sdp_xt+Sdp_x1t);#Barom tide
            Dtide=0.5*(Tide[tt]+Tide[tt+1]);#Astron. tide
            
            d=Dh+Dtide+Dsp+0.5*(S_t[xx+1]+S_t[xx]);#Total depth for V
            bv=0.25*(Twy_xt+Twy_x1t+Twy_xt1+Twy_x1t1);
            V_t1[xx+1]=(1.0*bv*dt+V_t[xx+1])/(1.0+Cd[xx+1]*abs(V_t[xx+1])*dt/d);
            Lim=sqrt(abs(1.0*Twy_xt1+Twy_x1t1)*1.0*d/(2.0*Cd[xx+1]));
            if abs(V_t1[xx+1])>Lim:
                V_t1[xx+1]=Lim*sign(V_t1[xx+1])
            
            D[xx,tt]=Dh+Tide[tt+1]+0.5*(Sdp_xt1+Sdp_x1t1)+0.5*(S_t[xx+1]+S_t[xx]);
            bl[xx,tt]=Twy_xt;#Total depth for S
            temp=0.5*(Twx_xt1+Twx_x1t1);
            fs=0.5*fc*(V_t1[xx]+V_t1[xx+1]);
            S_t1[xx+1]=S_t1[xx]+1.0*dx/(g*D[xx,tt])*(fs+temp); 

        S_t=S_t1;V_t=V_t1;
        Twy_t=Twy_t1;Sdp_t=Sdp_t1;
    
        S[:,tt+1]=S_t1;V[:,tt+1]=V_t1;

    return S,V,bl
#End of surge

def ForceSeawall(Hwall,T,d):
    #====Calculates pressures acting on seawall=======
    L=T*sqrt(9.81*d);
    
    ro = 1024.0; # density of water (kg/m**3)
    s = pi*Hwall**2*1.0/L/tanh(2*pi*d/L);   # vertical run-up on the wall
    pb = ro*9.81*Hwall/cosh(2*pi*d/L)/1024.0;   # Wave pressure at bottom (KPa)
    pc = (pb+ro*9.81*d)*(Hwall+s)/(d+Hwall+s)/1024.0;  # Wave pressure at crest (KPa)
    pt = ro*9.81*(Hwall-s)/1024;   # Wave pressure at trough (KPa)
    ps = ro*9.81*d/1024.0;   # Hydrostatic pressure at bottom (KPa)
    
    G=300;
    y=[1.0*ii/G*(d+Hwall+s) for ii in range(G)] 
    E =1.0*G*d/(1.0*d+Hwall+s);
    e = int(E);    # determines point of mean water level
    xs=[ro/1024.0*9.81*d-1.0*ii/e*(ro/1024*9.81*d) for ii in range(e)] 
    xw1=[pb+1.0*(pc-pb)/(1.0*e)*ii for ii in range(e)]
    xw2=[pc-(1.0*(ii-e))/(1.0*(G-e))*pc for ii in range(e,G,1)]
    for pp in range(len(xw2)):
        xw1.append(xw2[pp])
    F=num.trapz(xw1,y);
    return F
