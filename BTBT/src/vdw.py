import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils import Rod

vdw_path='/Users/jigenji/Working/interaction/BTBT/vdw/'
A1_list=[round(A1) for A1 in np.linspace(0,45,46)]
A2_list=[round(A2) for A2 in np.linspace(0,45,46)]
A3_list=[round(A3) for A3 in np.linspace(0,45,46)]
theta_list=np.linspace(0,90,91)

def convertor(atom_list,A1,A2,A3):
    n=np.array([np.sin(np.radians(A1))*np.cos(np.radians(A2)),
                     np.sin(np.radians(A1))*np.sin(np.radians(A2)),
                     np.cos(np.radians(A1))])
    atom_list_rt=[]
    for x,y,z,R in atom_list:
        x1,y1,z1=np.matmul(Rod(np.array([0,1,0]),A1),np.array([x,y,z]).T)
        x2,y2,z2=np.matmul(Rod(np.array([0,0,1]),A2),np.array([x1,y1,z1]).T)
        x3,y3,z3=np.matmul(Rod(n,A3),np.array([x2,y2,z2]).T)
        atom_list_rt.append([x3,y3,z3,R])
    return np.array(atom_list_rt)

def get_c_vec_vdw(A1,A2,A3,a_,b_,glide_mode='a'):#,name_csv
    
    assert glide_mode=='a' or glide_mode=='b'
    
    a=np.array([a_,0,0]);b=np.array([0,b_,0]);t1=(a+b)/2;t2=(a-b)/2;glide = 180.0 if glide_mode=='a' else 0.0
    df_anth=pd.read_csv('assets/monomer.csv')###x,y,z,rad
    anth=df_anth[['X','Y','Z','R']].values
    anth_i0=convertor(anth,A1,A2,A3)#層間
    anth_p=convertor(anth,A1,A2,A3)#層内
    anth_t=convertor(anth,A1,-A2,-A3+glide)#層内
    arr_list=[[np.zeros(3),'p'],[b,'p'],[-b,'p'],[a,'p'],[-a,'p'],[t1,'t'],[-t1,'t'],[t2,'t'],[-t2,'t']]
    Ra_list=[np.round(Ra,1) for Ra in np.linspace(-np.round(a_/2,1),np.round(a_/2,1),int(np.round(2*np.round(a_/2,1)/0.1))+1)]
    Rb_list=[np.round(Rb,1) for Rb in np.linspace(-np.round(b_/2,1),np.round(b_/2,1),int(np.round(2*np.round(b_/2,1)/0.1))+1)]
    z_list=[];V_list=[]
    N_step = len(Ra_list) if glide_mode=='a' else len(Rb_list)
    for step in range(N_step):######## should be renamed
        if glide_mode=='a':
            Ra = Ra_list[step]
        else:
            Rb = Rb_list[step]
        z_max=0
        for R,arr in arr_list:
            if arr=='t':
                anth=anth_t
            elif arr=='p':
                anth=anth_p
            for x1,y1,z1,R1 in anth:#層内
                x1,y1,z1=np.array([x1,y1,z1])+R
                for x2,y2,z2,R2 in anth_i0:#i0
                    if glide_mode=='a':
                        x2+=Ra
                    else:
                        x1+=Rb
                    z_sq=(R1+R2)**2-(x1-x2)**2-(y1-y2)**2
                    if z_sq<0:
                        z_clps=0.0
                    else:
                        z_clps=np.sqrt(z_sq)+z1-z2
                    z_max=max(z_max,z_clps)
        z_list.append(z_max)
        V_list.append(a_*b_*z_max)
    if glide_mode=='a':
        return np.array([Ra_list[np.argmin(V_list)],0,z_list[np.argmin(V_list)]])
    else:
        return np.array([0,Rb_list[np.argmin(V_list)],z_list[np.argmin(V_list)]])
    
# theta=arctan(b/a)
def vdw_R(A1,A2,A3,theta,dimer_mode,glide_mode='b'):
    df_monomer=pd.read_csv('../assets/monomer.csv')###x,y,z,rad
    monomer=df_monomer[['X','Y','Z','R']].values
    monomer_1=convertor(monomer,A1,A2,A3)
    glide=180.0 if glide_mode=='a' else 0.0
    if dimer_mode=='t':
        monomer_2=convertor(monomer,A1,-A2,-A3+glide)
    elif dimer_mode=='a' or dimer_mode=='b':
        monomer_2=convertor(monomer,A1,A2,A3)
    R_clps=0
    for x1,y1,z1,rad1 in monomer_1:
        for x2,y2,z2,rad2 in monomer_2:
            eR=np.array([np.cos(np.radians(theta)),np.sin(np.radians(theta)),0.0])
            R_1b=np.dot(eR,np.array([x1,y1,z1]))
            R_2b=np.dot(eR,np.array([x2,y2,z2]))
            R_12=np.array([x2-x1,y2-y1,z2-z1])
            R_12b=np.dot(eR,R_12)
            R_12a=np.linalg.norm(R_12-R_12b*eR)
            if (rad1+rad2)**2-R_12a**2<0:
                continue
            else:
                R_clps=max(R_clps,R_1b-R_2b+np.sqrt((rad1+rad2)**2-R_12a**2))
    return R_clps

def make_csv(name_csv,glide_mode):
    df_vdw=pd.DataFrame(columns=['A1','A2','A3','theta','R','S','a','b'])
    for A1 in tqdm(A1_list):
        for A2 in A2_list:
            for A3 in A3_list:
                a_clps=vdw_R(A1,A2,A3,0.0,'a','b')
                b_clps=vdw_R(A1,A2,A3,90.0,'b','b')
                for theta in theta_list:
                    R_clps=vdw_R(A1,A2,A3,theta,'t',glide_mode)
                    a=2*R_clps*np.cos(np.radians(theta))
                    b=2*R_clps*np.sin(np.radians(theta))
                    if (a_clps > a) or (b_clps > b):
                        continue
                    else:
                        S=a*b
                        data=pd.Series([A1,A2,A3,theta,R_clps,S,a,b],index=df_vdw.columns)
                        df_vdw=df_vdw.append(data,ignore_index=True)

    df_vdw.to_csv(os.path.join(vdw_path,name_csv))

def make_csv_vdwmin(in_csv,out_csv,edge_mode):
        df_vdw=pd.read_csv(vdw_path+in_csv)
        df_contact=pd.DataFrame(columns=df_vdw.columns)
#         idx_edge=0 if edge_mode=='a' else -1 
        for A1 in A1_list:
            for A2 in A2_list:
                for A3 in A3_list:
                    df_A=df_vdw[(df_vdw['A1']==A1)&(df_vdw['A2']==A2)&(df_vdw['A3']==A3)]
                    data=pd.Series(df_A.iloc[df_A['S'].idxmin()],index=df_contact.columns)
                    df_contact=df_contact.append(data,ignore_index=True)
        df_contact.to_csv(os.path.join(vdw_path,out_csv),index=False)
