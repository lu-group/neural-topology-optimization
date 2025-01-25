#include "udf.h" 
#include "mem.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"

#define C0 0.154
#define C1 1.9
#define C2 2.1
#define C4 0.88
#define rho 998.2
#define D0 0.00001003
#define SC 0.9
#define alpha_max 600000  /*900000*/


DEFINE_DIFFUSIVITY(Diff,c,t,i) 
{
	real De;
	real Dt;
	real kth;
	real epth;
		
	kth=C_UDSI(c,t,1);
	epth=C_UDSI(c,t,2);
	if (kth<=1e-12||epth<=1e-12)
	{
		Dt=C_MU_T(c,t)/SC/rho;
	}
	else
	{
		Dt=C0*C_K(c,t)*sqrt(C_K(c,t)*kth/(C_D(c,t)*epth));
	}
	
	De=(Dt)*rho+D0*C_UDMI(c,t,3);  /* considering the solid fraction in diffusivity calculation*/ 
	C_UDMI(c,t,0)=De;
	return De;
}
 
 DEFINE_SOURCE(ks1,c,t,ds,eqn)
{
	real k1;
	real Dt;
	real kth;
	real epth;
	real k2;
                real k;
		
	kth=C_UDSI(c,t,1);
	epth=C_UDSI(c,t,2);

	if (kth<=0||epth<=0)
	{
		Dt=C_MU_T(c,t)/SC/rho;
	}
	else
	{
		Dt=C0*C_K(c,t)*sqrt(C_K(c,t)*kth/(C_D(c,t)*epth));
	}
	k1=2*rho*Dt*(C_UDSI_G(c,t,0)[0]*C_UDSI_G(c,t,0)[0]+C_UDSI_G(c,t,0)[1]*C_UDSI_G(c,t,0)[1]);
                k2=-2*rho*epth;
                k=k1+k2;
                C_UDMI(c,t,1)=k;
	ds[eqn]=0;
              
return k;
}


 DEFINE_SOURCE(es1,c,t,ds,eqn)
{
	real e1;
	real Dt;
	real kth;
	real epth;
                real e2;
                real e3;
                real e;
		
	kth=C_UDSI(c,t,1);
	epth=C_UDSI(c,t,2);
	
	if (kth<=0||epth<=0)
	{
		Dt=C_MU_T(c,t)/SC/rho;
                                e=0;
	}
	else
	{
	Dt=C0*C_K(c,t)*sqrt(C_K(c,t)*kth/(C_D(c,t)*epth));
                e1=rho*C1*Dt*epth*(C_UDSI_G(c,t,0)[0]*C_UDSI_G(c,t,0)[0]+C_UDSI_G(c,t,0)[1]*C_UDSI_G(c,t,0)[1])/kth;	
                e3=-rho*C4*epth*epth/kth;	
                e2=-rho*C2*C_D(c,t)*epth/C_K(c,t);
                e=e1+e2+e3;	

	}
		
               C_UDMI(c,t,2)=e;  
	ds[eqn]=0;

return e;
}
 

DEFINE_PROFILE(k_inlet,thread,i)
 {
    face_t f;
    begin_f_loop(f,thread)
        F_PROFILE(f,thread,i) = 0.0067*F_UDSI(f,thread,0)*F_UDSI(f,thread,0);
    end_f_loop(f,thread)
 } 


DEFINE_PROFILE(es_inlet,thread,i)
 {
    face_t f;
    begin_f_loop(f,thread)
        F_PROFILE(f,thread,i) = 0.4*F_D(f,thread)*F_UDSI(f,thread,1)/F_K(f,thread);
    end_f_loop(f,thread)
   }


DEFINE_ON_DEMAND(reset_vf)
/*no less than 20% in x[0.006,0.018] was solid, topology structure is randomly generated in coarse mesh */
 { 
      Domain *domain;
      cell_t c;
      Thread *c_thread;

      face_t f;
      Thread *tf;
      cell_t  c0;
      Thread *t0;
      cell_t  c1;
      Thread *t1;

      int i,a,bb,b[20001],mx,my,n,one1,one2,one;
      int m,nsolid,gama,gama0,gama1,gamaf;
      real x[ND_ND],xf[ND_ND],direso;
      domain = Get_Domain(1);
      c_thread= Lookup_Thread(domain, 12);

      srand((unsigned)time(NULL));
      
/*initialize the number of solid cell*/    
      nsolid =  rand() % 3+1; 

/*initial position of solid cells randomly assigned according to their position*/
      for (i = 1; i<= 20000; i++)
	{
		b[i]=1;
	}
      for (i = 1; i<= nsolid; i++)
	{
		bb = rand() % 12001+6000;
		b[bb] = 0;
	}
      begin_c_loop(c, c_thread) /* loops over cells in a cell thread */
        {
         C_CENTROID(x,c,c_thread);
         mx=floor(x[0]/0.0001);
         my=floor(x[1]/0.0001);
         n=mx*100+my+1;
         C_UDMI(c,c_thread,3)=b[n];
        }
      end_c_loop(c, c_thread) 

/*accumulate numbers of solid cells to 20%*/
while(nsolid<1200)
{
/* loops over cells in a cell thread,
 randomly assign the fluid cell adjecent to the solid cell into solid cell */     
       begin_c_loop(c, c_thread) 
        {
          C_CENTROID(x,c,c_thread);
          if(x[0]>0.006,x[0]<0.018)
          {
           if(C_UDMI(c,c_thread,3)==1)/*assign partial gas cell as solid cell*/
          {
              /*Recording the number of solid cells adjecent to the current cell, no changes for gama=0 */
              gama = 0;
              gamaf = 0;              
              direso = 0; 
              /* loops over all faces of a cell */
              c_face_loop(c,c_thread, m)   
              {
                f = C_FACE(c,c_thread,m);
                tf = C_FACE_THREAD(c,c_thread,m);
                /*one is this cell with gama=1 and another is the adjecent cell*/
                c0 = F_C0(f,tf);
                t0 = THREAD_T0(tf);
                if (BOUNDARY_FACE_THREAD_P(tf))
                   {
                    gamaf=C_UDMI(c0,t0,3)+C_UDMI(c0,t0,3); 
                   }
                else
                   {
                    c1 = F_C1(f,tf);
                    t1 = THREAD_T1(tf);                  
                    gamaf=C_UDMI(c0,t0,3)+C_UDMI(c1,t1,3);
                   }
                 /*get the coordinate of faces adjecent to solid cell*/
                 if(gamaf<2)/*existing solid cell near this face */
                 {
                   F_CENTROID(xf,f,tf);
                   gama = gama+1;
                   direso = direso + x[0]-xf[0]; /*equal to 0 if the solid face at the y direction*/
                 }               
               }     /*c_face loop end*/
              if (gama>2)
                 {
                  C_UDMI(c,c_thread,3)=0;            
                 }
              else if (gama>0)/*existing solid cell near this cell */
                 {
                   if (direso=0)
                    {
                     one1 = rand()%2;
                     one2 = rand()%2;
                     one = one1;
                     if(one2<one1)
                     {one=one2;}
                     C_UDMI(c,c_thread,3)=one;
                    }
                   else
                    {
                     one1 = rand()%2;
                     one2 = rand()%2;
                     one = one1;
                     if(one2>one1)
                     {one=one2;}
                     C_UDMI(c,c_thread,3)=one;
                    }                   
                 }
              nsolid = nsolid+1-C_UDMI(c,c_thread,3);
            }
          }
        }
      end_c_loop(c, c_thread)

/* loops over cells in the domain agian,
 assign the fluid cell surrounded by the solid cell to solid cell */   
 for(i=1;i<5;i++)     
  {
       begin_c_loop(c, c_thread) 
        {
        if(C_UDMI(c,c_thread,3)==1)/*assign partial gas cell as solid cell*/
          {
            C_CENTROID(x,c,c_thread);
            if(x[0]>0.006 && x[0]<0.018)
            {
              gama = 0; /*Recording the number of solid cells adjecent to the current cell */
              c_face_loop(c,c_thread, m)   /* loops over all faces of a cell */
              {
                f = C_FACE(c,c_thread,m);
                tf = C_FACE_THREAD(c,c_thread,m);
                /*one is this cell with gama=1 and another is the adjecent cell*/
                c0 = F_C0(f,tf);
                t0 = THREAD_T0(tf);
                if (BOUNDARY_FACE_THREAD_P(tf))
                   {
                    if(x[0]>0.02-0.0001)
                     {
                      gama0=1;
                      gama1=1;
                     }
                    else
                     {
                      gama0=0;
                      gama1=1;        
                     }
                   }
                else
                   {
                    c1 = F_C1(f,tf);
                    t1 = THREAD_T1(tf);  
                    gama0=C_UDMI(c0,t0,3);          
                    gama1=C_UDMI(c1,t1,3);
                   }
                 gama = gama+2-gama0-gama1;
               }
              if (gama>rand()%2+1)
                 {
                  C_UDMI(c,c_thread,3)=0;            
                 }
              nsolid = nsolid+1-C_UDMI(c,c_thread,3);
            }
          }
        }
      end_c_loop(c, c_thread) 
     }

   begin_c_loop(c, c_thread) 
        {
            C_CENTROID(x,c,c_thread);
            if(x[0]<0.006)
            {
            if(C_UDMI(c,c_thread,3)==0)/*assign partial gas cell as solid cell*/
             {
              C_UDMI(c,c_thread,3)=1;
              nsolid = nsolid-1;
             }
          }
        }
      end_c_loop(c, c_thread) 
   }
}

DEFINE_SOURCE(usource,c,t,ds,eqn)
{
    real f;
    real alpha;
    real gama;
       
    gama = C_UDMI(c,t,3);
    /*C_UDMI(c,t,2)=gama;/327.67*/
    alpha = alpha_max*(1-gama)/(1+gama);
    f=-alpha*C_U(c,t);
    C_UDMI(c,t,4)=f;
    ds[eqn]=0;

    return f;
}

 DEFINE_SOURCE(vsource,c,t,ds,eqn)
{
    real f;
    real alpha;
    real gama;
       
    gama = C_UDMI(c,t,3);
    /*C_UDMI(c,t,2)=gama;/327.67*/
    alpha = alpha_max*(1-gama)/(1+gama);
    f=-alpha*C_V(c,t);
    C_UDMI(c,t,5)=f;
    ds[eqn]=0;

    return f;
}
