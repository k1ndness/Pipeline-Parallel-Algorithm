#include <math.h>
__device__
double tanhn(double x)
{
    double a;
    a=tanh(x);
    return a;
}
#include <math.h>
__device__
double ppow(double a , double b)
{
        double x;
        x=pow(a,b);
        return x;
}
#include <math.h>
__device__
double llog2(int a)
{
        int x,b,c;
        b=a;
     for(int i=0;i<10;i++)
     {
        b=b/2;
        if(b!=1)
        {
            x++;
        }
     }
        return x;
}
__device__
double atomicAddd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
__device__
double atomicMaxx(double* address, double val)
{
   unsigned long long int* address_as_i = (unsigned long long int*) address;
   unsigned long long int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __double_as_longlong(fmax(val,
                                __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}
extern "C"
__global__ void non(){
int id = threadIdx.x;
};
__device__
void delay(int a)
{for(int i=0 ; i<a ;i++){printf("");}}
extern "C"
__global__ void test(double *in,double *out,double *w,double *b,int *pix,int *kernel){
int id = threadIdx.x;
int row=threadIdx.x / 5;
int col=threadIdx.x % 5;
int pixr = pix[0]/32;
int pixc = pix[0]%32;
int outkernel=kernel[0]%6;
out[(col+pixc)+(row+pixr)*36]+=in[0]*w[col+row*5];
//printf("id=%d ",id);
//printf("out[%d][%d] pix[%d][%d] w[%d][%d]\n",row+pixr,col+pixc,pixr,pixc,row,col);

};



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////2222222222222222
extern "C"
__global__ void PartialCore(double *in,double *out,double *w,int *pix){
int id =  threadIdx.x;     //deviceinput deviceout2
int pixr = pix[0];
int xx=threadIdx.x + blockIdx.x*blockDim.x;
double m=0;
int n=0;
int z=0;
__syncthreads();
int i=blockIdx.x;
int ker = id  / 25;
int row=  id  % 25 /5;
int col=  id  % 25 %5;
__syncthreads();
__shared__ double Ns[5*5*6];
Ns[col+row*5+ker*5*5]=w[col+row*5+ker*5*5];
__syncthreads();
if(pixr%2==1)
{
    n=6;
}
__syncthreads();
//if(pix[0]<32)
{
out[(col+i+z)+(row)*36+(ker+n)*36*5]=0;
__syncthreads();
atomicAddd(&out[(col+i+z)+(row)*36+(ker+n)*36*5],in[i+z+pixr*32]*Ns[col+row*5+ker*5*5]);
__syncthreads();
}
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////2222222222222222222
extern "C"
__global__ void ACore(double *in,double *b,int *pix,double *tanout
,double *AM1,double *AM2,double *AM3,double *AM4,double *AM5,double *AM6)
{
int id = threadIdx.x;
int ker = blockIdx.x;
int pixr = pix[0];
int pixnum = pixr%6;
int n=0;
if(pixr%2==1)  n=6;

if(id<28)
  {
    if(pixr>4)
    {   int a=0;
        int col = id;
        if(pixnum==0) tanout[(col-a)+(0)*28+ker*28*28]=tanhn(AM2[col+ker*28]+b[ker]);

        if(pixnum==1) tanout[(col-a)+(1)*28+ker*28*28]=tanhn(AM3[col+ker*28]+b[ker]);

        if(pixnum==2) tanout[(col-a)+(0)*28+ker*28*28]=tanhn(AM4[col+ker*28]+b[ker]);

        if(pixnum==3) tanout[(col-a)+(1)*28+ker*28*28]=tanhn(AM5[col+ker*28]+b[ker]);

        if(pixnum==4) tanout[(col-a)+(0)*28+ker*28*28]=tanhn(AM6[col+ker*28]+b[ker]);

        if(pixnum==5) tanout[(col-a)+(1)*28+ker*28*28]=tanhn(AM1[col+ker*28]+b[ker]);
    }
 }


if(id>27)
{
    int a=4;
    int col=threadIdx.x % 28;
    {
           if(pixnum==0)
              {
                  if(id>27&&id<56)
                  AM1[col+ker*28]=in[col+a+4*36+(ker+n)*36*5];
                  if(id>55&&id<84)
                  AM3[col+ker*28]+=in[col+a+0*36+(ker+n)*36*5];
                  if(id>83&&id<112)
                  AM4[col+ker*28]+=in[col+a+1*36+(ker+n)*36*5];
                  if(id>111&&id<140)
                  AM5[col+ker*28]+=in[col+a+2*36+(ker+n)*36*5];
                  if(id>139&&id<168)
                  AM6[col+ker*28]+=in[col+a+3*36+(ker+n)*36*5];
              }
          if(pixnum==1)
              {
                  if(id>27&&id<56)
                  AM1[col+ker*28]+=in[col+a+3*36+(ker+n)*36*5];
                  if(id>55&&id<84)
                  AM2[col+ker*28]=in[col+a+4*36+(ker+n)*36*5];
                  if(id>83&&id<112)
                  AM4[col+ker*28]+=in[col+a+0*36+(ker+n)*36*5];
                  if(id>111&&id<140)
                 AM5[col+ker*28]+=in[col+a+1*36+(ker+n)*36*5];
                  if(id>139&&id<168)
                  AM6[col+ker*28]+=in[col+a+2*36+(ker+n)*36*5];
              }
          if(pixnum==2)
              {
                  if(id>27&&id<56)
                  AM1[col+ker*28]+=in[col+a+2*36+(ker+n)*36*5];
                  if(id>55&&id<84)
                  AM2[col+ker*28]+=in[col+a+3*36+(ker+n)*36*5];
                  if(id>83&&id<112)
                  AM3[col+ker*28]=in[col+a+4*36+(ker+n)*36*5];
                  if(id>111&&id<140)
                  AM5[col+ker*28]+=in[col+a+0*36+(ker+n)*36*5];
                  if(id>139&&id<168)
                  AM6[col+ker*28]+=in[col+a+1*36+(ker+n)*36*5];


              }
          if(pixnum==3)
              {
                  if(id>27&&id<56)
                  AM1[col+ker*28]+=in[col+a+1*36+(ker+n)*36*5];
                  if(id>55&&id<84)
                  AM2[col+ker*28]+=in[col+a+2*36+(ker+n)*36*5];
                  if(id>83&&id<112)
                  AM3[col+ker*28]+=in[col+a+3*36+(ker+n)*36*5];
                  if(id>111&&id<140)
                  AM4[col+ker*28]=in[col+a+4*36+(ker+n)*36*5];
                  if(id>139&&id<168)
                  AM6[col+ker*28]+=in[col+a+0*36+(ker+n)*36*5];
              }

          if(pixnum==4)
              {
                  if(id>27&&id<56)
                  AM1[col+ker*28]+=in[col+a+0*36+(ker+n)*36*5];
                  if(id>55&&id<84)
                  AM2[col+ker*28]+=in[col+a+1*36+(ker+n)*36*5];
                  if(id>83&&id<112)
                  AM3[col+ker*28]+=in[col+a+2*36+(ker+n)*36*5];
                  if(id>111&&id<140)
                  AM4[col+ker*28]+=in[col+a+3*36+(ker+n)*36*5];
                  if(id>139&&id<168)
                  AM5[col+ker*28]=in[col+a+4*36+(ker+n)*36*5];

              }
          if(pixnum==5)
              {

                 if(id>27&&id<56)
                 AM2[col+ker*28]+=in[col+a+0*36+(ker+n)*36*5];
                 if(id>55&&id<84)
                 AM3[col+ker*28]+=in[col+a+1*36+(ker+n)*36*5];
                 if(id>83&&id<112)
                 AM4[col+ker*28]+=in[col+a+2*36+(ker+n)*36*5];
                 if(id>111&&id<140)
                 AM5[col+ker*28]+=in[col+a+3*36+(ker+n)*36*5];
                 if(id>139&&id<168)
                 AM6[col+ker*28]=in[col+a+4*36+(ker+n)*36*5];

              }
    }

}


};


 ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 extern "C"
 __global__ void poolcore(double *pool,double *in,int *pix){

    int id = threadIdx.x;
    int ker = blockIdx.x;
    int col = id % 28;
    int Crow;
    int indexx=1;
    Crow = pix[0]-5;
    if(Crow%2==1) indexx=0;

    if(Crow>=0)
    {
        atomicMaxx(&pool[(col/2)+(Crow/2)*14+ker*14*14],in[col+(indexx)*28+ker*28*28]);
    }
 };












////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
 __global__ void Pool2Core(double *pool,double *in,int *index,int *pix){
    int xx = threadIdx.x + blockIdx.x*blockDim.x;
    int id = threadIdx.x;
    int ker = blockIdx.x;
    int col = threadIdx.x %10;

    int Crow;
    Crow = pix[0]-5;
    int indexxx = index[0];
    int inndex = 1;
    if(Crow%2==1) inndex=0;
    __syncthreads();


        if(Crow>=0)
        {


            atomicMaxx(&pool[(col/2)*16+(Crow/2)*5*16+ker]   ,in[col+(inndex)*10+ker*10*10]);
            __syncthreads();



        }



 };
/////////222222222222222222222222222222222222222222222222222222222
extern "C"
__global__ void Partial2Core(double *in,double *out,double *w,int *pix){
int id = blockIdx.x * blockDim.x + threadIdx.x;    //deviceinputrow2 deviceout2_2
int pixr = pix[0];

int ker = threadIdx.x / 25;
int j =blockIdx.x ;// /6
int row = threadIdx.x %25 / 5;
int col = threadIdx.x %25 % 5;

int n =0;
double ac;
int xx=threadIdx.x + blockIdx.x*blockDim.x;


if(pixr%2==1) n=16;

{


                out[(col+j)+(row)*18+(ker+n)*18*5]=0;
                __syncthreads();


                for(int iiker=0;iiker<6;iiker++)

                {

                    atomicAddd(&out[(col+j)+(row)*18+(ker+n)*18*5] , in[j+pixr*14+iiker*14*14]*w[col+row*5+iiker*5*5+ker*5*5*6]);
                    __syncthreads();
                }


}
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
__global__ void A2Core(double *in,double *b,int *pix,double *tanout,double *AM1,double *AM2,double *AM3,
double *AM4,double *AM5,double *AM6)//
{
int xx = threadIdx.x + blockIdx.x*blockDim.x;
int id = threadIdx.x;
int ker = blockIdx.x;
int n =0;
int z =0;


int pixr = pix[0];
int pixnum = pixr%6;
if(pix[0]%2==1)n=16;
{

    if(id<10)
    {
        int col=threadIdx.x;
        int a=0;
        if(pixr>4)
            {
                    if(pixnum==0)
                        {

                          tanout[(col-a)+(0)*10+ker*10*10]=tanhn(AM2[col+ker*10]+b[ker]);


                        }
                    if(pixnum==1)
                        {

                             tanout[(col-a)+(1)*10+ker*10*10]=tanhn(AM3[col+ker*10]+b[ker]);



                        }
                    if(pixnum==2)
                        {

                            tanout[(col-a)+(0)*10+ker*10*10]=tanhn(AM4[col+ker*10]+b[ker]);

                        }
                    if(pixnum==3)
                        {

                            tanout[(col-a)+(1)*10+ker*10*10]=tanhn(AM5[col+ker*10]+b[ker]);



                        }

                    if(pixnum==4)
                        {

                            tanout[(col-a)+(0)*10+ker*10*10]=tanhn(AM6[col+ker*10]+b[ker]);


                        }
                    if(pixnum==5)
                        {

                           tanout[(col-a)+(1)*10+ker*10*10]=tanhn(AM1[col+ker*10]+b[ker]);


                        }

            }
    }
}


{
    if(id>9)
    {
        int col= threadIdx.x % 10;
        int a=4;


        {
            if(pixnum==0)
            {
		            if(id>9&&id<20)
            AM1[col+ker*10]=in[col+a+4*18+(ker+n)*18*5];    //+ker*18*10     +ker*18*5
                  if(id>19&&id<30)
            AM3[col+ker*10]+=in[col+a+0*18+(ker+n)*18*5];
                  if(id>29&&id<40)
            AM4[col+ker*10]+=in[col+a+1*18+(ker+n)*18*5];
                  if(id>39&&id<50)
            AM5[col+ker*10]+=in[col+a+2*18+(ker+n)*18*5];
                  if(id>49&&id<60)
            AM6[col+ker*10]+=in[col+a+3*18+(ker+n)*18*5];
            //__syncthreads();

           // AM2[col+ker*10]=0;



            }
        if(pixnum==1)
            {

		  if(id>9&&id<20)
            AM1[col+ker*10]+=in[col+a+3*18+(ker+n)*18*5];
                  if(id>19&&id<30)
            AM2[col+ker*10]=in[col+a+4*18+(ker+n)*18*5];
                  if(id>29&&id<40)
            AM4[col+ker*10]+=in[col+a+0*18+(ker+n)*18*5];
                  if(id>39&&id<50)
            AM5[col+ker*10]+=in[col+a+1*18+(ker+n)*18*5];
                  if(id>49&&id<60)
            AM6[col+ker*10]+=in[col+a+2*18+(ker+n)*18*5];


            }
        if(pixnum==2)
            {
		  if(id>9&&id<20)
            AM1[col+ker*10]+=in[col+a+2*18+(ker+n)*18*5];
                  if(id>19&&id<30)
            AM2[col+ker*10]+=in[col+a+3*18+(ker+n)*18*5];
                  if(id>29&&id<40)
            AM3[col+ker*10]=in[col+a+4*18+(ker+n)*18*5];
                  if(id>39&&id<50)
            AM5[col+ker*10]+=in[col+a+0*18+(ker+n)*18*5];
                  if(id>49&&id<60)
            AM6[col+ker*10]+=in[col+a+1*18+(ker+n)*18*5];




            }
        if(pixnum==3)
            {
		  if(id>9&&id<20)
            AM1[col+ker*10]+=in[col+a+1*18+(ker+n)*18*5];
                  if(id>19&&id<30)
            AM2[col+ker*10]+=in[col+a+2*18+(ker+n)*18*5];
                  if(id>29&&id<40)
            AM3[col+ker*10]+=in[col+a+3*18+(ker+n)*18*5];
                  if(id>39&&id<50)
            AM4[col+ker*10]=in[col+a+4*18+(ker+n)*18*5];
                  if(id>49&&id<60)
            AM6[col+ker*10]+=in[col+a+0*18+(ker+n)*18*5];




            }
            ///////////////////////////////////////////////////////////////////
        if(pixnum==4)
            {

		  if(id>9&&id<20)
            AM1[col+ker*10]+=in[col+a+0*18+(ker+n)*18*5];
                  if(id>19&&id<30)
            AM2[col+ker*10]+=in[col+a+1*18+(ker+n)*18*5];
                  if(id>29&&id<40)
            AM3[col+ker*10]+=in[col+a+2*18+(ker+n)*18*5];
                  if(id>39&&id<50)
            AM4[col+ker*10]+=in[col+a+3*18+(ker+n)*18*5];
                  if(id>49&&id<60)
            AM5[col+ker*10]=in[col+a+4*18+(ker+n)*18*5];







            }
        if(pixnum==5)
            {
          if(id>9&&id<20)
           AM2[col+ker*10]+=in[col+a+0*18+(ker+n)*18*5];
                  if(id>19&&id<30)
           AM3[col+ker*10]+=in[col+a+1*18+(ker+n)*18*5];
                  if(id>29&&id<40)
           AM4[col+ker*10]+=in[col+a+2*18+(ker+n)*18*5];
                  if(id>39&&id<50)
           AM5[col+ker*10]+=in[col+a+3*18+(ker+n)*18*5];
                  if(id>49&&id<60)
           AM6[col+ker*10]=in[col+a+4*18+(ker+n)*18*5];





            }
            //__syncthreads();
        }

    }
}

};

/////////////////////////////////////////////////////////////////////////////
extern "C"
__global__ void FC(int *par,double *w, double *s, double *in,double *b){
int inSize=par[0],outSize=par[1];
int id = threadIdx.x;
int row = id / outSize;
int col = id % outSize;
double tol;
if (id <outSize)
{
  tol=0;
  for(int j=0; j<inSize; j++)

  {
   tol+=in[j]*w[id+j*outSize];

  }
    __syncthreads();


    s[id]=tanh(tol+b[id]);
    __syncthreads();
}
};

extern "C"
__global__ void FCL3(int *par,double *w, double *s, double *in,double *b){
int inSize=par[0],outSize=par[1];
double E=2.7182818284590452354;
__shared__ double exp[10];
int id = threadIdx.x + blockIdx.x*blockDim.x;
int row = id / outSize;
int col = id % outSize;
double tol;
if (id <outSize)
{
  exp[id]=0;
  tol=0;
  for(int j=0; j<inSize; j++)
  {
   tol+=in[j]*w[id+j*outSize];
  }
 __syncthreads();
 tol=tol+b[id];
 exp[id]+=ppow(E,tol);
 exp[0]=exp[0]+exp[1]+exp[2]+exp[3]+exp[4]+exp[5]+exp[6]+exp[7]+exp[8]+exp[9];
 __syncthreads();

 s[id]=ppow(E,tol)/exp[0];

}
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

