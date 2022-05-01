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
extern "C"
__global__ void add(int *n, float *a, float *b, float *sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n[0])
    {
        sum[i] = a[i] + b[i];
    }

};
const int XXX = 64*64;
extern "C"
__global__ void PARSAD(int *par,double *s, double *in, double*w, double*b){
 int width=par[0],height=par[1],channeltime=par[2],ichan=par[3],kernelSize=par[4];//32,32,6,1,5
 int id = threadIdx.x + blockIdx.x*blockDim.x;                                    //14,14,16,6,5

 if (id <width*height*channeltime)
 {
 int channel = id/(width*height);
 int row = (id %(width*height))/ width;
 int col = (id %(width*height))% width;
 int x=col-4,y=row-4;
 int z=width-kernelSize+1;
 int zh=height-kernelSize+1;
//__shared__ double Ns[XXX];
//Ns[col+row*width]=in[col+row*width];
//__syncthreads();
   for (int ch = 0; ch < ichan; ch++) {
     for (int i = 0; i < kernelSize; i++)
     {
      for (int j = 0; j < kernelSize; j++)
      {
          if(row+i>3&&col+j>3&&row+i<height&&col+j<width)
          {
          s[(x+j)+(y+ i)*z+channel*z*zh]+=in[col+row*width+ch*width*width]
          *w[j+i*kernelSize+ch*kernelSize*kernelSize+channel*ichan*kernelSize*kernelSize];
          }
          __syncthreads();

      }
     }
       if(row>3&&col>3&&ch==ichan-1)
       s[(x)+(y)*z+channel*z*zh]=tanhn(s[(x)+(y)*z+channel*z*zh]+b[channel]);
   }
 }
};
extern "C"
__global__ void Conv1(int *par,double *s, double *in, double*w, double*b){

 int id = blockIdx.x * blockDim.x + threadIdx.x;
int insizeW=par[0],insizeH=par[1],outchannel=par[2],inchannel=par[3],kernelsize=par[4],stride=1;
  int outsizeW=insizeW-kernelsize+1;
  int outsizeH=insizeH-kernelsize+1;
    int channel, row, col;
    int ch, i, j;

    if (id < outsizeW*outsizeH*outchannel) {
        channel = id / (outsizeW * outsizeH);
        row = (id % (outsizeW * outsizeH)) / outsizeW;
        col = (id % (outsizeW * outsizeH)) % outsizeW;

        s[channel*outsizeW*outsizeH+row*outsizeW+col] = 0.0;
        for (ch = 0; ch < inchannel; ch++) {
        for (i = 0; i < kernelsize; i++) {
            for (j = 0; j < kernelsize; j++) {
                   s[channel*outsizeW*outsizeH+row*outsizeW+col] +=w[channel*inchannel*kernelsize*kernelsize+ch*kernelsize*kernelsize+i*kernelsize+j]
                   *in[ch*insizeW*insizeH+(i+row)*insizeW+(j+col)];
                    __syncthreads();
                }
            }
        }
          s[channel*outsizeW*outsizeH+row*outsizeW+col] = tanhn(s[channel*outsizeW*outsizeH+row*outsizeW+col]+b[channel]);
    }

};
extern "C"
__global__ void PARSADAC(int *par,double *s, double*b){
 int width=par[0]-4,height=par[1]-4,channeltime=par[2],ichan=par[3],kernelSize=par[4];//32,32,6,1,5
 int id = threadIdx.x + blockIdx.x*blockDim.x;                                    //14,14,16,6,5

 if (id <width*height*channeltime)
 {

 int channel = id/(width*height);
 int row = (id %(width*height))/ width;
 int col = (id %(width*height))% width;
 int x=col-4,y=row-4;
 int z=width-kernelSize+1;
 int zh=height-kernelSize+1;

       s[(x)+(y)*z+channel*z*zh]=tanhn(s[(x)+(y)*z+channel*z*zh]+b[channel]);

 }
};










extern "C"
__global__ void pool1(int *par,double *pool,double *s){
       int id =threadIdx.x + blockIdx.x*blockDim.x;    //14,14,6,28
       int width=par[0],height=par[1],channel,channeltime=par[2],z=par[3];
       int row,col;
       double max,tmp;
       if (id <width * height*channeltime)
        {
        channel = id/(width*height);
        row = (id %(width*height))/ width;
        col = (id %(width*height))% width;
        max=-256.0;
           for(int k = 0 ; k < 2 ; k++)
           {
              for (int l = 0 ; l < 2 ; l++)
              {
                tmp=s[(col*2+l)+(row*2*z+k*z)+channel*z*z];
                if(max < tmp)
                {
                max=tmp;
                }
              }
              if(width==14)
              pool[col+row*width+channel*width*height]=max;
              if(width==5)
              pool[col*16+row*5*16+channel]=max;
           }

        }
};
extern "C"
__global__ void PARSAD2(int *par,double *s, double *in, double*w, double*b){
 int width=14,height=14,channeltime=16,ichan=6,kernelSize=5;//32,32,6,1,5
 int id = threadIdx.x + blockIdx.x*blockDim.x;                                    //14,14,16,6,5

 if (id <width*height*channeltime)
 {

 int channel = id/(width*height);
 int row = (id %(width*height))/ width;
 int col = (id %(width*height))% width;
 int z=width-kernelSize+1;
   for (int ch = 0; ch < ichan; ch++) {
     for (int i = 0; i < kernelSize; i++)
     {
      for (int j = 0; j < kernelSize; j++)
      {
          if(row+i>3&&col+j>3&&row+i<width&&col+j<width)
          s[(col+j-4)+(row+ i-4)*z+channel*z*z]+=in[col+row*width+ch*width*width]*
          w[j+i*kernelSize+ch*kernelSize*kernelSize+channel*ichan*kernelSize*kernelSize];
          __syncthreads();
      }
     }
      if(row>3&&col>3&&ch==ichan-1)
      s[(col-4)+(row-4)*z+channel*z*z]=tanhn(s[(col-4)+(row-4)*z+channel*z*z]+b[channel]);

   }


 }



};
extern "C"
__global__ void FC(int *par,double *w, double *s, double *in,double *b){
int inSize=par[0],outSize=par[1];
int id = threadIdx.x + blockIdx.x*blockDim.x;
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










