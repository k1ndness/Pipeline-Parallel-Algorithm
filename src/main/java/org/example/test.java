package org.example;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import java.io.IOException;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

public class test {
   static void ttt(double i[],double w[])
    {
        double s[]=new double[6*36*36];
        for(int kernel=0; kernel<6; kernel++){
            for(int row=0; row<32; row++){
                for(int column=0; column<32; column++){
                    for(int  t1 = 0; t1 < 5; t1++)
                    {
                        for (int t2 = 0; t2 < 5; t2++)
                        {
                            {
                                s[(column + t2) + (row+ t1)* 32 + kernel * 32 * 32] = s[(column + t2) + (row+ t1 )* 32 + kernel * 32 * 32] + i[column + row * 32] * w[t2 + (t1 * 5) + kernel * 5 * 5];
                            }
                        }
                    }

                }
            }
        }
    }
    void GPU_Multiplication(int k,int r,int c,double[][] i ,double[][][] w)
    {
        int numkernel=k,numrow=r,numcolumn=c;
        int wrow=5,wcolumn=5;
        int kernel,row,column;
        double A[][] =i; //new double[][]{{1,2,3,4,5},{2,3,4,5,6},{3,4,5,6,7},{4,5,6,7,8},{5,6,7,8,9}};
        double B[][][]=w;// new double[][][]{{{1,1,1,1,1},{2,2,2,2,2},{3,3,3,3,3},{4,4,4,4,4},{5,5,5,5,5}},{{9,9,9,9,9},{8,8,8,8,8},{7,7,7,7,7},{7,7,7,7,7},{6,6,6,6,6}}};/** 宣告HOST要做運算的陣列 **/
        double deviceOutput[] = new double[5];       /** 宣告一個陣列作為Device輸出 **/
        double hostinput[]= new double[numrow*numcolumn];
        double hostweight[]= new double[numkernel*numrow*numcolumn];
        double C []=new double [6*36*36];
        for(kernel=0;kernel < numkernel; kernel++)
        {
            for (row = 0; row < numrow; row++)
            {
                for (column = 0; column < numcolumn; column++)
                {
                    hostinput[column + row * numcolumn] = A[row][column];
                    if(row<5&&column<5)
                    hostweight[column+row*w[0][0].length+kernel*w[0][0].length*w[0].length] = B[kernel][row][column];
                }
            }
        }
        for(kernel=0;kernel < numkernel; kernel++)
        {
            for (row = 0; row < numrow; row++)
            {
                for (column = 0; column < numcolumn; column++)
                {
                    for ( int t1 = 0; t1 < 5; t1++) { /**每個pixel要和卷積核的每個元素相乘**/
                        for (int t2 = 0; t2 < 5; t2++) {
                            C[(column+t2) + (row * numcolumn)+t1 + kernel * numcolumn * numrow] =C[(column+t2) + (row * numcolumn)+t1 + kernel * numcolumn * numrow]+ hostinput[column + row * numcolumn] * hostweight[t2 + (t1 * wcolumn) + kernel * wcolumn * wrow];
                        }
                    }
                    if(row>=4&&column>=4) {
                        if (Double.isNaN(LeNet.tanh(C[column + row * numcolumn + kernel * numcolumn * numrow] + Layer.bias_1[kernel]))) {
                            C[column + row * numcolumn + kernel * numcolumn * numrow] = 1;
                        }
                         else{
                            C[column + row * numcolumn + kernel * numcolumn * numrow] = LeNet.tanh(C[column + row * numcolumn + kernel * numcolumn * numrow] + Layer.bias_1[kernel]);
                        }
                    }

                }
            }
        }
        int xx=0;
        for(kernel=0;kernel < 6; kernel++) {
            for (row = 0; row < 28; row++) {
                for (column = 0; column < 28; column++) {
                    if(LeNet.con1out[kernel][row][column]-C[(column+4) + (row * numcolumn)+4 + kernel * numcolumn * numrow]>0.000001){
                       xx++;
                       System.out.println(kernel+" "+row+" "+column+" XXXXXXXXXX "+C[(column+4) + (row * numcolumn)+4 + kernel * numcolumn * numrow]);
                    }
                    //System.out.println("kernel=" + kernel + " row=" + row + " column=" + column + " C=" + C[column + row * numcolumn + kernel * numcolumn * numrow]);
                   // if (column == 35) System.out.println();
                }
            }
        }
        if(xx==0) System.out.println("cpu con == gpu vlsi1");

    }
    public static void main(String[] args) throws IOException{
        test a= new test();
        //a.GPU_Multiplication(6,32,32,Inputimage.one,Layer.weight_1);
        test.ttt(LeNet.input,LeNet.weight);
    }

    void GPU(){
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);
        // Create the PTX file by calling the NVCC
        String ptxFileName = JCudaSamplesUtils.preparePtxFile("src/main/resources/hello.cu");

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Load the ptx file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain a function pointer to the "add" function.
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "hello");


        char device1[] = new char[50];
        char hostOutput[] = new char[50];
        CUdeviceptr deviceout = new CUdeviceptr();
        cuMemAlloc(deviceout, 50 * Sizeof.CHAR);
        cuMemcpyHtoD(deviceout, Pointer.to(device1),
                50 * Sizeof.CHAR);
      /* CUdeviceptr deviceInputB = new CUdeviceptr();
        cuMemAlloc(deviceInputB, 100 * Sizeof.DOUBLE);
        cuMemcpyHtoD(deviceInputB, Pointer.to(hostInputB),
                100 * Sizeof.DOUBLE);
        int g=3,b=4;
        Pointer kernelParameters = Pointer.to(
                Pointer.to(deviceInputA),
                Pointer.to(deviceInputB)

        );*/
        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) 1024 / blockSizeX);
        Pointer kernelParameters = Pointer.to(
                Pointer.to(deviceout));
        cuLaunchKernel(function,
                gridSizeX, 1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceout,
                50 * Sizeof.CHAR);
        for (int i = 0; i < 50; i++) {
            System.out.print(hostOutput[i]);
        }
        cuMemFree(deviceout);
    }
}