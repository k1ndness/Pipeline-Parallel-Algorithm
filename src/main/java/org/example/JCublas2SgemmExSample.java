package org.example;

import static jcuda.cudaDataType.CUDA_R_32F;
import static jcuda.jcublas.JCublas.cublasDgemm;
import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_ALGO0;
import static jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_ALGO2;
import static jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_ALGO4;
import static jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_ALGO5;
import static jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_ALGO6;
import static jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_ALGO7;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
//import jcuda.samples.utils.JCudaSamplesUtils;

/**
 * This is a sample class demonstrating the application of JCublas2 for
 * performing a BLAS 'sgemm' operation, i.e. for computing the matrix <br>
 * <code>C = alpha * A * B + beta * C</code> <br>
 * for single-precision floating point values alpha and beta, and matrices
 * A, B and C, using the extended CUBLAS GEMM function
 */
public class JCublas2SgemmExSample
{
    static void testSgemmBatched(double a,double input[][],int b, int n)
    {
        System.out.println("Testing Sgemm with " + b + " batches of size " + n);

        double alpha = a;
        double beta = 1;
        int nn = n;
        // int nn = n * n;

        double h_A[][] = new  double[][]{};
        h_A=input;
        //double h_B[][] = new  double[][]{{1,0,0,0,0},{0,1,0,0,0},{0,0,1,0,0},{0,0,0,1,0},{0,0,0,0,1}};
        double h_B[][] = new  double[][]{{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}};
        double h_C[][] = new  double[][]{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}};
        System.out.println("Performing Sgemm with JCublas2...");
        sgemmBatchedJCublas2(n, alpha, h_A, h_B, beta, h_C);
        System.out.println("InputImage="+alpha);
        // Print the test results
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                System.out.println(h_C[i][j]);
            }
        }

        /**

         **/
    }
    static void sgemmBatchedJCublas2(int n,  double alpha,
                                     double h_A[][],  double h_B[][],  double beta,  double h_C[][])
    {
        int nn = n;
        int b = h_A.length;
        Pointer[] h_Aarray = new Pointer[b];
        Pointer[] h_Barray = new Pointer[b];
        Pointer[] h_Carray = new Pointer[b];
        for (int i = 0; i < b; i++)
        {
            h_Aarray[i] = new Pointer();
            h_Barray[i] = new Pointer();
            h_Carray[i] = new Pointer();
            cudaMalloc(h_Aarray[i], nn * Sizeof.DOUBLE);
            cudaMalloc(h_Barray[i], nn * Sizeof.DOUBLE);
            cudaMalloc(h_Carray[i], nn * Sizeof.DOUBLE);
            cudaMemcpy(h_Aarray[i], Pointer.to(h_A[i]),
                    nn * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
            cudaMemcpy(h_Barray[i], Pointer.to(h_B[i]),
                    nn * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
            cudaMemcpy(h_Carray[i], Pointer.to(h_C[i]),
                    nn * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
        }
        Pointer d_Aarray = new Pointer();
        Pointer d_Barray = new Pointer();
        Pointer d_Carray = new Pointer();
        cudaMalloc(d_Aarray, b * Sizeof.POINTER);
        cudaMalloc(d_Barray, b * Sizeof.POINTER);
        cudaMalloc(d_Carray, b * Sizeof.POINTER);

        cudaMemcpy(d_Aarray, Pointer.to(h_Aarray),
                b * Sizeof.POINTER, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Barray, Pointer.to(h_Barray),
                b * Sizeof.POINTER, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Carray, Pointer.to(h_Carray),
                b * Sizeof.POINTER, cudaMemcpyHostToDevice);

        cublasHandle handle = new cublasHandle();
        cublasCreate(handle);

        cublasDgemmBatched(
                handle,
                cublasOperation.CUBLAS_OP_N,
                cublasOperation.CUBLAS_OP_N,
                n, n, n,
                Pointer.to(new double[]{ alpha }),
                d_Aarray, n, d_Barray, n,
                Pointer.to(new double[]{ beta }),
                d_Carray, n, b);

        for (int i = 0; i < b; i++)
        {
            cudaMemcpy(Pointer.to(h_C[i]), h_Carray[i],
                    nn * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
            cudaFree(h_Aarray[i]);
            cudaFree(h_Barray[i]);
            cudaFree(h_Carray[i]);
        }
        cudaFree(d_Aarray);
        cudaFree(d_Barray);
        cudaFree(d_Carray);
        cublasDestroy(handle);

    }

    static void test(double a ){

        double b=1;
        double h_A[][]=new double[][]{{1,1,1},{1,1,1},{1,1,1}};
        double h_B[][]=new double[][]{{1,0,0},{0,1,0},{0,0,1}};
        double h_C[][]=new double[][]{{1,0,0},{0,1,0},{0,0,1}};
        cublasHandle handle = new cublasHandle();
        cublasCreate(handle);

        Pointer[] h_Aarray = new Pointer[3];
        Pointer[] h_Barray = new Pointer[3];
        Pointer[] h_Carray = new Pointer[3];



        for (int i = 0; i < 3; i++)
        {
            h_Aarray[i] = new Pointer();
            h_Barray[i] = new Pointer(); 
            h_Carray[i] = new Pointer();
            cudaMalloc(h_Aarray[i], 3 * Sizeof.DOUBLE);
            cudaMalloc(h_Barray[i], 3 * Sizeof.DOUBLE);
            cudaMalloc(h_Carray[i], 3 * Sizeof.DOUBLE);
            // Copy the memory from the host to the device
            cudaMemcpy(h_Aarray[i],Pointer.to(h_A[i]), 3*Sizeof.DOUBLE,cudaMemcpyHostToDevice);
            cudaMemcpy(h_Barray[i],Pointer.to(h_B[i]), 3*Sizeof.DOUBLE,cudaMemcpyHostToDevice);
            cudaMemcpy(h_Carray[i],Pointer.to(h_C[i]), 3*Sizeof.DOUBLE,cudaMemcpyHostToDevice);

        }
        // Execute sgemm
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        Pointer pAlpha = Pointer.to(new double[] {a});
        Pointer pBeta = Pointer.to(new double[] { b });
        cudaMalloc(d_A, 3 * Sizeof.POINTER);
        cudaMalloc(d_B, 3 * Sizeof.POINTER);
        cudaMalloc(d_C, 3 * Sizeof.POINTER);
        cudaMemcpy(d_A,Pointer.to(h_Aarray), 3*Sizeof.DOUBLE,cudaMemcpyHostToDevice);
        cudaMemcpy(d_B,Pointer.to(h_Barray), 3*Sizeof.DOUBLE,cudaMemcpyHostToDevice);
        cudaMemcpy(d_C,Pointer.to(h_Carray), 3*Sizeof.DOUBLE,cudaMemcpyHostToDevice);



        JCublas2.cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,3,3,3,pAlpha,d_A,3,d_B,3,pBeta,d_C,3);

        for (int i = 0; i < 3; i++)
        {
            cudaMemcpy(Pointer.to(h_C[i]), h_Carray[i],
                    3 * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
            cudaFree(h_Aarray[i]);
            cudaFree(h_Barray[i]);
            cudaFree(h_Carray[i]);
        }
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                System.out.println(h_C[i][j]);
                //System.out.println(h_C[1]);
                // System.out.println(h_C[2]);
            }
        }
    }
    public static void main(String args[])
    {
        JCublas2.setExceptionsEnabled(true);
        //testSgemm(2000);
        test(2);
    }

    // The list of CUBLAS GEMM algorithms to use. Note that the set of
    // supported algorithms will likely depend on the platform, the
    // size of the matrix, and other factors.
    //使用的CUBLAS GEMM算法列表請注意，支持的算法集可能取決於平台，矩陣的大小和其他因素。
    private static final List<Integer> GEMM_ALGORITHMS = Arrays.asList(
            CUBLAS_GEMM_ALGO2,
            CUBLAS_GEMM_ALGO4,
            CUBLAS_GEMM_ALGO5,
            CUBLAS_GEMM_ALGO6,
            CUBLAS_GEMM_ALGO7
    );
    private static int GEMM_ALGO = CUBLAS_GEMM_ALGO0;
    public static float[] createRandomFloatData(int n)
    {
        Random random = new Random(0);
        float a[] = new float[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = random.nextFloat();
        }
        return a;
    }
    /**
     * Test the JCublas sgemm operation for matrices of size n x x
     *
     * @param n The matrix size
     */
    public static void testSgemm(int n)
    {
        float alpha = 0.3f;
        float beta = 0.7f;
        int nn = n * n;

        System.out.println("Creating input data...");
        float h_A[] = createRandomFloatData(nn);
        float h_B[] = createRandomFloatData(nn);
        float h_C[] = createRandomFloatData(nn);

        System.out.println("Performing Sgemm with JCublas...");
        for (int i : GEMM_ALGORITHMS)
        {
            GEMM_ALGO = i;
            try
            {
                sgemmJCublas(n, alpha, h_A, h_B, beta, h_C);
            }
            catch (Exception e)
            {
                e.printStackTrace();
            }
        }

    }

    /**
     * Implementation of sgemm using JCublas
     */
    private static void sgemmJCublas(
            int n, float alpha, float A[], float B[], float beta, float C[])
    {
        int nn = n * n;

        // Create a CUBLAS handle
        cublasHandle handle = new cublasHandle();
        cublasCreate(handle);

        // Allocate memory on the device
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        cudaMalloc(d_A, nn * Sizeof.FLOAT);
        cudaMalloc(d_B, nn * Sizeof.FLOAT);
        cudaMalloc(d_C, nn * Sizeof.FLOAT);

        // Copy the memory from the host to the device
        cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(A), 1, d_A, 1);
        cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(B), 1, d_B, 1);
        cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(C), 1, d_C, 1);

        // Execute sgemm
        Pointer pAlpha = Pointer.to(new float[] { alpha });
        Pointer pBeta = Pointer.to(new float[] { beta });

        long before = System.nanoTime();

        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                pAlpha, d_A, CUDA_R_32F, n, d_B, CUDA_R_32F, n,
                pBeta, d_C, CUDA_R_32F, n, CUDA_R_32F, GEMM_ALGO);

        cudaDeviceSynchronize();

        long after = System.nanoTime();
        double durationMs = (after - before) / 1e6;
        System.out.println(
                "Algorithm " + GEMM_ALGO + " took " + durationMs + " ms");

        // Copy the result from the device to the host
        cublasGetVector(nn, Sizeof.FLOAT, d_C, 1, Pointer.to(C), 1);

        // Clean up
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
    }

}