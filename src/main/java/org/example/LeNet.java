package org.example;

import com.sun.org.apache.xerces.internal.impl.dv.dtd.IDREFDatatypeValidator;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.cudaEvent_t;
import jcuda.driver.CUstream;
import jcuda.runtime.cudaStream_t;
import sun.security.provider.NativePRNG;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;
import static org.example.Layer2.fc3_w;
import static org.example.Reading2DArrayFromFile.*;

public class LeNet {
    public static double con1out[][][]=new double[6][28][28];
    static double[][][] turn= new double[6][5][5];
    static double[][][][]turn2 = new double[16][6][5][5];
    double[][][] vlsi = new double[6][36][36];
    double[][][] vlsi1out = new double[6][28][28];

    double[][][] vlsi2 = new double[16][18][18];
    double[][][] vlsi2out = new double[16][10][10];

    double[][][] pool1out = new double [6][14][14];
    double[][][] pool2out = new double [16][5][5];
    double hosttwww[] = new double[400*120];
    double[][][] con2out = new double [16][10][10];
    double [] flat = new double[400];
    double[] FC1 = new double[120];
    double[] FC2 = new double[84];
    double[] FC3 = new double[10];
    double[] OUT = new double[10];
    int numElements2 =32;
    int numElements3 =32;
    static double[]input;
    static double[][]Input;
    static double[][]Turn;
    static double[]weight;
    static double[]inputC2;
    static double[]weight2;
    static double[]COnv_weight1=new double[5*5*6];
    static double[][]COnv_weight=new double[6][5*5];
    static double[]COnv_weight2=new double[5*5*6*16];
    public static double[][]InputImage = new double[32][32];
    public static long time0;
    public static double tanh(double x) {
        return (Math.pow(Math.E, x) - Math.pow(Math.E, -x)) / (Math.pow(Math.E, x) + Math.pow(Math.E, -x));
    }
    void Import_weight(){

        Reading2DArrayFromFile.fc1_w=Reading2DArrayFromFile.writeToDat("FCwb2.txt");
        Reading2DArrayFromFile.fc2_w=Reading2DArrayFromFile.writeToDat2("fcw2.txt");    /** 1/16  **/
        int dst1 = Layer.weight_1[0][0].length-1;
        int dst2 = Layer.weight_1[0][0].length-1;
        for (int ker0 = 0; ker0 < 16; ker0++) {
            for (int ker = 0; ker < 6; ker++) {
                for (int i = 0; i < 5; i++, dst1--) {
                    for (int j = 0; j < 5; j++, dst2--) {
                        if(ker0==0){
                        turn[ker][i][j] = Layer.weight_1[ker][dst1][dst2];}
                        turn2[ker0][ker][i][j] = Layer.weight_2[ker0][ker][dst1][dst2];
                        if (dst2 == 0) dst2 = 5;
                    }
                    if (dst1 == 0) dst1 = 5;
                }
            }//轉180
        }
    }

    //記得改回TURN
    void GPU_parameters(double i [][] ,double w[][][],double w2[][][][]){
        int numkernel=w.length,numrow=w[0].length,numcolumn=w[0][0].length;
        int kernel,row,column;
        input=new double[i.length*i[0].length];
        Input=new double[i.length*i[0].length][1];
        Turn = new double[6][25];
        weight=new double[w.length*w[0].length*w[0][0].length];
        weight2=new double[16*6*5*5];
        double hostinput[]= new double[i.length*i[0].length];
        double hostweight[]= new double[numkernel*numrow*numcolumn];
        for(kernel=0;kernel < numkernel; kernel++)
        {
            for (row = 0; row < i.length; row++)
            {
                for (column = 0; column < i[0].length; column++)
                {
                    if(kernel==0)input[column + row * i[0].length] = i[row][column];
                    if(kernel==0)Input[column + row * i[0].length][0]= i[row][column];
                    if(row<numrow&&column<numcolumn)weight[column + row * numcolumn + kernel * numcolumn * numrow] = w[kernel][row][column];
                    if(row<numrow&&column<numcolumn)Turn[kernel][column + row * numcolumn] = w[kernel][row][column];
                }
            }
        }
        for(int ker0=0; ker0 <16 ; ker0 ++) {
            for (int ker = 0; ker < 6; ker++) {
                for (int k = 0; k < 5; k++) {
                    for (int j = 0; j < 5; j++) {
                    weight2[j+k*5+ker*5*5+ker0*5*5*6]=w2[ker0][ker][k][j];
                    }
                }
            }
        }
        for (int ker = 0; ker < 6; ker++) {
            for (int k = 0; k < 5; k++) {
                for (int j = 0; j < 5; j++) {
                     COnv_weight1[j+k*5+ker*5*5]=Layer.weight_1[ker][k][j];
                     COnv_weight[ker][j+k*5]=Layer.weight_1[ker][k][j];
                     //if(ker==0)
                         //System.out.println(COnv_weight1[j+k*5+ker*5*5]);
                }
            }
        }
        for (int ker0 = 0; ker0 < 16; ker0++) {
            for (int ker = 0; ker < 6; ker++) {
                for (int k = 0; k < 5; k++) {
                    for (int j = 0; j < 5; j++) {
                        COnv_weight2[j + k * 5 + ker * 5 * 5+ker0*5*5*6] = Layer.weight_2[ker0][ker][k][j];
                        //if(ker==0)
                        //System.out.println(COnv_weight1[j+k*5+ker*5*5]);
                    }
                }
            }
        }
    }

    //測試kernel程式邏輯









    void Conv1(){ /**32x32x1 → 28x28x6 Conv1總共做了117600次乘法和加法**/
    int counter=0;
        for (int ker = 0; ker < 6; ker++) {/**共六個卷積核**/
            for (int i = 0; i < 28; i++) {/**卷積核每次卷積走一步 共要在列走28次**/
                for (int j = 0; j < 28; j++) {

                    for (int k = 0; k < 5; k++) {/**卷積一次有五個列要卷積**/
                        for (int l = 0; l < 5; l++) {/**卷積一個列有五個元素要卷積**/
                            con1out[ker][i][j] +=  InputImage[k + i][l + j] * Layer.weight_1[ker][k][l];
                            counter++;
                        }
                    }

                    con1out[ker][i][j] = tanh(con1out[ker][i][j]+Layer.bias_1[ker]);
                    if(Double.isNaN(tanh(con1out[ker][i][j]+Layer.bias_1[ker]))) {
                        con1out[ker][i][j] = 1.0;
                    }
                   // System.out.println("con["+ker+"]["+i+"]["+j+"]= "+con1out[ker][i][j]);
                }
            }
        }//for ker
        //System.out.println("Conv1總共做了"+counter+"次乘法和加法");
    }
    void showvlsiout(){
        for (int ker = 0; ker < 6; ker++) {
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    if (j==0)  System.out.print("Convout["+ ker +"][" + i + "][" + j + "]=");
                    System.out.print( con1out[ker][i][j]+" " );
                    if (j==27) System.out.print("\n");
                }
            }
        }
    }
    void CoonnVV(){/**32x32x1 → 28x28x6 VLSI1總共做了153600次乘法和加法**/

        long time,time2;
        time = System.nanoTime();
        for (int ker = 0; ker < 6; ker++) {/**共六個卷積核**/
            for (int i = 0; i < 28; i++) {/**卷積核每次卷積走一步 共要在列走28次**/
                for (int j = 0; j < 28; j++) {

                    for (int k = 0; k < 5; k++) {/**卷積一次有五個列要卷積**/
                        for (int l = 0; l < 5; l++) {/**卷積一個列有五個元素要卷積**/
                            con1out[ker][i][j] +=  InputImage[k + i][l + j] * Layer.weight_1[ker][k][l];
                        }
                    }

                    con1out[ker][i][j] = tanh(con1out[ker][i][j]+Layer.bias_1[ker]);
                    //if(Double.isNaN(tanh(con1out[ker][i][j]+Layer.bias_1[ker]))) {
                    //    con1out[ker][i][j] = 1.0;
                    //     }
                    // System.out.println("con["+ker+"]["+i+"]["+j+"]= "+con1out[ker][i][j]);
                }
            }
        }//for ker
        for(int ker = 0 ; ker < 6 ; ker++) {
            for (int i = 0,strideC=0; i < 14; i++,strideC+=2) {/**步數2**/
                for (int j = 0,strideR=0; j < 14; j++,strideR+=2) {
                    for(int k = 0 ; k < 2 ; k++){
                        for (int l = 0 ; l < 2 ; l++){
                            if(k==0&&l==0)//先設定初值再做比較
                                pool1out[ker][i][j]=con1out[ker][strideC+k][strideR+l];
                            // pool1out[ker][i][j]=vlsi1out[ker][strideC+k][strideR+l];
                            if(pool1out[ker][i][j]<con1out[ker][strideC+k][strideR+l])

                                pool1out[ker][i][j]=con1out[ker][strideC+k][strideR+l];//vlsi1out
                        }
                    }
                }
            }
        }

        for (int ker = 0; ker <16; ker++){
            for (int ker2 = 0; ker2 < 6; ker2++) {
                for (int i = 0; i < 10; i++) {
                    for (int j = 0; j < 10; j++) {
                        for (int k = 0; k < 5; k++) {
                            for (int l = 0; l < 5; l++) {
                                con2out[ker][i][j] = con2out[ker][i][j] + pool1out[ker2][k+i][l+j] * Layer.weight_2[ker][ker2][k][l];
                            }
                        }
                        if(ker2==5){
                            con2out[ker][i][j] = tanh(con2out[ker][i][j]+Layer.bias_2[ker]);
                            //if(Double.isNaN(tanh(con2out[ker][i][j]+Layer.bias_2[ker]))) {
                                //System.out.println("c2 "+ker+" "+i+" "+j+" ");
                                //con2out[ker][i][j] = 1.0;
                            //}
                        }
                    }
                }
            }
        }


        for(int ker = 0 ; ker < 16 ; ker++) {
            for (int i = 0,strideC=0; i < 5; i++,strideC+=2) {
                for (int j = 0,strideR=0; j < 5; j++,strideR+=2) {
                    for(int k = 0 ; k < 2 ; k++){
                        for (int l = 0 ; l < 2 ; l++){
                            if(k==0&&l==0)//POOL剛開始是0 比負值大
                                pool2out[ker][i][j]=con2out[ker][strideC+k][strideR+l];
                            if(pool2out[ker][i][j]<con2out[ker][strideC+k][strideR+l])
                                pool2out[ker][i][j]=con2out[ker][strideC+k][strideR+l];//vlsi2out
                        }
                    }
                }
            }
        }//Pooling END


        int z=0;
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                for(int ker = 0; ker < 16; ker++){
                    flat[z]=pool2out[ker][i][j ];
                    z++;
                }
            }
        }
        time2 = System.nanoTime();//////////////////////////////

        for (int i = 0; i < 120; i++) {
            for (int j = 0; j < 400; j++) {
                FC1[i]=FC1[i]+(flat[j]*fc1_w[j][i]);//
            }
            //System.out.println("FFCB1 ["+Layer2.fc1_b[i]+"]");
            FC1[i]=tanh(FC1[i]+Layer2.fc1_b[i]);//+Layer2.fc1_b[i]
        }

        for (int i = 0; i < 84; i++) {
            for (int j = 0; j < 120; j++) {
                FC2[i]=FC2[i]+FC1[j]*fc2_w[j][i];
            }
            FC2[i]=tanh(FC2[i]+Layer2.fc2_b[i]);
        }

        double exp_d = 0.0;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 84; j++) {
                FC3[i]=FC3[i]+FC2[j]* fc3_w[j][i];
            }
            FC3[i]=FC3[i]+Layer2.fc3_b[i];
            exp_d=exp_d+Math.pow(Math.E, FC3[i]);
        }
        //System.out.print("exp="+exp_d+"\n");
        for (int i = 0; i < 10; i++) {/**softmax **/
            OUT[i]=Math.pow(Math.E, FC3[i])/exp_d;
            System.out.print("OUT["+i+"]="+OUT[i]+"\n");

        }

        //////////////
        int max=0;
        double maxout=0.0;
        for (int i = 0; i < 10; i++) {
            if(OUT[i]>maxout) {
                maxout=OUT[i];
                max=i;
            }
        }
        System.out.print("The number is "+max+"\n");
        //System.out.println((time2-time)+" ns  ");


    }///////////////////////////


    void checkpool1gpu(){
        int xxxx=0;
        for(int ker = 0 ; ker < 6 ; ker++){
            for (int row = 0 ; row < 14 ; row++) {
                for (int col = 0;col < 14; col++) {
                    if(pool1out[ker][row][col]- hosttwww[col + row *14 + ker * 14 * 14]>0.0001){
                        xxxx++;
                        System.out.println("poolcheck "+pool1out[ker][row][col]+" "+hosttwww[col + row *14 + ker * 14 * 14]+" ["+ker+"]["+row+"]["+col+"]"+"XXX\n");
                        //System.out.print("\n"+vlsi[ker][i+4][j+4]+" ["+ker+"]["+(i+4)+"]["+(j+4)+"]"+"XXX\n");
                    }
                }
            }
        }
        if(xxxx==0)System.out.println("pool1GPU pass error="+xxxx);
        else System.out.println("pool1GPUxxxx= "+xxxx);
    }
    void check(){
        int xx=0;
        for(int ker = 0 ; ker < 6 ; ker++){
            for (int i = 0 ; i < 28 ; i++) {
                for (int j = 0; j < 28; j++) {
                    if(con1out[ker][i][j]!= vlsi1out[ker][i+0][j+0]){
                        xx++;
                        System.out.print("\n"+vlsi1out[ker][i+0][j+0]+" ["+ker+"]["+i+"]["+j+"]"+"XXX\n");
                        //System.out.print("\n"+vlsi[ker][i+4][j+4]+" ["+ker+"]["+(i+4)+"]["+(j+4)+"]"+"XXX\n");
                    }
                }
            }
        }
        System.out.println("xx="+xx);
        if(xx==0) System.out.println("VLSI1GPU PASS");
    }
    void checkpool2(){
        int xx=0;
        for(int ker = 0 ; ker <16 ; ker++){
            for (int i = 0 ; i < 5 ; i++) {
                for (int j = 0; j < 5; j++) {
                    if(Matrix.b[ker][i][j]- pool2out[ker][i][j] >0.00001){
                        xx++;
                        System.out.print("\n"+vlsi2out[ker][i][j]+" "+ker+" "+i+" "+j+" "+"XXX\n");
                    }
                }
            }
        }
        if(xx==0) System.out.println("Matrix b==pool2out");
    }
    void check2(){
        int xx=0;
        for(int ker = 0 ; ker <16 ; ker++){
            for (int i = 0 ; i < 10 ; i++) {
                for (int j = 0; j < 10; j++) {
                    if(con2out[ker][i][j]!= vlsi2out[ker][i][j]){
                        xx++;
                        System.out.print("\n"+vlsi2out[ker][i][j]+" "+ker+" "+i+" "+j+" "+"XXX\n");
                    }
                }
            }
        }
        if(xx==0) System.out.println("con2out==vlsi2out");
    }
    void Pooling1() {
        for(int ker = 0 ; ker < 6 ; ker++) {
            for (int i = 0,strideC=0; i < 14; i++,strideC+=2) {/**步數2**/
                for (int j = 0,strideR=0; j < 14; j++,strideR+=2) {
                    for(int k = 0 ; k < 2 ; k++){
                        for (int l = 0 ; l < 2 ; l++){
                            if(k==0&&l==0)//先設定初值再做比較
                                pool1out[ker][i][j]=con1out[ker][strideC+k][strideR+l];
                               // pool1out[ker][i][j]=vlsi1out[ker][strideC+k][strideR+l];
                            if(pool1out[ker][i][j]<con1out[ker][strideC+k][strideR+l])

                                pool1out[ker][i][j]=con1out[ker][strideC+k][strideR+l];//vlsi1out
                        }
                    }
                }
            }
        }
    }//Pooling END
    void Pooling2() {
        for(int ker = 0 ; ker < 16 ; ker++) {
            for (int i = 0,strideC=0; i < 5; i++,strideC+=2) {
                for (int j = 0,strideR=0; j < 5; j++,strideR+=2) {
                    for(int k = 0 ; k < 2 ; k++){
                        for (int l = 0 ; l < 2 ; l++){
                            if(k==0&&l==0)//POOL剛開始是0 比負值大
                                pool2out[ker][i][j]=con2out[ker][strideC+k][strideR+l];
                            if(pool2out[ker][i][j]<con2out[ker][strideC+k][strideR+l])
                                pool2out[ker][i][j]=con2out[ker][strideC+k][strideR+l];//vlsi2out
                        }
                    }
                }
            }
        }//Pooling END
        int z=0;
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                for(int ker = 0; ker < 16; ker++){
                    flat[z]=pool2out[ker][i][j ];
                    z++;
                }
            }
        }
    }
    void Conv2(){
        for (int ker = 0; ker <16; ker++){
            for (int ker2 = 0; ker2 < 6; ker2++) {
                for (int i = 0; i < 10; i++) {
                    for (int j = 0; j < 10; j++) {
                        for (int k = 0; k < 5; k++) {
                            for (int l = 0; l < 5; l++) {
                                con2out[ker][i][j] = con2out[ker][i][j] + pool1out[ker2][k+i][l+j] * Layer.weight_2[ker][ker2][k][l];
                            }
                        }
                        if(ker2==5){
                            con2out[ker][i][j] = tanh(con2out[ker][i][j]+Layer.bias_2[ker]);
                            if(Double.isNaN(tanh(con2out[ker][i][j]+Layer.bias_2[ker]))) {
                                System.out.println("c2 "+ker+" "+i+" "+j+" ");
                                //con2out[ker][i][j] = 1.0;
                            }
                        }
                    }
                }
            }
        }
    }//Conv2 END
    void flatten(){
    int z=0;
        for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for(int ker = 0; ker < 16; ker++){
                flat[z]=pool2out[ker][i][j ];
                z++;
            }
        }
    }
}

    void shownums(){
        for (int j = 0; j < 1; j++) {
            for (int i = 0; i < 400; i++) {

                if (i==0)  System.out.print("numsout[" + i + "][" + j + "]=");
                System.out.print( fc1_w[i][j]+" " );
                if (i==399) System.out.print("\n");
            }
        }
    }
    void showFC1out(){
    for (int i = 0; i < 120; i++) {
        if (i%10==0)  System.out.print("FC1[" + i + "]=");
        System.out.print( FC1[i]+" " );
        if (i%10==9) System.out.print("\n");
    }
}
    void showc2out(){
        for (int ker = 0; ker < 16; ker++) {
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 10; j++) {
                    if (j==0)  System.out.print("con2out["+ ker +"][" + i + "][" + j + "]=");
                    System.out.print( con2out[ker][i][j]+" " );
                    if (j==9) System.out.print("\n");
                }
            }
        }
    }
    void showvlsi2out(){
        for (int ker = 0; ker < 16; ker++) {
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 10; j++) {
                    if (j==0)  System.out.print("vlsi2out["+ ker +"][" + i + "][" + j + "]=");
                    System.out.print( vlsi2out[ker][i][j]+" " );
                    if (j==9) System.out.print("\n");
                }
            }
        }
    }
    void showvlsi2(){
        for (int ker = 0; ker < 16; ker++) {
            for (int i = 0; i < 18; i++) {
                for (int j = 0; j < 18; j++) {
                    if (j==0)  System.out.print("vlsi2["+ ker +"][" + i + "][" + j + "]=");
                    System.out.print( vlsi2[ker][i][j]+" " );
                    if (j==17) System.out.print("\n");
                }
            }
        }
    }
    void showflat(){
        for (int i = 0,z=0; i < 400; i++,z++) {
            System.out.print( flat[i]+" " );
            if(z==9) {
                System.out.print( "\n" );
                z=0;
            }
        }
    }
    void showvls1(){
        for (int ker = 0; ker < 1; ker++) {
            for (int i = 0; i < 36; i++) {
                for (int j = 0; j < 36; j++) {
                    if (j==0)  System.out.print("vlsi["+ ker +"][" + i + "][" + j + "]=");
                    System.out.print( vlsi[ker][i][j]+" " );
                    if(vlsi[ker][i][j]!=0) System.out.print("[" + i + "][" + j + "]");
                    if (j==35) System.out.print("\n");
                }
            }
        }
    }

    void checkflat() {
        int xx=0;
        for (int i = 0 ; i < 400; i++) {
            if (Matrix.c[i] - flat[i] > 0.00001) {
                xx++;
                System.out.print("\n" + flat[i] + " " + i + " " + "XXX\n");
            }
        }
        if(xx==0) System.out.print("flat == python flat \n");
    }
    void showpool(){
        for(int ker = 0 ; ker < 6 ; ker++) {
            for (int i = 0; i < 14; i++) {
                for (int j = 0; j < 14; j++) {
                    if (j==0) System.out.print("pool["+ ker +"][" + (i) + "][" + (j) + "]=");
                    System.out.print(" "+pool1out[ker][i][j]+" ");
                    if(j==13) System.out.print("\n");
                }
            }
        }
    }
    void showpool2(){
        for(int ker = 0 ; ker < 16 ; ker++) {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    if (j==0)  System.out.print("pool2["+ ker +"][" + (i) + "][" + (j) + "]=");
                    System.out.print(" "+pool2out[ker][i][j]+" ");
                    if(j==4) System.out.print("\n");
                }
            }
        }
    }
    void showturn(){
        for (int ker0 = 0; ker0 < 16; ker0++) {
            for (int ker = 0; ker < 6; ker++) {
                for (int i = 0; i < 5; i++) {
                    for (int j = 0; j < 5; j++) {
                        if (j==0)  System.out.print("turn2["+ker0+"]["+ ker +"][" + (i) + "][" + (j) + "]=");
                        System.out.print(" "+turn2[ker0][ker][i][j]+" ");
                        if(j==4) System.out.print("\n");
                }
            }
        }
    }

    }
    void FullyConnect1(){
        for (int i = 0; i < 120; i++) {
            for (int j = 0; j < 400; j++) {
                FC1[i]=FC1[i]+(flat[j]*fc1_w[j][i]);//
            }
            //System.out.println("FFCB1 ["+Layer2.fc1_b[i]+"]");
            FC1[i]=tanh(FC1[i]+Layer2.fc1_b[i]);//+Layer2.fc1_b[i]
        }
    }
    void FullyConnect2(){
        for (int i = 0; i < 84; i++) {
            for (int j = 0; j < 120; j++) {
                FC2[i]=FC2[i]+FC1[j]*fc2_w[j][i];
            }
            FC2[i]=tanh(FC2[i]+Layer2.fc2_b[i]);
        }
    }
    void showfc2(){
        for (int i = 0; i < 84; i++) {
            if (i%10==0)  System.out.print("FC2[" + i + "]=");
            System.out.print( FC2[i]+" " );
            if (i%10==9) System.out.print("\n");
            if (i==83) System.out.print("\n");
        }
    }
    void checkFC2(){
        int xx=0;
        for (int j = 0; j < 84; j++) {
            if(FC2[j]-Matrix.FC2py[j]>0.00001){
                xx++;
                System.out.print("XXX is "+j+"\n");
            }
        }
        if(xx==0) System.out.print("FC2JAVA==FC2py\n");
    }
    void FullyConnect3(){
        double exp_d = 0.0;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 84; j++) {
                FC3[i]=FC3[i]+FC2[j]* fc3_w[j][i];
            }
            FC3[i]=FC3[i]+Layer2.fc3_b[i];
            exp_d=exp_d+Math.pow(Math.E, FC3[i]);
        }
        //System.out.print("exp="+exp_d+"\n");
        for (int i = 0; i < 10; i++) {/**softmax **/
            OUT[i]=Math.pow(Math.E, FC3[i])/exp_d;


        }
    }
    void Result_of_classification(){
        int max=0;
        double maxout=0.0;
        for (int i = 0; i < 10; i++) {
            System.out.print("OUT["+i+"]="+OUT[i]+"\n");
            if(OUT[i]>maxout) {
                maxout=OUT[i];
                max=i;
            }
        }
        System.out.print("The number is "+max+"\n");
    }



void NEW_LeNet(){

    int xx=0,r=28;

    double deviceOutput[] = new double[6*r*r];       /** 宣告一個陣列作為Device輸出 **/
    double hostinput[]= input;
    double hosttestin[]= weight;
    // Enable exceptions and omit all subsequent error checks
    JCudaDriver.setExceptionsEnabled(true);
    // Initialize the driver and create a context for the first device.
    cuInit(0);
    CUdevice device = new CUdevice();
    cuDeviceGet(device, 0);
    CUcontext context = new CUcontext();
    cuCtxCreate(context, 0, device); /** 顯示卡初始化 **/
    // Create the PTX file by calling the NVCC
    // Load the ptx file.
    String ptxFileName = JCudaSamplesUtils.preparePtxFile("src/main/resources/test.cu");
    CUmodule module = new CUmodule();
    cuModuleLoad(module, ptxFileName);

    CUfunction Pcore = new CUfunction();
    cuModuleGetFunction(Pcore, module, "PartialCore");   /**讀取cu檔kernel  **/
    CUfunction Acore = new CUfunction();
    cuModuleGetFunction(Acore, module, "ACore");   /**讀取cu檔kernel  **/


    CUfunction P2core = new CUfunction();
    cuModuleGetFunction(P2core, module, "Partial2Core");   /**讀取cu檔kernel  **/
    CUfunction A2core = new CUfunction();
    cuModuleGetFunction(A2core, module, "A2Core");   /**讀取cu檔kernel  **/
    CUfunction PoolCore = new CUfunction();
    cuModuleGetFunction(PoolCore, module, "poolcore");   /**讀取cu檔kernel  **/
    CUfunction Pool2core = new CUfunction();
    cuModuleGetFunction(Pool2core, module, "Pool2Core");

    CUfunction FC = new CUfunction();
    cuModuleGetFunction(FC, module, "FC");
    CUfunction FC3 = new CUfunction();
    cuModuleGetFunction(FC3, module, "FCL3");
    float[] time = new float[1];
    float[] time2 = new float[1];
    float[] timetoa = new float[1];
    CUevent start= new CUevent();                           //建立用來計算 演算法運算時間的事件
    cuEventCreate(start,0);
    CUevent stop= new CUevent();
    cuEventCreate(stop,0);
    CUevent start2= new CUevent();
    cuEventCreate(start2,0);
    CUevent stop2= new CUevent();
    cuEventCreate(stop2,0);


    CUstream stream1 = new CUstream();
    CUstream stream2 = new CUstream();      //cuda流建立
    CUstream stream3 = new CUstream();
    cuStreamCreate(stream1,0);
    cuStreamCreate(stream2,0);
    cuStreamCreate(stream3,0);
    CUstream stream4 = new CUstream();
    CUstream stream5 = new CUstream();
    CUstream stream6 = new CUstream();
    cuStreamCreate(stream4,0);
    cuStreamCreate(stream5,0);
    cuStreamCreate(stream6,0);
    //cuStreamCreateWithPriority(stream5,0,0);
    //CUdeviceptr deviceinput2 = new CUdeviceptr();
    CUdeviceptr deviceInputimage = new CUdeviceptr();
    //CUdeviceptr deviceInputrow = new CUdeviceptr();
    CUdeviceptr deviceW = new CUdeviceptr();
    CUdeviceptr deviceB = new CUdeviceptr();

    CUdeviceptr deviceP_1 = new CUdeviceptr();
    CUdeviceptr deviceP_2 = new CUdeviceptr();
    CUdeviceptr deviceP_3 = new CUdeviceptr();
    //CUdeviceptr deviceK = new CUdeviceptr();
    CUdeviceptr AM1 = new CUdeviceptr();
    CUdeviceptr AM2 = new CUdeviceptr();
    CUdeviceptr AM3 = new CUdeviceptr();
    CUdeviceptr AM4 = new CUdeviceptr();
    CUdeviceptr AM5 = new CUdeviceptr();
    CUdeviceptr AM6 = new CUdeviceptr();

    CUdeviceptr deviceout = new CUdeviceptr();
    CUdeviceptr devicetanout = new CUdeviceptr();
    CUdeviceptr deviceout2 = new CUdeviceptr();
    CUdeviceptr devicePout = new CUdeviceptr();

    int[][]inputpix=Layer.Inputpix;
    int[][]inpixrow=Layer.Inputpixrow;
    //double[][] hostdata =Layer.hohoho;
    double[][] hostdata =InputImage;//32*32
    //double[]hostinputdata=Layer.Inputtest;
    double[]hostinputdata=input;

    double[]hostwe=weight;//5*5*6                           //權重係數
   double[]   hostbias=Layer.bias_1;
    double[][] hosttestt=new double[6][36*36];



    /////////////////////////////////////222

    //CUdeviceptr deviceInputrow2 = new CUdeviceptr();
    CUdeviceptr deviceW2 = new CUdeviceptr();
    CUdeviceptr deviceB2 = new CUdeviceptr();

    CUdeviceptr deviceP2_1 = new CUdeviceptr();
    CUdeviceptr deviceP2_2 = new CUdeviceptr();
    CUdeviceptr deviceP2_3 = new CUdeviceptr();
    //CUdeviceptr deviceK2 = new CUdeviceptr();
    CUdeviceptr AM1_2 = new CUdeviceptr();
    CUdeviceptr AM2_2 = new CUdeviceptr();
    CUdeviceptr AM3_2 = new CUdeviceptr();
    CUdeviceptr AM4_2 = new CUdeviceptr();
    CUdeviceptr AM5_2 = new CUdeviceptr();
    CUdeviceptr AM6_2 = new CUdeviceptr();
    //CUdeviceptr deviceout_2 = new CUdeviceptr();
    CUdeviceptr devicetanout_2 = new CUdeviceptr();
    CUdeviceptr deviceout2_2 = new CUdeviceptr();

/********************pool1********************/
    int[] parp1={14,14,6,28};
    double[] hostpool1=new double[14*14*6];
    double[]pooll=new double[14*14*6];
    for(int ker=0;ker<6;ker++) {
        for (int j = 0; j < 14; j++) {
            for (int i = 0; i < 14; i++) {
                pooll[i+j*14+ker*14*14]=-256;
            }
        }
    }
    CUdeviceptr devicepool = new CUdeviceptr();

    //CUdeviceptr deviceparP1 = new CUdeviceptr();
    CUdeviceptr devicepoolindex = new CUdeviceptr();

    cuMemAlloc(devicepool, 6*14*14 * Sizeof.DOUBLE);
    cuMemcpyHtoD(devicepool, Pointer.to(pooll), 14*14*6 * Sizeof.DOUBLE);
    //cuMemAlloc(deviceparP1, 4 * Sizeof.DOUBLE);
    cuMemAlloc(devicepoolindex, 1 * Sizeof.INT);

    //cuMemcpyHtoD(devicepool, Pointer.to(devicepool), 6*14*14 * Sizeof.DOUBLE);
    //cuMemcpyHtoD(deviceparP1, Pointer.to(parp1), 4 * Sizeof.DOUBLE);
    Pointer pool1par=Pointer.to(                                            //各個Kernel函數使用的參數建立
            //Pointer.to(deviceparP1),//參數
            Pointer.to(devicepool),//輸出
            Pointer.to(devicetanout)//輸入

    );

    Pointer poolcorepar1=Pointer.to(
            Pointer.to(devicepool),//輸出
            Pointer.to(devicetanout),//輸入
            Pointer.to(deviceP_1)

    );
    Pointer poolcorepar2=Pointer.to(
            Pointer.to(devicepool),//輸出
            Pointer.to(devicetanout),//輸入
            Pointer.to(deviceP_2)

    );
    Pointer poolcorepar3=Pointer.to(
            Pointer.to(devicepool),//輸出
            Pointer.to(devicetanout),//輸入
            Pointer.to(deviceP_3)

    );
////////////////////////////////////////////////////////////////////
    int[] parp2={5,5,16,10};

    CUdeviceptr devicepool2 = new CUdeviceptr();
    CUdeviceptr deviceparP2 = new CUdeviceptr();
    CUdeviceptr devicepool2index = new CUdeviceptr();
    cuMemAlloc(devicepool2, 5*5*16 * Sizeof.DOUBLE);
    double[]pooll2=new double[5*5*16];
    for(int ker=0;ker<16;ker++) {
        for (int j = 0; j < 5; j++) {
            for (int i = 0; i < 5; i++) {
                pooll2[i+j*5+ker*5*5]=-256;
            }
        }
    }
    cuMemcpyHtoD(devicepool2, Pointer.to(pooll2), 5*5*16 * Sizeof.DOUBLE);
    cuMemAlloc(deviceparP2, 4 * Sizeof.DOUBLE);
    cuMemAlloc(devicepool2index, 1 * Sizeof.INT);
    cuMemcpyHtoD(deviceparP2, Pointer.to(parp2), 4 * Sizeof.DOUBLE);
    Pointer pool2par=Pointer.to(
            Pointer.to(deviceparP2),//參數
            Pointer.to(devicepool2),//輸出
            Pointer.to(devicetanout_2)//輸入
    );

    Pointer pool2corepar1=Pointer.to(

            Pointer.to(devicepool2),//輸出
            Pointer.to(devicetanout_2),//輸入
            Pointer.to(devicepool2index),
            Pointer.to(deviceP2_1)

    );
    Pointer pool2corepar2=Pointer.to(

            Pointer.to(devicepool2),//輸出
            Pointer.to(devicetanout_2),//輸入
            Pointer.to(devicepool2index),
            Pointer.to(deviceP2_2)

    );
    Pointer pool2corepar3=Pointer.to(

            Pointer.to(devicepool2),//輸出
            Pointer.to(devicetanout_2),//輸入
            Pointer.to(devicepool2index),
            Pointer.to(deviceP2_3)

    );


    /**********Conv2************/
    double host2bias[]=Layer.bias_2;
    int inpixrow2[][]=Layer.Inputpixrow2;
    double[] host2we=new double[5*5*6*16];
    for(int oker=0;oker<16;oker++) {
        for (int ker = 0; ker < 6; ker++) {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    //host2we[j + i * 5 + ker * 5 * 5 + oker * 6 * 5 * 5] = oker;
                    host2we[j + i * 5 + ker * 5 * 5 + oker * 6 * 5 * 5] =turn2[oker][ker][i][j];
                }
            }
        }
    }
    //cudaHostAlloc(deviceInputimage2, 14*14*6 * Sizeof.DOUBLE,0);

    //cuMemAlloc(deviceInputrow2, 14*6*2 * Sizeof.DOUBLE);             //14*6*2輸入進Pcore 原14*6
    cuMemAlloc(deviceout2_2, 5*18*16*3 * Sizeof.DOUBLE);              //還沒改
    cuMemAlloc(deviceB2, 16 * Sizeof.DOUBLE);

    cuMemAlloc(deviceP2_1,14*Sizeof.INT);
    cuMemAlloc(deviceP2_2,14*Sizeof.INT);
    cuMemAlloc(deviceP2_3,14*Sizeof.INT);  //現在處理第幾列
    //cuMemAlloc(deviceK2,16*Sizeof.INT);
    //cudaHostAlloc(deviceW2, 5*5*6*16 * Sizeof.DOUBLE,0);
    cuMemAlloc(deviceW2, 5*5*6*16 * Sizeof.DOUBLE);
    int AM_2=10*16;
    cuMemAlloc(AM1_2,AM_2*Sizeof.DOUBLE);                       //還沒改
    cuMemAlloc(AM2_2,AM_2*Sizeof.DOUBLE);
    cuMemAlloc(AM3_2,AM_2*Sizeof.DOUBLE);
    cuMemAlloc(AM4_2,AM_2*Sizeof.DOUBLE);
    cuMemAlloc(AM5_2,AM_2*Sizeof.DOUBLE);
    cuMemAlloc(AM6_2,AM_2*Sizeof.DOUBLE);
    //cuMemAlloc(deviceout_2, 18*18*16 * Sizeof.DOUBLE);
    cuMemAlloc(devicetanout_2, 10*10*16 * Sizeof.DOUBLE);
    /////////////////////////////////////222
 /*   Pointer inputpar1=Pointer.to(
            Pointer.to(deviceInputimage),
            Pointer.to(deviceInputrow),
            Pointer.to(deviceP_1)
    );
    Pointer inputpar2=Pointer.to(
            Pointer.to(deviceInputimage),
            Pointer.to(deviceInputrow),
            Pointer.to(deviceP_2)
    );    Pointer inputpar3=Pointer.to(
            Pointer.to(deviceInputimage),
            Pointer.to(deviceInputrow),
            Pointer.to(deviceP_3)
    );*/

    Pointer Ppar1=Pointer.to(
            Pointer.to(deviceInputimage),//輸入
            Pointer.to(devicePout),//輸出
            Pointer.to(deviceW),
            Pointer.to(deviceP_1)
    );

    Pointer Ppar2=Pointer.to(
            Pointer.to(deviceInputimage),
            Pointer.to(devicePout),
            Pointer.to(deviceW),
            Pointer.to(deviceP_2)
    );    Pointer Ppar3=Pointer.to(
            Pointer.to(deviceInputimage),
            Pointer.to(devicePout),
            Pointer.to(deviceW),
            Pointer.to(deviceP_3)
    );

    Pointer Apar1=Pointer.to(
            Pointer.to(devicePout), //輸入

            Pointer.to(deviceB),
            Pointer.to(deviceP_1),
            Pointer.to(devicetanout),//輸出
            Pointer.to(AM1),
            Pointer.to(AM2),
            Pointer.to(AM3),
            Pointer.to(AM4),
            Pointer.to(AM5),
            Pointer.to(AM6)
    );

    Pointer Apar2=Pointer.to(
            Pointer.to(devicePout),

            Pointer.to(deviceB),
            Pointer.to(deviceP_2),
            Pointer.to(devicetanout),
            Pointer.to(AM1),
            Pointer.to(AM2),
            Pointer.to(AM3),
            Pointer.to(AM4),
            Pointer.to(AM5),
            Pointer.to(AM6));
    Pointer Apar3=Pointer.to(
            Pointer.to(devicePout),

            Pointer.to(deviceB),
            Pointer.to(deviceP_3),
            Pointer.to(devicetanout),
            Pointer.to(AM1),
            Pointer.to(AM2),
            Pointer.to(AM3),
            Pointer.to(AM4),
            Pointer.to(AM5),
            Pointer.to(AM6));


    Pointer tanbiasp=Pointer.to(
            Pointer.to(deviceout),
            Pointer.to(deviceB),
            Pointer.to(devicetanout)
        );

    //////////////////////////////////////////////////

    Pointer inputpar_21=Pointer.to(
            Pointer.to(devicepool),//deviceInputimage2
           // Pointer.to(deviceInputrow2),
            Pointer.to(deviceP2_1)

    );
    Pointer inputpar_22=Pointer.to(
            Pointer.to(devicepool),//deviceInputimage2
            //Pointer.to(deviceInputrow2),
            Pointer.to(deviceP2_2)

    );
    Pointer inputpar_23=Pointer.to(
            Pointer.to(devicepool),
            //Pointer.to(deviceInputrow2),
            Pointer.to(deviceP2_3)

    );

    Pointer Ppar_21=Pointer.to(
            Pointer.to(devicepool),//[0][32] in 14*14*6
            Pointer.to(deviceout2_2),//out   5*18*16 *2
            Pointer.to(deviceW2),
            Pointer.to(deviceP2_1)

    );
    Pointer Ppar_22=Pointer.to(
            Pointer.to(devicepool),//[0][32] in 14*14*6
            Pointer.to(deviceout2_2),//out   5*18*16 *2
            Pointer.to(deviceW2),
            Pointer.to(deviceP2_2)

    );
    Pointer Ppar_23=Pointer.to(
            Pointer.to(devicepool),//[0][32] in  14*14*6
            Pointer.to(deviceout2_2),//out    5*18*16 *2
            Pointer.to(deviceW2),
            Pointer.to(deviceP2_3)

    );

    Pointer Apar_21=Pointer.to(
            Pointer.to(deviceout2_2),//in
            //Pointer.to(deviceout_2),
            Pointer.to(deviceB2),
            Pointer.to(deviceP2_1),
            Pointer.to(devicetanout_2),//out
            Pointer.to(AM1_2),
            Pointer.to(AM2_2),
            Pointer.to(AM3_2),
            Pointer.to(AM4_2),
            Pointer.to(AM5_2),
            Pointer.to(AM6_2)
    );
    Pointer Apar_22=Pointer.to(
            Pointer.to(deviceout2_2),
            //Pointer.to(deviceout_2),
            Pointer.to(deviceB2),
            Pointer.to(deviceP2_2),
            Pointer.to(devicetanout_2),
            Pointer.to(AM1_2),
            Pointer.to(AM2_2),
            Pointer.to(AM3_2),
            Pointer.to(AM4_2),
            Pointer.to(AM5_2),
            Pointer.to(AM6_2)
    );
    Pointer Apar_23=Pointer.to(
            Pointer.to(deviceout2_2),
            //Pointer.to(deviceout_2),
            Pointer.to(deviceB2),
            Pointer.to(deviceP2_3),
            Pointer.to(devicetanout_2),
            Pointer.to(AM1_2),
            Pointer.to(AM2_2),
            Pointer.to(AM3_2),
            Pointer.to(AM4_2),
            Pointer.to(AM5_2),
            Pointer.to(AM6_2)
    );
    Pointer tanbiasp2=Pointer.to(
            //Pointer.to(deviceout_2),
            Pointer.to(deviceB2),
            Pointer.to(devicetanout_2)
    );



    cuMemAlloc(devicePout, 5*36*6*2 * Sizeof.DOUBLE);              //用這個當partial result 暫存


    //cuMemAlloc(deviceK,6*Sizeof.INT);

    cuMemAlloc(deviceW, 25*6 * Sizeof.DOUBLE);
    cuMemAlloc(deviceInputimage, 32*32 * Sizeof.DOUBLE);
    //cuMemAlloc(deviceInputrow, 35*2 * Sizeof.DOUBLE);      //原32*2
    cuMemAlloc(deviceP_1,32*Sizeof.INT);
    cuMemAlloc(deviceP_2,32*Sizeof.INT);
    cuMemAlloc(deviceP_3,32*Sizeof.INT);
    cuMemAlloc(deviceB, 6 * Sizeof.DOUBLE);

  /* cuMemAllocHost(deviceW, 25*6 * Sizeof.DOUBLE);
    cuMemAllocHost(deviceInputimage, 32*32 * Sizeof.DOUBLE);
    cuMemAllocHost(deviceInputrow, 32 * Sizeof.DOUBLE);
    cuMemAllocHost(deviceP_1,32*Sizeof.INT);
    cuMemAllocHost(deviceP_2,32*Sizeof.INT);
    cuMemAllocHost(deviceP_3,32*Sizeof.INT);
    cuMemAllocHost(deviceB, 6 * Sizeof.DOUBLE);*/
    int AMDIM=28*6;
    cuMemAlloc(AM1,AMDIM*Sizeof.DOUBLE);//36*9*6
    cuMemAlloc(AM2,AMDIM*Sizeof.DOUBLE);
    cuMemAlloc(AM3,AMDIM*Sizeof.DOUBLE);
    cuMemAlloc(AM4,AMDIM*Sizeof.DOUBLE);
    cuMemAlloc(AM5,AMDIM*Sizeof.DOUBLE);
    cuMemAlloc(AM6,AMDIM*Sizeof.DOUBLE);
    //cudaMallocHost(hostinput123,5 * Sizeof.DOUBLE);
    //cuMemAlloc(deviceout, 36*36 * Sizeof.DOUBLE);
    /**FC1**/
    long time1,time22;
    double []FFC1=new double[120];
    double []FFCW1=new  double[120*400];
    int []FFCpar1 ={400,120};
    for (int i = 0; i <400; i++) {
        for (int j = 0; j < 120; j++) {
            FFCW1[j+i*120]= fc1_w[i][j];
            //System.out.println("FFCW1 ["+j+"] "+FFCW1[j+0*120]+" ");
        }
    }
    CUdeviceptr deviceFC1par = new CUdeviceptr();
    cuMemAlloc(deviceFC1par, FFCpar1.length * Sizeof.DOUBLE);
    cuMemcpyHtoD(deviceFC1par, Pointer.to(FFCpar1),
            FFCpar1.length * Sizeof.DOUBLE);
    CUdeviceptr deviceFC1 = new CUdeviceptr();
    cuMemAlloc(deviceFC1, 120 * Sizeof.DOUBLE);
    cuMemcpyHtoD(deviceFC1, Pointer.to(FFC1), 120 * Sizeof.DOUBLE);

    CUdeviceptr deviceFCW1 = new CUdeviceptr();
    cuMemAlloc(deviceFCW1, 120*400 * Sizeof.DOUBLE);
    cuMemcpyHtoD(deviceFCW1, Pointer.to(FFCW1), 400*120 * Sizeof.DOUBLE);

    CUdeviceptr deviceFCb1 = new CUdeviceptr();
    cuMemAlloc(deviceFCb1, 120 * Sizeof.DOUBLE);
    cuMemcpyHtoD(deviceFCb1, Pointer.to(Layer2.fc1_b),
            120 * Sizeof.DOUBLE);
    Pointer FC1par=Pointer.to(
            Pointer.to(deviceFC1par),
            Pointer.to(deviceFCW1),//400*120
            Pointer.to(deviceFC1),//輸出 120
            Pointer.to(devicepool2),//輸入 16*5*5
            Pointer.to(deviceFCb1)
    );
    /**FC2**/

    int[] FFCpar2 ={120,84};
    double []FFC2=new double[84];
    double []FFCW2=new  double[120*84];
    for (int i = 0; i <120; i++) {
        for (int j = 0; j < 84; j++) {
            FFCW2[j+i*84]= fc2_w[i][j];
            //System.out.println("FFCW1 ["+j+" "+FFCW1[j+i*120]+" "+fc1_w[i][j]);
        }
    }

    CUdeviceptr deviceFC2par = new CUdeviceptr();
    cuMemAlloc(deviceFC2par, FFCpar2.length * Sizeof.DOUBLE);
    cuMemcpyHtoD(deviceFC2par, Pointer.to(FFCpar2),
            FFCpar2.length * Sizeof.DOUBLE);
    CUdeviceptr deviceFC2 = new CUdeviceptr();
    cuMemAlloc(deviceFC2, FFC2.length * Sizeof.DOUBLE);

    CUdeviceptr deviceFCW2 = new CUdeviceptr();
    cuMemAlloc(deviceFCW2, FFCW2.length * Sizeof.DOUBLE);
    cuMemcpyHtoD(deviceFCW2, Pointer.to(FFCW2),
            FFCW2.length * Sizeof.DOUBLE);
    CUdeviceptr deviceFCb2 = new CUdeviceptr();
    cuMemAlloc(deviceFCb2, Layer2.fc2_b.length* Sizeof.DOUBLE);
    cuMemcpyHtoD(deviceFCb2, Pointer.to(Layer2.fc2_b),
            Layer2.fc2_b.length * Sizeof.DOUBLE);
    Pointer FC2par=Pointer.to(
            Pointer.to(deviceFC2par),
            Pointer.to(deviceFCW2),//400*120
            Pointer.to(deviceFC2),//輸出 120
            Pointer.to(deviceFC1),//輸入 16*5*5
            Pointer.to(deviceFCb2)
    );
    /**FC3**/
    int[] FFCpar3 ={84,10};
    double []FFC3=new double[10];
    double []FFCW3=new  double[84*10];
    double []FFCb3=new  double[10];
    for (int i = 0; i <84; i++) {
        for (int j = 0; j < 10; j++) {
            FFCW3[j+i*10]= Layer2.fc3_w[i][j];
            //System.out.println("FFCW1 ["+j+" "+FFCW1[j+i*120]+" "+fc1_w[i][j]);
        }
    }
    CUdeviceptr deviceFC3par = new CUdeviceptr();
    cuMemAlloc(deviceFC3par, FFCpar3.length * Sizeof.DOUBLE);
    cuMemcpyHtoD(deviceFC3par, Pointer.to(FFCpar3),
            FFCpar3.length * Sizeof.DOUBLE);

    CUdeviceptr deviceFC3 = new CUdeviceptr();
    cuMemAlloc(deviceFC3, FFC3.length * Sizeof.DOUBLE);
    cuMemcpyHtoD(deviceFC3, Pointer.to(FFC3),
            FFC3.length * Sizeof.DOUBLE);

    CUdeviceptr deviceFCW3 = new CUdeviceptr();
    cuMemAlloc(deviceFCW3, FFCW3.length * Sizeof.DOUBLE);
    cuMemcpyHtoD(deviceFCW3, Pointer.to(FFCW3),
            FFCW3.length * Sizeof.DOUBLE);

    CUdeviceptr deviceFCb3 = new CUdeviceptr();
    cuMemAlloc(deviceFCb3, Layer2.fc3_b.length* Sizeof.DOUBLE);
    cuMemcpyHtoD(deviceFCb3, Pointer.to(Layer2.fc3_b),
            Layer2.fc3_b.length * Sizeof.DOUBLE);

    Pointer FC3par=Pointer.to(
            Pointer.to(deviceFC3par),
            Pointer.to(deviceFCW3),//400*120
            Pointer.to(deviceFC3),//輸出 120
            Pointer.to(deviceFC2),//輸入 16*5*5
            Pointer.to(deviceFCb3)
    );


    //cuMemAlloc(deviceW, 6*28*28 * Sizeof.DOUBLE);
    cudaEvent_t startsss = new cudaEvent_t();
    cudaEvent_t stopsss = new cudaEvent_t();
    float[] timesss = new float[1];
    cudaEventCreate(startsss);
    cudaEventCreate(stopsss);
    cudaEvent_t startsss2 = new cudaEvent_t();
    cudaEvent_t stopsss2 = new cudaEvent_t();
    float[] timesss2 = new float[1];
    cudaEventCreate(startsss2);
    cudaEventCreate(stopsss2);
    double[] deo=new double[6*28*28];
/*    double[]hosttestout=new double[36*36*6];

    double[] hostoutttttttttt=new double[5*36*6];
    double[] hostoutAM=new double[10*36*6];*/
/**i+6   3個stream**/


        //cuMemAlloc(deviceout, 36*36*6 * Sizeof.DOUBLE);
        //cudaHostAlloc(deviceout, 36*36 * Sizeof.DOUBLE,0);
        cuMemAlloc(devicetanout, 28*28*6 * Sizeof.DOUBLE);
        //cudaHostAlloc(devicetanout, 28*28*6 * Sizeof.DOUBLE,0);
        cuMemcpyHtoDAsync(deviceInputimage, Pointer.to(hostinputdata), 32*32 * Sizeof.DOUBLE, stream1);       //input Layer.Inputtest
        cuMemcpyHtoDAsync(deviceInputimage, Pointer.to(hostinputdata), 32*32 * Sizeof.DOUBLE, stream2);
        cuMemcpyHtoDAsync(deviceInputimage, Pointer.to(hostinputdata), 32*32 * Sizeof.DOUBLE, stream3);

        cuMemcpyHtoDAsync(deviceW, Pointer.to(hostwe), 25*6 * Sizeof.DOUBLE, stream1);
        cuMemcpyHtoDAsync(deviceB, Pointer.to(hostbias), 6 * Sizeof.DOUBLE, stream1);

        cuMemcpyHtoDAsync(deviceW, Pointer.to(hostwe), 25*6 * Sizeof.DOUBLE, stream2);
        cuMemcpyHtoDAsync(deviceB, Pointer.to(hostbias), 6 * Sizeof.DOUBLE, stream2);

        cuMemcpyHtoDAsync(deviceW, Pointer.to(hostwe), 25*6 * Sizeof.DOUBLE, stream3);
        cuMemcpyHtoDAsync(deviceB, Pointer.to(hostbias), 6 * Sizeof.DOUBLE, stream3);                     //從HOST複製權重係數到DEVICE(GPU)

        //cudaDeviceSetSharedMemConfig(2);
        //cudaDeviceGetSharedMemConfig();
        cuFuncSetCacheConfig(Pcore,0);

        CUevent pool1 = new CUevent();
        CUevent p1 = new CUevent();
        CUevent a1 = new CUevent();
        CUevent p2 = new CUevent();
        CUevent a2 = new CUevent();
        CUevent pool2 = new CUevent();
        cuEventCreate(pool1, 0);
        cuEventCreate(p1, 0);
        cuEventCreate(a1, 0);
        cuEventCreate(p2, 0);
        cuEventCreate(a2, 0);
        cuEventCreate(pool2, 0);
        int PgridDimx= 32;     //4
           /**開始運算卷積層1及池化層1 **/
        cuEventRecord(start, stream1);
        cudaEventRecord(startsss, null);

            cuMemcpyHtoDAsync(deviceP_1, Pointer.to(inpixrow[0 ]), 1 * Sizeof.DOUBLE, stream1);
            cuMemcpyHtoDAsync(deviceP_2, Pointer.to(inpixrow[1 ]), 1 * Sizeof.DOUBLE, stream2);
            cuMemcpyHtoDAsync(deviceP_3, Pointer.to(inpixrow[2 ]), 1 * Sizeof.DOUBLE, stream3);


            cuLaunchKernel(Pcore,                                   /**Pcore 第一個stage Acore 第二個stage Poolcore 第三個stage**/
            PgridDimx, 1, 1,      // Grid dimension
            25*6, 1, 1,      // Block dimension
            0, stream1,               // stream
            Ppar1, null // Kernel- and extra parameters
    );
            cuEventRecord(p1, null);/**插旗               null流要先等待前面其他CUDA流都運算完才會插旗**/
                                                            /**後面的所有CUDA流等待null流插旗完才開始平行運算**/
            cuLaunchKernel(Acore,
            6, 1, 1,      // Grid dimension
            28*6, 1, 1,      // Block dimension
            0, stream1,               //  stream
            Apar1, null // Kernel- and extra parameters
    );
            cuLaunchKernel(Pcore,
            PgridDimx, 1, 1,      // Grid dimension
            25*6 , 1, 1,      // Block dimension
            0, stream2,               //  stream
            Ppar2, null // Kernel- and extra parameters
    );
            cuEventRecord(p2, null);/**插旗**/


    for (int i = 0; i < 32 ; i+=3)   //26
            {
/** ****************  1111111111   ********************* **/
            cuMemcpyHtoDAsync(deviceP_3, Pointer.to(inpixrow[2 + i]), 1 * Sizeof.DOUBLE, stream3);/**更新要運算的資料的索引**/

                cuLaunchKernel(PoolCore,
                        6, 1, 1,      // Grid dimension
                        28, 1, 1,      // Block dimension
                        0, stream1,               //  stream
                        poolcorepar1, null // Kernel- and extra parameters
                );

                cuLaunchKernel(Acore,
                        6, 1, 1,      // Grid dimension
                        28*6, 1, 1,      // Block dimension
                        0, stream2,               //  stream
                        Apar2, null // Kernel- and extra parameters
                );
            cuLaunchKernel(Pcore,
                    PgridDimx, 1, 1,      // Grid dimension
                    25*6 , 1, 1,      // Block dimension
                    0, stream3,               // stream
                    Ppar3, null // Kernel- and extra parameters
            );

                cuEventRecord(a1, null);/**插旗**/

/** ****************  22222222222   *********************/
            cuMemcpyHtoDAsync(deviceP_1, Pointer.to(inpixrow[0 + i+3]), 1 * Sizeof.DOUBLE, stream1);
                cuLaunchKernel(Pcore,
                        PgridDimx, 1, 1,      // Grid dimension
                        25*6, 1, 1,      // Block dimension
                        0, stream1,               // stream
                        Ppar1, null // Kernel- and extra parameters
                );

                cuLaunchKernel(PoolCore,
                        6, 1, 1,      // Grid dimension
                        28, 1, 1,      // Block dimension
                        0, stream2,               // stream
                        poolcorepar2, null // Kernel- and extra parameters
                );

                cuLaunchKernel(Acore,
                        6, 1, 1,      // Grid dimension
                        28*6, 1, 1,      // Block dimension
                        0, stream3,               // stream
                        Apar3, null // Kernel- and extra parameters
                );
                cuEventRecord(a2, null);/**插旗**/


/** **************** 333333333333333  *********************/
                cuMemcpyHtoDAsync(deviceP_2, Pointer.to(inpixrow[1 + i+3]), 1 * Sizeof.DOUBLE, stream2);

                cuLaunchKernel(Acore,
                        6, 1, 1,      // Grid dimension
                        28*6, 1, 1,      // Block dimension
                        0, stream1,               //  stream
                        Apar1, null // Kernel- and extra parameters
                );
                cuLaunchKernel(Pcore,
                        PgridDimx, 1, 1,      // Grid dimension
                        25*6 , 1, 1,      // Block dimension
                        0, stream2,               //  stream
                        Ppar2, null // Kernel- and extra parameters
                );

            cuLaunchKernel(PoolCore,
                    6, 1, 1,      // Grid dimension
                    28, 1, 1,      // Block dimension
                    0, stream3,               //  stream
                    poolcorepar3, null // Kernel- and extra parameters
            );

                cuEventRecord(pool1, null);/**插旗**/

        }//for迴圈結束

        cuEventRecord(stop, stream1);
        cuEventSynchronize(stop);
        cuEventElapsedTime(time,start,stop);


    double []C2out = new double[10*10*16];

    /**開始運算捲基層2及池化層2**/

    {
            cuMemcpyHtoDAsync(deviceW2, Pointer.to(host2we), 5*5*6*16 * Sizeof.DOUBLE, stream1);
            cuMemcpyHtoDAsync(deviceB2, Pointer.to(host2bias), 16 * Sizeof.DOUBLE, stream1);

            cuMemcpyHtoDAsync(deviceW2, Pointer.to(host2we), 5*5*6*16 * Sizeof.DOUBLE, stream2);
            cuMemcpyHtoDAsync(deviceB2, Pointer.to(host2bias), 16 * Sizeof.DOUBLE, stream2);

            cuMemcpyHtoDAsync(deviceW2, Pointer.to(host2we), 5*5*6*16 * Sizeof.DOUBLE, stream3);
            cuMemcpyHtoDAsync(deviceB2, Pointer.to(host2bias), 16 * Sizeof.DOUBLE, stream3);

        CUevent pa1 = new CUevent();
        CUevent ac1 = new CUevent();
        CUevent pa2 = new CUevent();
        CUevent ac2 = new CUevent();
        CUevent plll1 = new CUevent();
        CUevent plll2 = new CUevent();
        CUevent plll3 = new CUevent();
        cuEventCreate(pa1, 0);
        cuEventCreate(ac1, 0);
        cuEventCreate(pa2, 0);
        cuEventCreate(ac2, 0);
        cuEventCreate(plll1, 0);
        cuEventCreate(plll2, 0);
        cuEventCreate(plll3, 0);
        int pg=14;//14
        int pd=5*5*16;
            /**開始運算卷積層2 **/
            cudaEventRecord(startsss2,null);
            //cudaProfilerStart();
        cuMemcpyHtoDAsync(deviceP2_1, Pointer.to(inpixrow2[0 ]), 1 * Sizeof.DOUBLE,stream1);
        cuMemcpyHtoDAsync(deviceP2_2, Pointer.to(inpixrow2[1 ]), 1 * Sizeof.DOUBLE,stream2);
        cuMemcpyHtoDAsync(deviceP2_3, Pointer.to(inpixrow2[2 ]), 1 * Sizeof.DOUBLE,stream3);

        cuLaunchKernel(P2core,
                pg, 1, 1,      // Grid dimension
                pd, 1, 1,      // Block dimension
                0, stream1,               //  stream
                Ppar_21, null // Kernel- and extra parameters
        );
        cuEventRecord(pa1, null);
        cuLaunchKernel(P2core,
                pg, 1, 1,      // Grid dimension
                pd, 1, 1,      // Block dimension
                0, stream2,               //  stream
                Ppar_22, null // Kernel- and extra parameters
        );
        cuLaunchKernel(A2core,
                16, 1, 1,      // Grid dimension
                10*6, 1, 1,      // Block dimension   5*5*6*16
                0, stream1,               //  stream
                Apar_21, null // Kernel- and extra parameters
        );
        cuEventRecord(pa2, null);
        for (int i = 0, k2 = -1; i < 15; i+=3)
            {
                /*************1111111111111111******************/
                //cuStreamWaitEvent(stream2, pa1, 0);
                //cuStreamWaitEvent(stream3, pa2, 0);
                cuMemcpyHtoDAsync(deviceP2_3, Pointer.to(inpixrow2[2+i]), 1 * Sizeof.DOUBLE,stream3);
                cuLaunchKernel(Pool2core,
                        16, 1, 1,      // Grid dimension
                        10, 1, 1,      // Block dimension
                        0, stream1,               //  stream
                        pool2corepar1, null // Kernel- and extra parameters
                );
                cuLaunchKernel(A2core,
                        16, 1, 1,      // Grid dimension
                        10*6, 1, 1,      // Block dimension   5*5*6*16
                        0, stream2,               //  stream
                        Apar_22, null // Kernel- and extra parameters
                );

                cuLaunchKernel(P2core,
                        pg, 1, 1,      // Grid dimension
                        pd, 1, 1,      // Block dimension
                        0, stream3,               //  stream
                        Ppar_23, null // Kernel- and extra parameters
                );
                cuEventRecord(ac1, null);



                /**************************2222222***********************************************/




                //cuStreamWaitEvent(stream2, ac1, 0);
                cuMemcpyHtoDAsync(deviceP2_1, Pointer.to(inpixrow2[0+i+3]), 1 * Sizeof.DOUBLE,stream1);
                cuLaunchKernel(P2core,
                        pg, 1, 1,      // Grid dimension
                        pd, 1, 1,      // Block dimension
                        0, stream1,               //  stream
                        Ppar_21, null // Kernel- and extra parameters
                );
                cuLaunchKernel(Pool2core,
                        16, 1, 1,      // Grid dimension
                        10, 1, 1,      // Block dimension
                        0, stream2,               //  stream
                        pool2corepar2, null // Kernel- and extra parameters
                );

                cuLaunchKernel(A2core,
                        16, 1, 1,      // Grid dimension
                        10*6, 1, 1,      // Block dimension   5*5*6*16
                        0, stream3,               //  stream
                        Apar_23, null // Kernel- and extra parameters
                );
                cuEventRecord(ac2, null);

                //cuStreamWaitEvent(stream3, ac2, 0);








                /******************************3333***********************************************/



                //cuStreamWaitEvent(stream2, plll1, 0);
                cuMemcpyHtoDAsync(deviceP2_2, Pointer.to(inpixrow2[1+i+3 ]), 1 * Sizeof.DOUBLE,stream2);
                cuLaunchKernel(A2core,
                        16, 1, 1,      // Grid dimension
                        10*6, 1, 1,      // Block dimension   5*5*6*16
                        0, stream1,               //  stream
                        Apar_21, null // Kernel- and extra parameters
                );
                cuLaunchKernel(P2core,
                        pg, 1, 1,      // Grid dimension
                        pd, 1, 1,      // Block dimension
                        0, stream2,               //  stream
                        Ppar_22, null // Kernel- and extra parameters
                );
                cuLaunchKernel(Pool2core,
                        16, 1, 1,      // Grid dimension
                        10, 1, 1,      // Block dimension
                        0, stream3,               // stream
                        pool2corepar3, null // Kernel- and extra parameters
                );
                cuEventRecord(plll1, null);



                //cuStreamWaitEvent(stream3, plll2, 0);


                //cuStreamWaitEvent(stream1, ac2, 0);
                }

            cudaEventRecord(stopsss2,null);
            cudaEventSynchronize(stopsss2);
            cudaEventElapsedTime(timesss2,startsss2,stopsss2);

 /*           cuMemcpyDtoHAsync(Pointer.to(C2out), devicetanout_2, 10*10*16 * Sizeof.DOUBLE,stream5);


        for(int kk=0; kk<16;kk++){
            for (int ii = 0; ii < 10; ii++) {
                for (int jj = 0; jj < 10; jj++) {
                    if (jj == 0) System.out.print("C2out" + kk + "[" + ii + "][" + jj + "]= ");
                    System.out.print(C2out[jj + ii * 10+kk*10*10] + " ");
                    if (jj == 10 - 1) System.out.println();
                    if (ii==9&&jj == 10 - 1) System.out.println("\n\n");
                }
            }
        }
*/
    }/////////////純括弧





    double[]hostpool2=new double[5*5*16];

 //cuMemcpyDtoH(Pointer.to(hostpool2),devicepool2,5*5*16* Sizeof.DOUBLE);
    cudaEventRecord(stopsss, null);
    cudaEventSynchronize(stopsss);
    cudaEventElapsedTime(timesss,startsss,stopsss);

    double[] Fc1=new double[120];
    double[] Fc2=new double[84];
    double[] Fc3=new double[10];
    double[] out=new double[10];
    /////////////////////////////////////////////////////////////////////////////////////


    time1 = System.nanoTime();
    /**開始運算全連結層**/

    cuLaunchKernel(FC,
            1, 1, 1,      // Grid dimension
            120, 1, 1,      // Block dimension
            0, null,               // stream
            FC1par, null // Kernel- and extra parameters
    );
    cuLaunchKernel(FC,
            1, 1, 1,      // Grid dimension
            84, 1, 1,      // Block dimension
            0, null,               // stream
            FC2par, null // Kernel- and extra parameters
    );
    cuLaunchKernel(FC3,
            1, 1, 1,      // Grid dimension
            10, 1, 1,      // Block dimension
            0, null,               // stream
            FC3par, null // Kernel- and extra parameters
    );

    ////////////////////////////////////////////////////////////////////////////
    cuMemcpyDtoH(Pointer.to(out),deviceFC3,10* Sizeof.DOUBLE);

    time22 = System.nanoTime();
    System.out.println((double) (time22-time1)/1000000+" nssssss  ");/**三個全連結層總花費時間**/

    for (int j = 0; j < 10; j++) {
        System.out.print("FFFFFFFC3["+ (j)+"]=");
        System.out.println(" "+out[j]+" ");
        if(j%10==9) System.out.print("\n");

    }
    //FC3 END
    int num=0;
    double max=0;
    for(int i=0;i<10;i++)
    {
        if(out[i]>max)
        {
            num = i;
            max=out[i];
        }
    }

/**得到運算時間的結果**/
    System.out.println("the number is "+num);

    System.out.println("NEW Conv1time= "+time[0]+"ms");/**第一個流水縣**/
    System.out.println("NEW Conv2time= "+timesss2[0]+"ms");/**第二個流水縣**/
    System.out.println("NEW LeNet total time= "+(time[0]+timesss2[0]+(double)(time22-time1)/1000000)+"ms");
/**釋放GPU記憶體**/
    cuMemFree(deviceout2);
    cuMemFree(devicePout);
    cuMemFree(deviceout);
    cuMemFree(devicetanout);
    cuMemFree(deviceW);
    cuMemFree(deviceInputimage);
    //cuMemFree(deviceInputrow);
    cuMemFree(deviceP_1);
    cuMemFree(deviceP_2);
    cuMemFree(deviceP_3);
    cuMemFree(deviceB);
    cuMemFree(AM1);
    cuMemFree(AM2);
    cuMemFree(AM3);
    cuMemFree(AM4);
    cuMemFree(AM5);
    cuMemFree(AM6);


    //cuMemFreeHost(deviceW2);
    cuMemFree(deviceW2);
    cuMemFree(deviceB2);

    //cuMemFree(deviceK2);
    cuMemFree(deviceout2_2);
    //uMemFree(deviceInputrow2);
    cuMemFree(devicetanout_2);
    cuMemFree(AM1_2);
    cuMemFree(AM2_2);
    cuMemFree(AM3_2);
    cuMemFree(AM4_2);
    cuMemFree(AM5_2);
    cuMemFree(AM6_2);

    cuMemFree(devicepool);
    //cuMemFree(deviceparP1);
    cuMemFree(devicepoolindex);
    cuMemFree(devicepool2);
    cuMemFree(deviceparP2);



    cuMemFree(deviceFC1par);
    cuMemFree(deviceFC1);
    cuMemFree(deviceFCW1);
    cuMemFree(deviceFCb1);

    cuMemFree(deviceFC2par);
    cuMemFree(deviceFC2);
    cuMemFree(deviceFCW2);
    cuMemFree(deviceFCb2);

    cuMemFree(deviceFC3par);
    cuMemFree(deviceFC3);
    cuMemFree(deviceFCW3);
    cuMemFree(deviceFCb3);





    cudaDeviceSynchronize();






    cuCtxDestroy(context);
}




}
