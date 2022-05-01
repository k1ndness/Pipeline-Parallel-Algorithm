package org.example;

public class MatrixThread implements Runnable {


    public int i,j;

    @Override

    public void run() {

        // TODO Auto-generated method stub

        for(int k = 0; k <3; k++) {
            for(int l=0; l<3;l++) {

                //pipeline.asn[i][j] = pipeline.asn[i][j] + Matrix.a[k+i][l+j] * Matrix.b[k][l]; /*陣列A乘上陣列B,存入陣列C */
            }
        }

       //System.out.println("Main.asn"+"["+ i + "]" + "[" + j + "]=>"+pipeline.asn[i][j]);

    }

}
