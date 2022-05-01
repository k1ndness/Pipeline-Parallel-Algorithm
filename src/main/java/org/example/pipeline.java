package org.example;
import java.io.File;
import java.io.IOException;
import java.util.Date;
public class pipeline {



    public static void main(String args[]) {


        LeNet zxc = new LeNet();
        LeNet.InputImage=Inputimage.one;                            //輸入圖
        zxc.Import_weight();
        zxc.GPU_parameters(LeNet.InputImage,LeNet.turn, LeNet.turn2);//準備輸入數據給GPU做使用
        long time,time2;
        time = System.nanoTime();
        zxc.Conv1();                                                //CPU開始運算LeNet第一層
        zxc.Pooling1();
        zxc.Conv2();
        zxc.Pooling2();
        zxc.FullyConnect1();
        zxc.FullyConnect2();
        zxc.FullyConnect3();
        time2 = System.nanoTime();
        zxc.Result_of_classification();
        System.out.println((time2-time)+" ns  ");
        zxc.NEW_LeNet();                                            //流水線平行演算法

    }//main
}



















/*




*/









