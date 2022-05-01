package org.example;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class Reading2DArrayFromFile {
    public static double[][] fc1_w = new double[400][120];
    public static double[][] fc2_w= new double[120][84];
    static double[][] writeToDat(String path) {
        File file = new File("FCwb2.txt");
        List<String> list = new ArrayList<String>();
       // double[][] nums = null;
        try {
            BufferedReader bw = new BufferedReader(new FileReader(file));
            String line = null;
            // 因為不知道有幾行資料,所以先存入list集合中
            while ((line = bw.readLine()) != null) {
                list.add(line);
            }
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        // 確定陣列長度
        //nums = new double[400][120];
        int k=0;
        for (int i = 0; i < 400; i++) {
            for (int j = 0; j < 120; j++) {
                String s = list.get(k);
                fc1_w[i][j] = Double.parseDouble(s);
                k++;
            }
        }
        return fc1_w;
    }
    /** 1/16  **/
    static double[][] writeToDat2(String path) {
        File file = new File("fcw2.txt");
        List<String> list = new ArrayList<String>();
        // double[][] nums = null;
        try {
            BufferedReader bw = new BufferedReader(new FileReader(file));
            String line = null;
            // 因為不知道有幾行資料,所以先存入list集合中
            while ((line = bw.readLine()) != null) {
                list.add(line);
            }
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        // 確定陣列長度
        //nums = new double[400][120];
        int k=0;
        for (int i = 0; i < 120; i++) {
            for (int j = 0; j < 84; j++) {
                String s = list.get(k);
                fc2_w[i][j] = Double.parseDouble(s);
                k++;
            }
        }
        return fc2_w;
    }


    public static void main(String args[]) throws Exception {
        String path = "FCwb2.txt";
        fc1_w = writeToDat(path);
        for (int i = 0; i < fc1_w.length; i++) {
            for (int j = 0; j < fc1_w[0].length; j++) {
                if (j==0)  System.out.print("FCW1[" + i + "][" + j + "]=");
                System.out.print(fc1_w[i][j]+" ");
                if (j==119) System.out.print("\n");
            }
        }
    }
}
