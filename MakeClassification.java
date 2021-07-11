/*
Author: Tomasz Hachaj, 2021
Department of Signal Processing and Pattern Recognition
Institute of Computer Science in Pedagogical University of Krakow, Poland
https://sppr.up.krakow.pl/hachaj/
Data source:
https://credo.nkg-mn.com/hits.html
*/
package calculatedistances;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 *
 * @author Tomek
 */
public class MakeClassification {
        public static void LoadData(String csvFile, String csvFileOut, String cvsSplitBy, boolean debug)
    {
        String line = "";
        int line_id = 0;
        ArrayList data_al = new ArrayList();
        boolean has_header = false;
        
        String header = "id,file,dot,line,worm,artifact,class,class50,class75,class85";
        
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            try (BufferedWriter bw = new BufferedWriter(new FileWriter(csvFileOut))) {
                bw.write(header + "\n");
                while ((line = br.readLine()) != null) {
                    if (debug)
                        System.out.println(line_id);
                    // use comma as separator
                    String[] split_line = line.split(cvsSplitBy);
                    double max = Double.parseDouble(split_line[2]);
                    int max_id = 0;
                    double max_help = 0;
                    for (int a = 0; a < 3; a++)
                    {
                        max_help = Double.parseDouble(split_line[a + 3]);
                        if (max_help > max)
                        {
                            max_id = a + 1;
                            max = max_help;
                        }
                    }
                    line += "," + max_id;
                    if (max > 0.5)
                    {
                        line += "," + max_id;
                    } else
                    {
                        line += ",NA";
                    }
                    if (max > 0.75)
                    {
                        line += "," + max_id;
                    } else
                    {
                        line += ",NA";
                    }
                    if (max > 0.85)
                    {
                        line += "," + max_id;
                    } else
                    {
                        line += ",NA";
                    }
                    bw.write(line + "\n");

                    //System.out.println("Country [code= " + country[4] + " , name=" + country[5] + "]");
                }
            }
        } catch (IOException e) {
            return;
            //e.printStackTrace();
        }
        return;
    }
    
    public static void main(String[] args)
    {
        /*
        LoadData("d:\\Projects\\Python\\PycharmProjects\\DLIB_Pytorch_Kropki\\my_vgg\\v2\\results\\all\\99.zip.txt",
                "d:\\Projects\\Python\\PycharmProjects\\DLIB_Pytorch_Kropki\\my_vgg\\v2\\results\\all_class\\99.zip.txt",
                ",", false);
        */
        String my_path_in = "d:\\Projects\\Python\\PycharmProjects\\DLIB_Pytorch_Kropki\\my_vgg\\v2\\results\\all\\";
        String my_path_out = "d:\\Projects\\Python\\PycharmProjects\\DLIB_Pytorch_Kropki\\my_vgg\\v2\\results\\all_class\\";
        File folder = new File(my_path_in);
        File[] listOfFiles = folder.listFiles();
        for (File file : listOfFiles) {
            if (file.isFile()) {
                //System.out.println(file.getName());
                LoadData(my_path_in + file.getName(),
                my_path_out + file.getName(),
                ",", false);
            }
        }
    }
}
