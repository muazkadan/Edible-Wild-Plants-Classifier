package net.mouazkaadan.seniordesignprojecti;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if(OpenCVLoader.initDebug())
            Toast.makeText(MainActivity.this, "OpenCV Loaded Successfully", Toast.LENGTH_SHORT).show();
        else
            Toast.makeText(MainActivity.this, "Something Went Wrong While Loading OpenCV", Toast.LENGTH_SHORT).show();

    }
}