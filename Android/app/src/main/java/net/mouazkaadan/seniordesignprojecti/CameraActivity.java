package net.mouazkaadan.seniordesignprojecti;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class CameraActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private JavaCameraView javaCameraView;
    private Mat mat1;
    private Button classifyButton, probabilitiesButton;
    private TextView classTextView;
    private Interpreter tflite;
    private List<String> labels;
    private int[] imageShape, probabilityShape;
    private DataType imageDataType, probabilityDataType;
    private TensorImage inputImageBuffer;
    private int imageSizeY;
    private int imageSizeX;
    private TensorBuffer outputProbabilityBuffer;
    private TensorProcessor probabilityProcessor;
    private PriorityQueue<Recognition> priorityQueue;


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i("", "OpenCV loaded successfully");
                    javaCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        getSupportActionBar().hide();
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        OpenCVLoader.initDebug();

        javaCameraView = findViewById(R.id.java_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);

        classTextView = findViewById(R.id.class_textView);
        classifyButton = findViewById(R.id.classify_button);
        probabilitiesButton = findViewById(R.id.probabilities_button);

        try {
            labels = FileUtil.loadLabels(this, "labels.txt");
            tflite = new Interpreter(loadModel());
        } catch (IOException e) {
            e.printStackTrace();
        }

        imageShape = tflite.getInputTensor(0).shape();
        imageDataType = tflite.getInputTensor(0).dataType();
        probabilityDataType = tflite.getOutputTensor(0).dataType();
        probabilityShape =
                tflite.getOutputTensor(0).shape();
        inputImageBuffer = new TensorImage(imageDataType);
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];
        probabilityProcessor = new TensorProcessor.Builder().add(new NormalizeOp(127.5f, 127.5f)).build();

        classifyButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Bitmap bitmap = Bitmap.createBitmap(mat1.width(), mat1.height(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(mat1, bitmap);
                ImageProcessor imageProcessor =
                        new ImageProcessor.Builder()
                                .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                                .build();
                inputImageBuffer.load(bitmap);
                inputImageBuffer = imageProcessor.process(inputImageBuffer);
                tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
                Map<String, Float> labeledProbability =
                        new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                                .getMapWithFloatValue();
                classTextView.setText("");
                priorityQueue =
                        new PriorityQueue<>(
                                1,
                                new Comparator<Recognition>() {
                                    @Override
                                    public int compare(Recognition lhs, Recognition rhs) {
                                        return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                                    }
                                });
                for (Map.Entry<String, Float> entry : labeledProbability.entrySet()) {
                    priorityQueue.add(new Recognition(entry.getKey(), entry.getValue()));
                }
                Recognition recognition = priorityQueue.peek();
                classTextView.setText("Class: " + recognition.getTitle() + "  \nAccuracy: " + String.format("%.2f", 100 + (100 * recognition.getConfidence())));
                probabilitiesButton.setVisibility(View.VISIBLE);
            }
        });

        probabilitiesButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                classTextView.setText("");
                while(!priorityQueue.isEmpty()){
                    Recognition recognition = priorityQueue.poll();
                    classTextView.setText(classTextView.getText() + "Class: " + recognition.getTitle() + " / Accuracy: " + String.format("%.2f", 100 + (100 * recognition.getConfidence())) + "\n");
                }
            }
        });
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mat1 = new Mat(width, height, CvType.CV_8UC4);
    }

    @Override
    public void onCameraViewStopped() {
        mat1.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mat1 = inputFrame.rgba();
        return mat1;
    }

    private MappedByteBuffer loadModel() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("converted_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (javaCameraView != null)
            javaCameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (javaCameraView != null)
            javaCameraView.disableView();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug())
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        else
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
    }
}