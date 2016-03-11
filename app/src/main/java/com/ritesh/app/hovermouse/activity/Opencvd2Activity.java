

/*
This code to go into CameraBridgeViewBase in 
order to scale the bitmap to the size of the screen
 
The file is at:
 
[your path to the sdk]\OpenCV-2.4.4-android-sdk\sdk\java\src\org\opencv\android

   protected void deliverAndDrawFrame(CvCameraViewFrame frame) {
       Mat modified;

       if (mListener != null) {
           modified = mListener.onCameraFrame(frame);
       } else {
           modified = frame.rgba();
       }

       boolean bmpValid = true;
       if (modified != null) {
           try {
               Utils.matToBitmap(modified, mCacheBitmap);
           } catch(Exception e) {
               Log.e(TAG, "Mat type: " + modified);
               Log.e(TAG, "Bitmap type: " + mCacheBitmap.getWidth() + "*" + mCacheBitmap.getHeight());
               Log.e(TAG, "Utils.matToBitmap() throws an exception: " + e.getMessage());
               bmpValid = false;
           }
       }

       if (bmpValid && mCacheBitmap != null) {
           Canvas canvas = getHolder().lockCanvas();
           if (canvas != null) {
               canvas.drawColor(0, android.graphics.PorterDuff.Mode.CLEAR);
               
               
               /////////////////////////////////////////////////////
               ////// THIS IS THE CHANGED PART /////////////////////
               int width = mCacheBitmap.getWidth();
               int height = mCacheBitmap.getHeight();
               float scaleWidth = ((float) canvas.getWidth()) / width;
               float scaleHeight = ((float) canvas.getHeight()) / height;
               float fScale = Math.min(scaleHeight,  scaleWidth);
               // CREATE A MATRIX FOR THE MANIPULATION
               Matrix matrix = new Matrix();
               // RESIZE THE BITMAP
               matrix.postScale(fScale, fScale);

               /////////////////////////////////////////////////////

               // RECREATE THE NEW BITMAP
               Bitmap resizedBitmap = Bitmap.createBitmap(mCacheBitmap, 0, 0, width, height, matrix, false);
               
               canvas.drawBitmap(resizedBitmap, (canvas.getWidth() - resizedBitmap.getWidth()) / 2, (canvas.getHeight() - resizedBitmap.getHeight()) / 2, null);
               if (mFpsMeter != null) {
                   mFpsMeter.measure();
                   mFpsMeter.draw(canvas, 20, 30);
               }
               getHolder().unlockCanvasAndPost(canvas);
           }
       }
   }



*
*
*/

package com.ritesh.app.hovermouse.activity;

import android.app.Activity;
import android.content.Context;
import android.hardware.Camera;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout.LayoutParams;
import android.widget.Toast;

import com.ritesh.app.hovermouse.R;
import com.ritesh.app.hovermouse.activity.ServerUtils.Constants;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.video.Video;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;

public class Opencvd2Activity extends Activity implements CvCameraViewListener {

    private CascadeClassifier mCascade;

    private Point pt, pt2;

    private int x, y, iLineThickness = 3, iNumberOfCameras = 0, iGFFTMax = 40;

    private JavaCameraView mOpenCvCameraView0;
    private JavaCameraView mOpenCvCameraView1;

    private List<Byte> byteStatus;
    private List<Point> cornersThis, cornersPrev;

    private long lMilliStart = 0, lMilliNow = 0;

    private Mat mRgba, mErodeKernel, matOpFlowPrev, matOpFlowThis;

    private MatOfFloat mMOFerr;
    private MatOfByte mMOBStatus;
    private MatOfPoint2f mMOP2fptsPrev, mMOP2fptsThis, mMOP2fptsSafe;
    private MatOfPoint MOPcorners;

    private Scalar colorRed;
    private Size sSize3, sMatSize;

    private int dx = 0, dy = 0;//coordinates to be sent to server
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    mOpenCvCameraView0.enableView();

                    if (iNumberOfCameras > 1)
                        mOpenCvCameraView1.enableView();

                    try {
                        // DO FACE CASCADE SETUP

                        Context context = getApplicationContext();
                        InputStream is3 = context.getResources().openRawResource(R.raw.haarcascade_frontalface_default);
                        File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
                        File cascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");

                        FileOutputStream os = new FileOutputStream(cascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;

                        while ((bytesRead = is3.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }

                        is3.close();
                        os.close();

                        mCascade = new CascadeClassifier(cascadeFile.getAbsolutePath());

                        if (mCascade.empty()) {
                            //Log.d(TAG, "Failed to load cascade classifier");
                            mCascade = null;
                        }

                        cascadeFile.delete();
                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        // Log.d(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };
    //server program
    private boolean isConnected = false;
    private Socket socket;
    private PrintWriter out;
    Context context;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        iNumberOfCameras = Camera.getNumberOfCameras();

        //Log.d(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED);
        setContentView(R.layout.opencvd2_with_button);


        context = this;
        mOpenCvCameraView0 = (JavaCameraView) findViewById(R.id.java_surface_view0);

        if (iNumberOfCameras > 1)
            mOpenCvCameraView1 = (JavaCameraView) findViewById(R.id.java_surface_view1);

        mOpenCvCameraView0.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView0.setCvCameraViewListener(this);

        mOpenCvCameraView0.setLayoutParams(new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));

        if (iNumberOfCameras > 1) {
            mOpenCvCameraView1.setVisibility(SurfaceView.GONE);
            mOpenCvCameraView1.setCvCameraViewListener(this);
            mOpenCvCameraView1.setLayoutParams(new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));
        }

        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
        Button btLeftClick = (Button) findViewById(R.id.btLeftClick);
        btLeftClick.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {
                switch (motionEvent.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        //isButtonPressed=true;
                        if (out != null && isConnected) {
                            //while(isButtonPressed){
                            out.println("left_down");
                            //}
                        }

                        break;
                    case MotionEvent.ACTION_UP:
                        //isButtonPressed =false;
                        if (out != null && isConnected)
                            out.println("left_up");
                        break;
                }
                return false;
            }
        });
    }


    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView0 != null)
            mOpenCvCameraView0.disableView();
        if (iNumberOfCameras > 1)
            if (mOpenCvCameraView1 != null)
                mOpenCvCameraView1.disableView();
    }


    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
    }


    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView0 != null)
            mOpenCvCameraView0.disableView();
        if (iNumberOfCameras > 1)
            if (mOpenCvCameraView1 != null)
                mOpenCvCameraView1.disableView();

        if (isConnected && out != null) {
            try {
                out.println("exit"); //tell server to exit
                socket.close(); //close socket
            } catch (IOException e) {
                Log.e("remotedroid", "Error in closing socket", e);
            }
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.opencvd2, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        /*
        if (item.getItemId() == R.id.action_swapcamera) {
            if (iNumberOfCameras > 1) {
                if (iCamera == 0) {
                    mOpenCvCameraView0.setVisibility(SurfaceView.GONE);
                    mOpenCvCameraView1 = (JavaCameraView) findViewById(R.id.java_surface_view1);
                    mOpenCvCameraView1.setCvCameraViewListener(this);
                    mOpenCvCameraView1.setVisibility(SurfaceView.VISIBLE);

                    iCamera = 1;
                } else {
                    mOpenCvCameraView1.setVisibility(SurfaceView.GONE);
                    mOpenCvCameraView0 = (JavaCameraView) findViewById(R.id.java_surface_view0);
                    mOpenCvCameraView0.setCvCameraViewListener(this);
                    mOpenCvCameraView0.setVisibility(SurfaceView.VISIBLE);

                    iCamera = 0;
                }
            } else
                Toast.makeText(getApplicationContext(), "Sadly, your device does not have a second camera",
                        Toast.LENGTH_LONG).show();
        }
*/
        return true;
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        // TODO Auto-generated method stub
        byteStatus = new ArrayList<>();

        colorRed = new Scalar(255, 0, 0, 255);
        cornersThis = new ArrayList<>();
        cornersPrev = new ArrayList<>();

        mMOP2fptsPrev = new MatOfPoint2f();
        mMOP2fptsThis = new MatOfPoint2f();
        mMOP2fptsSafe = new MatOfPoint2f();
        mMOFerr = new MatOfFloat();
        mMOBStatus = new MatOfByte();
        MOPcorners = new MatOfPoint();
        mRgba = new Mat();
        matOpFlowThis = new Mat();
        matOpFlowPrev = new Mat();

        pt = new Point(0, 0);
        pt2 = new Point(0, 0);

        sMatSize = new Size();
        sSize3 = new Size(3, 3);

        mRgba = new Mat(height, width, CvType.CV_8UC4);

    }

    @Override
    public void onCameraViewStopped() {
        releaseMats();
    }

    public void releaseMats() {
        mRgba.release();
        if (mErodeKernel != null)
            mErodeKernel.release();
        MOPcorners.release();

    }

    @Override
    public Mat onCameraFrame(Mat inputFrame) {
        mErodeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, sSize3);

        // start the timing counter to put the framerate on screen
        // and make sure the start time is up to date, do
        // a reset every 10 seconds
        if (lMilliStart == 0)
            lMilliStart = System.currentTimeMillis();

        if ((lMilliNow - lMilliStart) > 10000) {
            lMilliStart = System.currentTimeMillis();
        }

        inputFrame.copyTo(mRgba);
        sMatSize.width = mRgba.width();
        sMatSize.height = mRgba.height();

        if (mMOP2fptsPrev.rows() == 0) {

            //Log.d("Baz", "First time opflow");
            // first time through the loop so we need prev and this mats
            // plus prev points
            // get this mat
            Imgproc.cvtColor(mRgba, matOpFlowThis, Imgproc.COLOR_RGBA2GRAY);

            // copy that to prev mat
            matOpFlowThis.copyTo(matOpFlowPrev);

            // get prev corners
            Imgproc.goodFeaturesToTrack(matOpFlowPrev, MOPcorners, iGFFTMax, 0.05, 20);
            mMOP2fptsPrev.fromArray(MOPcorners.toArray());

            // get safe copy of this corners
            mMOP2fptsPrev.copyTo(mMOP2fptsSafe);
        } else {
            //Log.d("Baz", "Opflow");
            // we've been through before so
            // this mat is valid. Copy it to prev mat
            matOpFlowThis.copyTo(matOpFlowPrev);

            // get this mat
            Imgproc.cvtColor(mRgba, matOpFlowThis, Imgproc.COLOR_RGBA2GRAY);

            // get the corners for this mat
            Imgproc.goodFeaturesToTrack(matOpFlowThis, MOPcorners, iGFFTMax, 0.05, 20);
            mMOP2fptsThis.fromArray(MOPcorners.toArray());

            // retrieve the corners from the prev mat
            // (saves calculating them again)
            mMOP2fptsSafe.copyTo(mMOP2fptsPrev);

            // and save this corners for next time through

            mMOP2fptsThis.copyTo(mMOP2fptsSafe);
        }

        	
           	/*
               Parameters:
           		prevImg first 8-bit input image
           		nextImg second input image
           		prevPts vector of 2D points for which the flow needs to be found; point coordinates must be single-precision floating-point numbers.
           		nextPts output vector of 2D points (with single-precision floating-point coordinates) containing the calculated new positions of input features in the second image; when OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
           		status output status vector (of unsigned chars); each element of the vector is set to 1 if the flow for the corresponding features has been found, otherwise, it is set to 0.
           		err output vector of errors; each element of the vector is set to an error for the corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't found then the error is not defined (use the status parameter to find such cases).
            */
        Video.calcOpticalFlowPyrLK(matOpFlowPrev, matOpFlowThis, mMOP2fptsPrev, mMOP2fptsThis, mMOBStatus, mMOFerr);

        cornersPrev = mMOP2fptsPrev.toList();
        cornersThis = mMOP2fptsThis.toList();
        byteStatus = mMOBStatus.toList();

        y = byteStatus.size() - 1;
        dx = dy = 0;
        int counter = 0;
        for (x = 0; x < y; x++) {
            if (byteStatus.get(x) == 1) {
                counter++;
                pt = cornersThis.get(x);
                pt2 = cornersPrev.get(x);
                dx = dx - ((int) pt2.x - (int) pt.x);
                dy = dy - ((int) pt2.y - (int) pt.y);
                Imgproc.circle(mRgba, pt, 5, colorRed, iLineThickness - 1);

                Imgproc.line(mRgba, pt, pt2, colorRed, iLineThickness);
            }
        }
        if (counter != 0) {
            dx /= counter;
            dy /= counter;
        } else {
            dx = dy = 0;
        }
        if (isConnected && out != null) {
            //Point point = new Point(dx,dy);
            dy
                    =dy*1366/720;
            dx=dx*720/480;
            out.println(dx + " " + dy);//send "play" to server
        }
        Log.d(TAG, "dx= " + dx + "    dy= " + dy);


        //Log.d("Baz", "Opflow feature count: "+x);


        // get the time now in every frame
        lMilliNow = System.currentTimeMillis();

        return mRgba;
    }

    String TAG = "OpencvActivity";
/*
    public boolean onTouchEvent(final MotionEvent event) {

        return false; // don't need more than one touch event

    }*/


    public class ConnectPhoneTask extends AsyncTask<String, Void, Boolean> {

        @Override
        protected Boolean doInBackground(String... params) {
            boolean result = true;
            try {
                InetAddress serverAddr = InetAddress.getByName(params[0]);
                socket = new Socket(serverAddr, Constants.SERVER_PORT);//Open socket on server IP and port
            } catch (IOException e) {
                Log.e("remotedroid", "Error while connecting", e);
                result = false;
            }
            return result;
        }

        @Override
        protected void onPostExecute(Boolean result) {
            isConnected = result;
            Toast.makeText(context, isConnected ? "Connected to server!" : "Error while connecting", Toast.LENGTH_LONG).show();
            try {
                if (isConnected) {
                    out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(socket
                            .getOutputStream())), true); //create output stream to send data to server
                }
            } catch (IOException e) {
                Log.e("remotedroid", "Error while creating OutWriter", e);
                Toast.makeText(context, "Error while connecting", Toast.LENGTH_LONG).show();
            }
        }
    }

    @Override
    public void onBackPressed() {
        //super.onBackPressed();
        ConnectPhoneTask connectPhoneTask = new ConnectPhoneTask();
        connectPhoneTask.execute(Constants.SERVER_IP);
    }

    public void onRightClick(View v) {
        //TODO right click options
        if (out != null && isConnected)
            out.println("right");
    }

}
