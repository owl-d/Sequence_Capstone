package com.example.canpstone_sequence;

import static com.example.canpstone_sequence.ETRIActivity.bills_mode;
import static com.example.canpstone_sequence.ETRIActivity.record_end;
import static com.example.canpstone_sequence.ETRIActivity.result;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.media.AudioManager;
import android.media.SoundPool;
import android.os.Build;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.List;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;


public class CameraActivity extends Activity implements SurfaceHolder.Callback {

    static boolean record = false;
    static String Post_String;
    static String byte_image_Stream;
    static long prev_millis_capture = 0;
    static long prev_millis_post = 0;
    static boolean first = true;

    private Camera camera;
    public List<Camera.Size> listPreviewSizes;
    SurfaceView surfaceView;
    SurfaceHolder surfaceHolder;
    Button btn_record;

    private int RESULT_PERMISSIONS = 100;

    @SuppressWarnings("deprecation")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        requestPermissionCamera();

        getWindow().setFormat(PixelFormat.UNKNOWN);

        surfaceView = (SurfaceView) findViewById(R.id.surfaceView);
        surfaceHolder = surfaceView.getHolder();
        surfaceHolder.addCallback(this);
        surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
        
        SoundPool Sound = new SoundPool(1, AudioManager.STREAM_MUSIC,0);
        int click = Sound.load(this, R.raw.asterick, 1);

        //Send ETRI Target to Server
        if(first) {
            Post_String = "etri"+result;
            post_http();
        }

        btn_record = (Button) findViewById(R.id.btn_record);
        btn_record.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Sound.play(click, 1, 1, 0, 0, (float)1.0);
                Log.d("TAG", "btn_record CLICK");

                if(record_end) {
                    Log.d("TAG", "YOLO END or BILLS END");
                    Intent intent = new Intent(getApplicationContext(), TermActivity.class);
                    startActivity(intent);
                }
                else if (record) {
                    prev_millis_capture = System.currentTimeMillis();
                    Log.d("TAG", "Sending Stop");
                    record = false;
                    record_end = true;
                    btn_record.setText("PUSH TO END");
                }
                else {
                    Log.d("TAG", "Sending Start");
                    btn_record.setText("PUSH TO STOP");
                    record = true;
                    record_end = false;
                }

                new Thread(new Runnable() {
                    public void run() {

                        while(record){
                            //Log.d("TAG", "Base64 Encoding : " + byte_image_Stream.substring(1000, 1005));

                            long now_millis_post = System.currentTimeMillis();
                            if(record && (now_millis_post - prev_millis_post > 4000)){
                                prev_millis_post = now_millis_post;
                                if(bills_mode == 0) {
                                    Post_String = byte_image_Stream;
                                    post_http();
                                    Log.d("TAG", "post_http : YOLO");
                                }
                                else if (bills_mode == 1) {
                                    Post_String = "bill" + byte_image_Stream;
                                    post_http();
                                    Log.d("TAG", "post_http : BILL");
                                    bills_mode = 2;
                                }

                            }
                        }
                    }
                }).start();
            }
        });
    }

    private void post_http() {
        Log.d("TAG", "Run post_http");
        String postUrl = "http://3.37.237.18:5000/";
        MediaType mediaType = MediaType.parse("text/plain");
        RequestBody postBody = RequestBody.create(mediaType, Post_String);

        postRequest(postUrl, postBody);
    }

    private void postRequest(String postUrl, RequestBody postBody) {

        SoundPool Sound = new SoundPool(1, AudioManager.STREAM_MUSIC,0);
        int category_milk = Sound.load(this, R.raw.category_milk, 1);
        int category_snack = Sound.load(this, R.raw.category_snack, 1);
        int category_beverage = Sound.load(this, R.raw.category_beverage, 1);
        int object = Sound.load(this, R.raw.target_object, 1);
        int bananakick = Sound.load(this, R.raw.bananakick, 1);
        int cocacola = Sound.load(this, R.raw.cocacola, 1);
        int jagalchi = Sound.load(this, R.raw.jagalchi, 1);
        int maeil_almond_breeze = Sound.load(this, R.raw.maeil_almond_breeze, 1);
        int namyang_milk_gt = Sound.load(this, R.raw.namyang_milk_gt, 1);
        int saewooggang_90g = Sound.load(this, R.raw.saewooggang_90g, 1);
        int seoul_milk = Sound.load(this, R.raw.seoul_milk, 1);
        int toraeta = Sound.load(this, R.raw.toraeta, 1);
        int bill_success = Sound.load(this, R.raw.bill_success, 1);
        int bill_fail = Sound.load(this, R.raw.bill_fail, 1);

        Log.d("TAG", "Run postRequest");
        OkHttpClient client = new OkHttpClient();

        Request request = new Request.Builder()

                .url(postUrl)
                .post(postBody)
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                // Cancel the post on failure.
                call.cancel();

                // In order to access the TextView inside the UI thread, the code is executed inside runOnUiThread()
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Log.d("TAG", "PostRequeset : Failure");
                    }
                });
            }

            @Override
            public void onResponse(Call call, final Response response) throws IOException {
                // In order to access the TextView inside the UI thread, the code is executed inside runOnUiThread()
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {

                        int _return = 0;
                        try {
                            _return = Integer.parseInt(response.body().string());
                        } catch (IOException e) {
                            e.printStackTrace();
                        }

                        if (_return == 1) {
                            Log.d("TAG", "AWS Response : ETRI FINISH");
                        }
                        else if (_return == 2) {
                            Log.d("TAG", "AWS Response : FINISH SMART BILL - SUCCESS");
                            Sound.play(bill_success, 1, 1, 0, 0, (float)1.0);
                        }
                        else if (_return == 7) {
                            Log.d("TAG", "AWS Response : FINISH SMART BILL - FAIL");
                            Sound.play(bill_fail, 1, 1, 0, 0, (float)1.0);
                        }


                        else if (_return == 10) {
                            Log.d("TAG", "AWS Response : Find MILK Category");
                            Toast.makeText(getApplicationContext(), "Find Target Category", Toast.LENGTH_SHORT).show();
                            Sound.play(category_milk, 1, 1, 0, 0, (float)1.0);
                        }
                        else if (_return == 11) {
                            Log.d("TAG", "AWS Response : Find SNACK Category");
                            Toast.makeText(getApplicationContext(), "Find Target Category", Toast.LENGTH_SHORT).show();
                            Sound.play(category_snack, 1, 1, 0, 0, (float)1.0);
                        }
                        else if (_return == 12) {
                            Log.d("TAG", "AWS Response : Find BEVERAGE Category");
                            Toast.makeText(getApplicationContext(), "Find Target Category", Toast.LENGTH_SHORT).show();
                            Sound.play(category_beverage, 1, 1, 0, 0, (float)1.0);
                        }

                        else if (_return == 20) {
                            Log.d("TAG", "AWS Response : Find Object '?????????????????????GT'");
                            Toast.makeText(getApplicationContext(), "Find Target Category", Toast.LENGTH_SHORT).show();
                            Sound.play(namyang_milk_gt, 1, 1, 0, 0, (float)1.0);
                        }
                        else if (_return == 21) {
                            Log.d("TAG", "AWS Response : Find Object '?????????????????????90G'");
                            Toast.makeText(getApplicationContext(), "Find Target Category", Toast.LENGTH_SHORT).show();
                            Sound.play(saewooggang_90g, 1, 1, 0, 0, (float)1.0);
                        }
                        else if (_return == 22) {
                            Log.d("TAG", "AWS Response : Find Object '?????????????????????????????????????????????'");
                            Toast.makeText(getApplicationContext(), "Find Target Category", Toast.LENGTH_SHORT).show();
                            Sound.play(maeil_almond_breeze, 1, 1, 0, 0, (float)1.0);
                        }
                        else if (_return == 23) {
                            Log.d("TAG", "AWS Response : Find Object '????????????)????????????350ML'");
                            Toast.makeText(getApplicationContext(), "Find Target Category", Toast.LENGTH_SHORT).show();
                            Sound.play(cocacola, 1, 1, 0, 0, (float)1.0);
                        }
                        else if (_return == 24) {
                            Log.d("TAG", "AWS Response : Find Object '???????????????90G'");
                            Toast.makeText(getApplicationContext(), "Find Target Category", Toast.LENGTH_SHORT).show();
                            Sound.play(jagalchi, 1, 1, 0, 0, (float)1.0);
                        }
                        else if (_return == 25) {
                            Log.d("TAG", "AWS Response : Find Object '??????????????????75G'");
                            Toast.makeText(getApplicationContext(), "Find Target Category", Toast.LENGTH_SHORT).show();
                            Sound.play(bananakick, 1, 1, 0, 0, (float)1.0);
                        }
                        else if (_return == 26) {
                            Log.d("TAG", "AWS Response : Find Object '???????????????500ML'");
                            Toast.makeText(getApplicationContext(), "Find Target Category", Toast.LENGTH_SHORT).show();
                            Sound.play(toraeta, 1, 1, 0, 0, (float)1.0);
                        }
                        else if (_return == 27) {
                            Log.d("TAG", "AWS Response : Find Object '????????????'");
                            Toast.makeText(getApplicationContext(), "Find Target Category", Toast.LENGTH_SHORT).show();
                            Sound.play(seoul_milk, 1, 1, 0, 0, (float)1.0);
                        }

                        else if (_return == 6) {
                            Log.d("TAG", "AWS Response : Detecting ...");
                        }
                        else {Log.d("TAG", "AWS Response : ELSE");}

                    }
                });
            }
        });
    }



    private boolean requestPermissionCamera() {
        int sdkVersion = Build.VERSION.SDK_INT;
        if(sdkVersion >= Build.VERSION_CODES.M) {

            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {

                ActivityCompat.requestPermissions(CameraActivity.this,
                        new String[]{Manifest.permission.CAMERA},
                        RESULT_PERMISSIONS);

            }else {
            }
        }else{  // version 6 ????????????
            return true;
        }

        return true;
    }


    public void refreshCamera() {
        Log.d("TAG", "refreshCamera");
        if (surfaceHolder.getSurface() == null) {
            return;
        }

        try {
            camera.stopPreview();
        }

        catch (Exception e) {
        }

        try {
            camera.setPreviewDisplay(surfaceHolder);
            camera.startPreview();
            camera.setPreviewCallback((Camera.PreviewCallback) this);

        }
        catch (Exception e) {
        }
    }

    @Override
    protected void onDestroy() {
        Log.d("TAG", "onDestroy");
        super.onDestroy();
    }


    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        Log.d("TAG", "SurfaceCreated");

        camera = Camera.open();
        camera.stopPreview();
        Camera.Parameters param = camera.getParameters();
        param.setRotation(90);
        camera.setParameters(param);
        listPreviewSizes = camera.getParameters().getSupportedPreviewSizes();

        // ???????????? ????????? ??????/???????????? ????????? ????????????.
        if (getResources().getConfiguration().orientation != Configuration.ORIENTATION_LANDSCAPE) {
            param.set("orientation", "portrait");
            camera.setDisplayOrientation(90);
            param.setRotation(90);
        } else {
            param.set("orientation", "landscape");
            camera.setDisplayOrientation(0);
            param.setRotation(0);
        }

        try {

            camera.setPreviewDisplay(surfaceHolder);
            camera.startPreview();
            // ??????????????? ??????
            camera.autoFocus(new Camera.AutoFocusCallback() {
                public void onAutoFocus(boolean success, Camera camera) {
                    if (success) {

                    }
                }
            });
        }

        catch (Exception e) {
            System.err.println(e);
            return;
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int i, int w, int h) {
        //refreshCamera();
        Log.d("TAG", "SurfaceChanged");

        camera.setPreviewCallback(new Camera.PreviewCallback(){
            @Override
            public void onPreviewFrame(byte[] data, Camera camera){

                long now_millis_capture = System.currentTimeMillis();
                if(first || bills_mode == 1 || (record && (now_millis_capture - prev_millis_capture > 4000))){
                    Log.d("TAG", "Image Capture Mode");

                    first = false;
                    prev_millis_capture = now_millis_capture;

                    //?????? SurfaceView??? ??????
                    Camera.Parameters param = camera.getParameters();
                    int w = param.getPreviewSize().width;
                    int h = param.getPreviewSize().height;
                    int format = param.getPreviewFormat();
                    YuvImage image = new YuvImage(data, format, w, h, null);

                    /// Base64 Image Encoding ////////////////////////////////////////////////////////////
                    ByteArrayOutputStream out = new ByteArrayOutputStream();
                    Rect area = new Rect(0, 0, w, h);
                    image.compressToJpeg(area, 100, out);
                    Bitmap imageBitmap = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size());
                    imageBitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
                    Log.d("TAG", "onActivityResult : Image Ready");

                    byte[] currentData = out.toByteArray();
                    byte_image_Stream = Base64.encodeToString(currentData, 0);
                }
                else if(record_end){
                    Log.d("TAG", "Record End");
                    releaseCamera();
                }
            }
        });
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        Log.d("TAG", "SurfaceDestroyed");
        camera.stopPreview();
        camera.release();
        camera = null;
    }

    // release Camera for other applications
    private void releaseCamera() {
        // check if Camera instance exists
        if (camera != null) {
            // first stop preview
            camera.stopPreview();
            // then cancel its preview callback
            camera.setPreviewCallback(null);
        }
    }

}