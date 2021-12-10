package com.parlab.smarthome;

import android.app.AlertDialog;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.util.TypedValue;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.parlab.smarthome.customview.OverlayView;
import com.parlab.smarthome.driver.JNIDriver;
import com.parlab.smarthome.driver.JNIListener;
import com.parlab.smarthome.env.BorderedText;
import com.parlab.smarthome.env.ImageUtils;
import com.parlab.smarthome.env.Logger;
import com.parlab.smarthome.env.Size;
import com.parlab.smarthome.tflite.SimilarityClassifier;
import com.parlab.smarthome.tflite.TFLiteObjectDetectionAPIModel;
import com.parlab.smarthome.tracking.MultiBoxTracker;

import java.io.IOException;
import java.lang.ref.WeakReference;
import java.util.LinkedList;
import java.util.List;

public class DetectorActivity extends CameraActivity implements OnImageAvailableListener, JNIListener {
    private static final Logger LOGGER = new Logger();
    private static final int TF_OD_API_INPUT_SIZE = 112;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "mobile_face_net.tflite";    // MobileFaceNet
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    private final BtnHandler btnHandler = new BtnHandler(this);
    OverlayView trackingOverlay;
    private Integer sensorOrientation;
    private SimilarityClassifier detector;
    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;
    private boolean computingDetection = false;
    private boolean addPending = false;
    private long timestamp = 0;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private MultiBoxTracker tracker;
    private BorderedText borderedText;
    private FaceDetector faceDetector;
    private Bitmap portraitBmp = null;
    private Bitmap faceBmp = null;
    private JNIDriver mDriver;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setNumThreads(3);

        FaceDetectorOptions options =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                        .setContourMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                        .build();

        faceDetector = FaceDetection.getClient(options);

        mDriver = new JNIDriver();
        mDriver.setListener(this);

        if (mDriver.open("/dev/sm9s5422_interrupt") < 0) {
            LOGGER.d("Driver Open Failed");
        }
    }

    @Override
    public void onPause() {
        mDriver.close();
        super.onPause();
    }

    @Override
    public void onResume() {
        if (mDriver.open("/dev/sm9s5422_interrupt") < 0) {
            LOGGER.d("Driver Open Failed");
        }
        super.onResume();
    }

    private void onAddClick() {
        addPending = true;
    }

    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        try {
            detector =
                    TFLiteObjectDetectionAPIModel.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED);
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);

        int targetW, targetH;
        if (sensorOrientation == 90 || sensorOrientation == 270) {
            targetH = previewWidth;
            targetW = previewHeight;
        } else {
            targetW = previewWidth;
            targetH = previewHeight;
        }
        int cropW = (int) (targetW / 2.0);
        int cropH = (int) (targetH / 2.0);

        croppedBitmap = Bitmap.createBitmap(cropW, cropH, Config.ARGB_8888);

        portraitBmp = Bitmap.createBitmap(targetW, targetH, Config.ARGB_8888);
        faceBmp = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropW, cropH,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                canvas -> {
                    tracker.draw(canvas);
                    if (isDebug()) {
                        tracker.drawDebug(canvas);
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }


    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;

        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        InputImage image = InputImage.fromBitmap(croppedBitmap, 0);
        faceDetector
                .process(image)
                .addOnSuccessListener(faces -> {
                    if (faces.size() == 0) {
                        updateResults(currTimestamp, new LinkedList<>());
                        return;
                    }
                    runInBackground(
                            () -> {
                                onFacesDetected(currTimestamp, faces, addPending);
                                addPending = false;
                            });
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onImageAvailable(ImageReader imageReader) {

    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> detector.setUseNNAPI(isChecked));
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector.setNumThreads(numThreads));
    }

    private Matrix createTransform(
            final int srcWidth,
            final int srcHeight,
            final int dstWidth,
            final int dstHeight,
            final int applyRotation) {

        Matrix matrix = new Matrix();

        if (applyRotation != 0) {
            if (applyRotation % 90 != 0) {
                LOGGER.w("Rotation of %d % 90 != 0", applyRotation);
            }

            matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

            matrix.postRotate(applyRotation);
        }

        if (applyRotation != 0) {
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
        }

        return matrix;
    }

    private void showAddFaceDialog(SimilarityClassifier.Recognition rec) {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        LayoutInflater inflater = getLayoutInflater();
        View dialogLayout = inflater.inflate(R.layout.image_edit_dialog, null);
        ImageView ivFace = (ImageView) dialogLayout.findViewById(R.id.dlg_image);
        TextView tvTitle = (TextView) dialogLayout.findViewById(R.id.dlg_title);
        EditText etName = (EditText) dialogLayout.findViewById(R.id.dlg_input);

        tvTitle.setText("얼굴 등록");
        ivFace.setImageBitmap(rec.getCrop());
        etName.setHint("이름을 입력해주세요.");

        builder.setPositiveButton("OK", (dlg, i) -> {

            String name = etName.getText().toString();
            if (name.isEmpty()) {
                return;
            }
            detector.register(name, rec);
            dlg.dismiss();
        });
        builder.setView(dialogLayout);
        builder.show();

    }

    private void updateResults(long currTimestamp, final List<SimilarityClassifier.Recognition> mappedRecognitions) {

        tracker.trackResults(mappedRecognitions, currTimestamp);
        trackingOverlay.postInvalidate();
        computingDetection = false;

        if (mappedRecognitions.size() > 0) {
            LOGGER.i("Adding results");
            SimilarityClassifier.Recognition rec = mappedRecognitions.get(0);
            if (rec.getExtra() != null) {
                showAddFaceDialog(rec);
            }
        }

        runOnUiThread(
                () -> {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(croppedBitmap.getWidth() + "x" + croppedBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");
                });
    }

    private void onFacesDetected(long currTimestamp, List<Face> faces, boolean add) {
        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
        final Canvas canvas = new Canvas(cropCopyBitmap);
        final Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Style.STROKE);
        paint.setStrokeWidth(2.0f);

        final List<SimilarityClassifier.Recognition> mappedRecognitions =
                new LinkedList<>();

        int sourceW = rgbFrameBitmap.getWidth();
        int sourceH = rgbFrameBitmap.getHeight();
        int targetW = portraitBmp.getWidth();
        int targetH = portraitBmp.getHeight();
        Matrix transform = createTransform(
                sourceW,
                sourceH,
                targetW,
                targetH,
                sensorOrientation);
        final Canvas cv = new Canvas(portraitBmp);

        cv.drawBitmap(rgbFrameBitmap, transform, null);

        final Canvas cvFace = new Canvas(faceBmp);

        for (Face face : faces) {

            LOGGER.i("FACE : " + face.toString());
            LOGGER.i("Running detection on face " + currTimestamp);

            final RectF boundingBox = new RectF(face.getBoundingBox());

            cropToFrameTransform.mapRect(boundingBox);

            RectF faceBB = new RectF(boundingBox);
            transform.mapRect(faceBB);

            float sx = ((float) TF_OD_API_INPUT_SIZE) / faceBB.width();
            float sy = ((float) TF_OD_API_INPUT_SIZE) / faceBB.height();
            Matrix matrix = new Matrix();
            matrix.postTranslate(-faceBB.left, -faceBB.top);
            matrix.postScale(sx, sy);

            cvFace.drawBitmap(portraitBmp, matrix, null);

            String label = "";
            float confidence = -1f;
            int color = Color.BLUE;
            Object extra = null;
            Bitmap crop = null;

            if (add) {
//                Matrix cropMatrix = new Matrix();
//                cropMatrix.postScale(-1, 1);
                crop = Bitmap.createBitmap(portraitBmp,
                        (int) faceBB.left,
                        (int) faceBB.top,
                        (int) faceBB.width(),
                        (int) faceBB.height());
                        // cropMatrix, false);
                // OpenCL
                crop = mDriver.mirror(crop);
            }

            final long startTime = SystemClock.uptimeMillis();
            final List<SimilarityClassifier.Recognition> resultsAux = detector.recognizeImage(faceBmp, add);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            if (resultsAux.size() > 0) {

                SimilarityClassifier.Recognition result = resultsAux.get(0);

                extra = result.getExtra();
                float conf = result.getDistance();
                if (conf < 1.0f) {

                    confidence = conf;
                    label = result.getTitle();
                    if (result.getId().equals("0")) {
                        color = Color.GREEN;
                    } else {
                        color = Color.RED;
                    }
                }

            }

            final SimilarityClassifier.Recognition result = new SimilarityClassifier.Recognition(
                    "0", label, confidence, boundingBox);

            result.setColor(color);
            result.setLocation(boundingBox);
            result.setExtra(extra);
            result.setCrop(crop);
            mappedRecognitions.add(result);
        }
        updateResults(currTimestamp, mappedRecognitions);
    }

    private static class BtnHandler extends Handler {
        private final WeakReference<DetectorActivity> weakReference;

        public BtnHandler(DetectorActivity activity) {
            this.weakReference = new WeakReference<>(activity);
        }

        @Override
        public void handleMessage(Message msg) {
            if (msg.arg1 == 5) {
                weakReference.get().onAddClick();
            }
        }
    }

    @Override
    public void onReceive(int val) {
        Message txt = Message.obtain();
        txt.arg1 = val;
        btnHandler.sendMessage(txt);
    }
}
