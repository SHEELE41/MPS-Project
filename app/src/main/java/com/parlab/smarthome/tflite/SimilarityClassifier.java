/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.parlab.smarthome.tflite;

import android.graphics.Bitmap;
import android.graphics.RectF;

import java.util.List;

public interface SimilarityClassifier {

    void register(String name, Recognition recognition);

    List<Recognition> recognizeImage(Bitmap bitmap, boolean getExtra);

    void enableStatLogging(final boolean debug);

    String getStatString();

    void close();

    void setNumThreads(int num_threads);

    void setUseNNAPI(boolean isChecked);

    class Recognition {
        private final String id;
        private final String title;
        private final Float distance;
        private Object extra;
        private RectF location;
        private Integer color;
        private Bitmap crop;

        public Recognition(
                final String id, final String title, final Float distance, final RectF location) {
            this.id = id;
            this.title = title;
            this.distance = distance;
            this.location = location;
            this.color = null;
            this.extra = null;
            this.crop = null;
        }

        public Object getExtra() {
            return this.extra;
        }

        public void setExtra(Object extra) {
            this.extra = extra;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getDistance() {
            return distance;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public void setLocation(RectF location) {
            this.location = location;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (distance != null) {
                resultString += String.format("(%.1f%%) ", distance * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }

        public Integer getColor() {
            return this.color;
        }

        public void setColor(Integer color) {
            this.color = color;
        }

        public Bitmap getCrop() {
            return this.crop;
        }

        public void setCrop(Bitmap crop) {
            this.crop = crop;
        }
    }
}
