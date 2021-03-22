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

package org.tensorflow.lite.examples.classification.tflite;

import android.app.Activity;
import java.io.IOException;
import org.tensorflow.lite.examples.classification.tflite.Classifier.Device;

/** This TensorFlowLite classifier works with the float EfficientNet model. */
public class ClassifierFloatEfficientNet extends Classifier {

  /**
   * Initializes a {@code ClassifierFloatMobileNet}.
   *
   * @param device a {@link Device} object to configure the hardware accelerator
   * @param numThreads the number of threads during the inference
   * @throws IOException if the model is not loaded correctly
   */
  public ClassifierFloatEfficientNet(Activity activity, Device device, int numThreads)
      throws IOException {
    super(activity, device, numThreads);
  }

  @Override
  protected String getModelPath() {
    // you can download this file from
    // see build.gradle for where to obtain this file. It should be auto
    // downloaded into assets.
    //return "efficientnet-lite0-fp32.tflite";
    return "model.tflite";
  }
}
