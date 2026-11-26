// Window pre-processing for the fall-detection CNN: raw IMU window -> int8 model input.
//
// This is the same code the XIAO firmware needs, so it is kept free of anything PC-specific.
// Include the model header (e.g. fold0_int8.h) BEFORE this file: it provides ACC_RMS,
// GYRO_RMS, ACC_MAG_MEAN/STD, GYRO_MAG_MEAN/STD, INPUT_SCALE and INPUT_ZERO_POINT.
//
// It mirrors train_cnn.model_input + quantize_cnn.int8_probability step by step, in float32:
//   1. |a| = sqrt(ax^2 + ay^2 + az^2), |w| = sqrt(gx^2 + gy^2 + gz^2)
//   2. ax, ay, az / ACC_RMS;  gx, gy, gz / GYRO_RMS
//      (|a| - ACC_MAG_MEAN) / ACC_MAG_STD;  (|w| - GYRO_MAG_MEAN) / GYRO_MAG_STD
//   3. q = round_half_to_even(x / INPUT_SCALE + INPUT_ZERO_POINT), clipped to [-128, 127]
//
// Rounding: numpy's np.round rounds halves to even. lrintf does the same in the default
// rounding mode; roundf / std::round would round halves away from zero instead.
#pragma once
#include <math.h>
#include <stdint.h>

constexpr int WINDOW_SAMPLES = 100;  // 2 s at 50 Hz
constexpr int RAW_CHANNELS = 6;      // ax, ay, az [g], gx, gy, gz [deg/s]
constexpr int MODEL_CHANNELS = 8;    // + |a|, |w|

inline int8_t quantize_input(float x) {
  long q = lrintf(x / INPUT_SCALE + (float)INPUT_ZERO_POINT);
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return (int8_t)q;
}

// Scaled float input (100 x 8), as the model sees it before quantization.
inline void scale_window(const float raw[WINDOW_SAMPLES][RAW_CHANNELS],
                         float out[WINDOW_SAMPLES][MODEL_CHANNELS]) {
  for (int t = 0; t < WINDOW_SAMPLES; ++t) {
    const float* s = raw[t];
    float acc_mag = sqrtf(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    float gyro_mag = sqrtf(s[3] * s[3] + s[4] * s[4] + s[5] * s[5]);
    out[t][0] = s[0] / ACC_RMS;
    out[t][1] = s[1] / ACC_RMS;
    out[t][2] = s[2] / ACC_RMS;
    out[t][3] = s[3] / GYRO_RMS;
    out[t][4] = s[4] / GYRO_RMS;
    out[t][5] = s[5] / GYRO_RMS;
    out[t][6] = (acc_mag - ACC_MAG_MEAN) / ACC_MAG_STD;
    out[t][7] = (gyro_mag - GYRO_MAG_MEAN) / GYRO_MAG_STD;
  }
}

// Raw window -> int8 model input, written straight into the input tensor (100 * 8 bytes).
inline void preprocess_window(const float raw[WINDOW_SAMPLES][RAW_CHANNELS], int8_t* input) {
  float scaled[WINDOW_SAMPLES][MODEL_CHANNELS];
  scale_window(raw, scaled);
  for (int t = 0; t < WINDOW_SAMPLES; ++t)
    for (int c = 0; c < MODEL_CHANNELS; ++c)
      input[t * MODEL_CHANNELS + c] = quantize_input(scaled[t][c]);
}

// Output byte -> fall probability, and the alarm decision.
inline float output_probability(int8_t q) { return (q - OUTPUT_ZERO_POINT) * OUTPUT_SCALE; }
inline bool is_fall(int8_t q) { return output_probability(q) >= FALL_CUTOFF; }
