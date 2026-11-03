// Simulation of the vblank PLL: reports arrive late by a random latency, sometimes very late,
// and the display sometimes repeats a frame. Checks the predicted vblank times and eyes.
#include <cmath>
#include <cstdio>
#include <random>
#include "../src/config.h"
#include "../src/emitter.h"
using namespace stereo3d;

int main(int argc, char** argv) {
  double hz = argc > 1 ? atof(argv[1]) : 119.982;
  Cfg().logLevel = 0;
  std::mt19937 rng(42);
  std::exponential_distribution<double> lat(1.0 / 150e-6);  // mean 150 us wake-up latency
  std::uniform_real_distribution<double> u(0, 1);
  const double P = 1.0 / hz, T0 = 1000.0;
  int eye = 0;
  double maxErr = 0, sumErr = 0, sumSq = 0;
  int n = 0, eyeWrong = 0, checks = 0, repeats = 0;
  for (int v = 0; v < 120 * 60; ++v) {  // one minute
    double tv = T0 + v * P;
    if (v > 0) {
      if (u(rng) < 0.002) ++repeats;     // display repeated the previous image (missed present)
      else eye ^= 1;
    }
    double t = tv + 30e-6 + lat(rng);
    if (u(rng) < 0.01) t += 0.003;       // scheduling hiccup
    if (u(rng) < 0.003) continue;        // report lost
    SyncObserve(t, eye);
    // predict the next vblank right after the report and compare with the truth
    double pv; int pe; int64_t num;
    if (v > 240 && SyncPredict(t + 0.0003, &pv, &pe, &num)) {
      double truth = T0 + (v + 1) * P;
      if (pv > truth + P / 2) truth += P;
      double err = pv - truth;
      maxErr = std::max(maxErr, std::fabs(err));
      sumErr += err;
      sumSq += err * err;
      ++n;
      ++checks;
      if (pe != (eye ^ 1)) ++eyeWrong;   // next vblank shows the other eye (unless it repeats)
    }
  }
  double mean = sumErr / n, sd = std::sqrt(std::max(0.0, sumSq / n - mean * mean));
  printf("refresh %.3f Hz -> model %.4f Hz | vblank prediction: offset %.0f us, jitter %.0f us, max %.0f us | "
         "eye wrong %d of %d (display repeats %d)\n",
         hz, 1.0 / SyncPeriod(), mean * 1e6, sd * 1e6, maxErr * 1e6, eyeWrong, checks, repeats);
  bool ok = std::fabs(mean) < 250e-6 && sd < 60e-6 && maxErr < 1e-3 && eyeWrong <= repeats * 2 + 5 &&
            std::fabs(1.0 / SyncPeriod() - hz) < 0.02;
  printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
