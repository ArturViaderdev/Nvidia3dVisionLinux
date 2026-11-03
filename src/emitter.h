// 3dvisionlinux - NVIDIA 3D Vision USB IR emitter (0955:0007) and shutter synchronisation.
//
// Talks to the emitter through Linux usbfs (/dev/bus/usb/BBB/DDD) with plain ioctls, so it
// needs no libusb and also works inside the Steam Linux Runtime container.
// Protocol from libnvstusb (Bjoern Paetzel, Johann Baudy, LGPL-3).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace stereo3d {

// ---- low level device ---------------------------------------------------------------
struct EmitterInfo {
  bool found = false;
  std::string devPath;      // /dev/bus/usb/BBB/DDD
  int endpoints = -1;       // 0 = no firmware loaded yet
};
EmitterInfo FindEmitter();

class Emitter {
 public:
  ~Emitter() { Close(); }
  // Opens the emitter; uploads the firmware first if the device has none. Blocks up to ~6 s
  // while the device re-enumerates after a firmware upload.
  bool Open(std::string* err);
  void Close();
  bool IsOpen() const { return fd_ >= 0; }
  bool SetRate(double hz);                  // monitor refresh rate (100-120 Hz)
  bool SetEye(int eye);                     // 0 = left shutter open, 1 = right
  // front button / wheel. Returns false on USB error.
  bool ReadKeys(int* wheel, int* wheelPressed, bool* button);
  double Rate() const { return rate_; }

 private:
  bool Bulk(int ep, const void* data, int len, int timeoutMs);
  int fd_ = -1;
  double rate_ = 0;
};

// ---- firmware ---------------------------------------------------------------------------
// Where the firmware is looked for: $NV3D_FIRMWARE, ~/.config/3dvision/nvstusb.fw,
// /usr/local/share/3dvision/nvstusb.fw, /usr/share/3dvision/nvstusb.fw, /lib/firmware/nvstusb.fw
std::vector<std::string> FirmwareSearchPaths();
bool LoadFirmwareFile(std::vector<uint8_t>& fw, std::string* usedPath);
// Finds the firmware inside a Windows driver file (nvstusb.sys or any file that contains it)
// and returns it in upload format (blocks of [len BE16][addr BE16][data], CPUCS reset
// records around it). Returns false if not found.
bool ExtractFirmware(const std::vector<uint8_t>& driver, std::vector<uint8_t>& fw, std::string* info);

// ---- synchronisation engine (used by the Vulkan layer) ---------------------------------
// The presenter reports "at time t (CLOCK_MONOTONIC seconds) the display started showing
// the image of eye E". A PLL models the display's vblank clock from those reports, and a
// high-priority thread tells the emitter which eye is on screen at every vblank.
double NowSeconds();
void SyncStart();                   // idempotent; opens the emitter in the background
void SyncStop();
void SyncObserve(double t, int eye);
void SyncReset();                   // display timing interrupted (swapchain recreated...)
bool SyncLocked();
// Model prediction: first vblank at or after time t, and the eye on screen from then on.
bool SyncPredict(double t, double* vblank, int* eye, int64_t* number);
double SyncPeriod();                // estimated refresh period (s), 0 if unknown
std::string SyncStatusLine();

}  // namespace stereo3d
