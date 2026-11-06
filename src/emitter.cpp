// 3dvisionlinux - NVIDIA 3D Vision USB IR emitter + shutter synchronisation (see emitter.h)
#include "emitter.h"

#include <dirent.h>
#include <fcntl.h>
#include <linux/usbdevice_fs.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <thread>

#include "config.h"

namespace stereo3d {

namespace {
constexpr int kVendor = 0x0955, kProduct = 0x0007;

// 8051 (Cypress FX2) clocks used by the firmware timers
constexpr double kClock = 48000000.0;
constexpr double kT0Clock = kClock / 12.0;  // 4 MHz
constexpr double kT2Clock = kClock / 4.0;   // 12 MHz
int32_t T0Count(double us) { return int32_t(std::lround(-us * (kT0Clock / 1e6) + 1)); }
int32_t T2Count(double us) { return int32_t(std::lround(-us * (kT2Clock / 1e6) + 1)); }

constexpr uint8_t kCmdWrite = 0x01, kCmdRead = 0x02, kCmdClear = 0x40, kCmdSetEye = 0xAA;

bool ReadText(const std::string& path, std::string& out) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return false;
  char buf[64];
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  buf[n] = 0;
  out = buf;
  while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
  return true;
}

// bNumEndpoints of interface 0, parsed from the descriptors usbfs returns on read()
int CountEndpoints(int fd) {
  uint8_t buf[4096];
  if (lseek(fd, 0, SEEK_SET) < 0) return -1;
  ssize_t n = read(fd, buf, sizeof(buf));
  if (n < 18) return -1;
  ssize_t p = buf[0];  // skip device descriptor
  while (p + 2 <= n) {
    uint8_t len = buf[p], type = buf[p + 1];
    if (len < 2) break;
    if (type == 4 && p + 4 < n) return buf[p + 4];  // first interface descriptor
    p += len;
  }
  return 0;
}

bool Control(int fd, uint8_t reqType, uint8_t req, uint16_t value, uint16_t index, void* data, uint16_t len,
             unsigned timeoutMs) {
  usbdevfs_ctrltransfer c{};
  c.bRequestType = reqType;
  c.bRequest = req;
  c.wValue = value;
  c.wIndex = index;
  c.wLength = len;
  c.timeout = timeoutMs;
  c.data = data;
  return ioctl(fd, USBDEVFS_CONTROL, &c) >= 0;
}

std::string HomeConfig() {
  if (const char* x = getenv("XDG_CONFIG_HOME"); x && *x) return std::string(x) + "/3dvision";
  if (const char* h = getenv("HOME"); h && *h) return std::string(h) + "/.config/3dvision";
  return "/tmp/3dvision";
}

bool ReadFile(const std::string& path, std::vector<uint8_t>& out) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  out.clear();
  uint8_t buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.insert(out.end(), buf, buf + n);
  fclose(f);
  return true;
}

// does fw look like a sequence of upload records?
bool ValidRecords(const std::vector<uint8_t>& fw, size_t* blocks) {
  size_t p = 0, b = 0;
  while (p + 4 <= fw.size()) {
    uint32_t len = (fw[p] << 8) | fw[p + 1], addr = (fw[p + 2] << 8) | fw[p + 3];
    if (len == 0 || len > 4096 || addr + len > 0x10000) return false;
    p += 4 + len;
    ++b;
  }
  if (blocks) *blocks = b;
  return p == fw.size() && b > 0;
}
}  // namespace

// ------------------------------------------------------------------------------------
// Device
// ------------------------------------------------------------------------------------
EmitterInfo FindEmitter() {
  EmitterInfo info;
  DIR* d = opendir("/sys/bus/usb/devices");
  if (!d) return info;
  while (dirent* e = readdir(d)) {
    if (e->d_name[0] == '.' || strchr(e->d_name, ':')) continue;
    std::string base = std::string("/sys/bus/usb/devices/") + e->d_name;
    std::string v, p, bus, dev;
    if (!ReadText(base + "/idVendor", v) || !ReadText(base + "/idProduct", p)) continue;
    if (strtol(v.c_str(), nullptr, 16) != kVendor || strtol(p.c_str(), nullptr, 16) != kProduct) continue;
    if (!ReadText(base + "/busnum", bus) || !ReadText(base + "/devnum", dev)) continue;
    char path[64];
    snprintf(path, sizeof(path), "/dev/bus/usb/%03d/%03d", atoi(bus.c_str()), atoi(dev.c_str()));
    info.found = true;
    info.devPath = path;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
      info.endpoints = CountEndpoints(fd);
      close(fd);
    } else {
      std::string ne;  // fall back to sysfs
      if (ReadText(base + "/" + e->d_name + ":1.0/bNumEndpoints", ne)) info.endpoints = int(strtol(ne.c_str(), nullptr, 16));
    }
    break;
  }
  closedir(d);
  return info;
}

std::vector<std::string> FirmwareSearchPaths() {
  std::vector<std::string> v;
  if (const char* f = getenv("NV3D_FIRMWARE"); f && *f) v.push_back(f);
  v.push_back(HomeConfig() + "/nvstusb.fw");
  v.push_back("/usr/local/share/3dvision/nvstusb.fw");
  v.push_back("/usr/share/3dvision/nvstusb.fw");
  v.push_back("/lib/firmware/nvstusb.fw");
  return v;
}

bool LoadFirmwareFile(std::vector<uint8_t>& fw, std::string* usedPath) {
  for (const auto& p : FirmwareSearchPaths()) {
    std::vector<uint8_t> data;
    if (!ReadFile(p, data)) continue;
    if (!ValidRecords(data, nullptr)) {  // maybe the user put nvstusb.sys there
      std::vector<uint8_t> ex;
      if (!ExtractFirmware(data, ex, nullptr)) continue;
      data.swap(ex);
    }
    fw.swap(data);
    if (usedPath) *usedPath = p;
    return true;
  }
  return false;
}

bool ExtractFirmware(const std::vector<uint8_t>& drv, std::vector<uint8_t>& fw, std::string* info) {
  static const uint8_t sig[8] = {0xC2, 0x55, 0x09, 0x07, 0x00, 0x00, 0x00, 0x00};
  // already in upload format?
  size_t blocks = 0;
  if (ValidRecords(drv, &blocks) && blocks >= 2) {
    fw = drv;
    if (info) *info = "file is already a firmware image (" + std::to_string(blocks) + " blocks)";
    return true;
  }
  for (size_t off = 0; off + 8 < drv.size(); ++off) {
    if (memcmp(&drv[off], sig, 8) != 0) continue;
    size_t p = off + 8;
    std::vector<uint8_t> out = {0x00, 0x01, 0xE6, 0x00, 0x01};  // CPUCS = 1: hold the 8051 in reset
    size_t nb = 0, bytes = 0;
    bool ok = false;
    while (p + 4 <= drv.size()) {
      if (drv[p] & 0x80) { ok = nb > 0; break; }  // end marker
      uint32_t len = (drv[p] << 8) | drv[p + 1], addr = (drv[p + 2] << 8) | drv[p + 3];
      if (len == 0 || len > 1024 || addr + len > 0x4000 || p + 4 + len > drv.size()) break;
      out.insert(out.end(), drv.begin() + p, drv.begin() + p + 4 + len);
      p += 4 + len;
      ++nb;
      bytes += len;
    }
    if (!ok || bytes < 1024) continue;  // false positive, keep searching
    static const uint8_t run[5] = {0x00, 0x01, 0xE6, 0x00, 0x00};  // CPUCS = 0: run
    out.insert(out.end(), run, run + 5);
    fw.swap(out);
    if (info) {
      char b[160];
      snprintf(b, sizeof(b), "firmware found at offset 0x%zx: %zu blocks, %zu bytes", off, nb, bytes);
      *info = b;
    }
    return true;
  }
  if (info) *info = "firmware signature not found";
  return false;
}

bool Emitter::Open(std::string* err) {
  Close();
  auto fail = [&](const std::string& m) {
    if (err) *err = m;
    return false;
  };
  EmitterInfo info = FindEmitter();
  if (!info.found) return fail("3D Vision emitter (USB 0955:0007) not connected");
  if (info.endpoints == 0) {
    std::vector<uint8_t> fw;
    std::string used;
    if (!LoadFirmwareFile(fw, &used))
      return fail("the emitter needs its firmware: run '3dvision-tool extract-fw <nvstusb.sys>' (see README)");
    bool hasCpucs = fw.size() > 4 && fw[2] == 0xE6 && fw[3] == 0x00;
    int fd = open(info.devPath.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0)
      return fail(std::string("cannot open ") + info.devPath + ": " + strerror(errno) +
                  " (install the udev rule: sudo ./install-udev.sh)");
    auto put = [&](uint16_t addr, uint8_t* data, uint16_t len) { return Control(fd, 0x40, 0xA0, addr, 0, data, len, 1000); };
    bool ok = true;
    uint8_t one = 1, zero = 0;
    if (!hasCpucs) ok &= put(0xE600, &one, 1);
    for (size_t p = 0; ok && p + 4 <= fw.size();) {
      uint16_t len = uint16_t((fw[p] << 8) | fw[p + 1]), addr = uint16_t((fw[p + 2] << 8) | fw[p + 3]);
      ok &= put(addr, &fw[p + 4], len);
      p += 4 + len;
    }
    if (ok && !hasCpucs) ok &= put(0xE600, &zero, 1);
    if (!ok) {
      int e = errno;
      close(fd);
      return fail(std::string("firmware upload failed: ") + strerror(e));
    }
    Log(0, "3D Vision: firmware uploaded (%s), restarting the emitter...", used.c_str());
    // like libnvstusb: a USB reset makes the host read the new descriptors (the firmware
    // stays in the emitter's RAM)
    ioctl(fd, USBDEVFS_RESET, 0);
    close(fd);
    auto waitBack = [&]() {
      for (int i = 0; i < 40; ++i) {
        usleep(100000);
        info = FindEmitter();
        if (info.found && info.endpoints > 0) return true;
      }
      return false;
    };
    if (!waitBack() && info.found) {  // still the old descriptors: one more reset
      int f2 = open(info.devPath.c_str(), O_RDWR | O_CLOEXEC);
      if (f2 >= 0) {
        ioctl(f2, USBDEVFS_RESET, 0);
        close(f2);
      }
      waitBack();
    }
    if (!info.found || info.endpoints <= 0)
      return fail("emitter did not come back after the firmware upload (endpoints " +
                  std::to_string(info.endpoints) + "); replug it and try again");
  }
  int fd = open(info.devPath.c_str(), O_RDWR | O_CLOEXEC);
  if (fd < 0)
    return fail(std::string("cannot open ") + info.devPath + ": " + strerror(errno) +
                " (install the udev rule: sudo ./install-udev.sh)");
  int cfg = 1;
  ioctl(fd, USBDEVFS_SETCONFIGURATION, &cfg);  // usually already configured
  int iface = 0;
  if (ioctl(fd, USBDEVFS_CLAIMINTERFACE, &iface) < 0) {
    usbdevfs_ioctl dc{};
    dc.ifno = 0;
    dc.ioctl_code = USBDEVFS_DISCONNECT;  // detach a kernel driver, if any
    ioctl(fd, USBDEVFS_IOCTL, &dc);
    if (ioctl(fd, USBDEVFS_CLAIMINTERFACE, &iface) < 0) {
      int e = errno;
      close(fd);
      return fail(std::string("cannot claim the emitter: ") + strerror(e) + " (is another program using it?)");
    }
  }
  fd_ = fd;
  rate_ = 0;
  return true;
}

void Emitter::Close() {
  if (fd_ < 0) return;
  int iface = 0;
  ioctl(fd_, USBDEVFS_RELEASEINTERFACE, &iface);
  close(fd_);
  fd_ = -1;
}

bool Emitter::Bulk(int ep, const void* data, int len, int timeoutMs) {
  if (fd_ < 0) return false;
  usbdevfs_bulktransfer b{};
  b.ep = unsigned(ep);
  b.len = unsigned(len);
  b.timeout = unsigned(timeoutMs);
  b.data = const_cast<void*>(data);
  int r = ioctl(fd_, USBDEVFS_BULK, &b);
  return (ep & 0x80) ? r >= 0 : r == len;
}

bool Emitter::SetRate(double hz) {
  if (fd_ < 0 || hz < 60 || hz > 160) return false;
  // timing values from libnvstusb (reverse engineered from the Windows driver)
  const double frameUs = 1e6 / hz;
  const double activeUs = 2080;  // time each shutter is open
  int32_t w = T2Count(4568.50), x = T0Count(4774.25), y = T0Count(activeUs), z = T2Count(frameUs);
  auto le = [](uint8_t* p, int32_t v) { p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); p[2] = uint8_t(v >> 16); p[3] = uint8_t(v >> 24); };
  uint8_t timings[28] = {kCmdWrite, 0x00, 0x18, 0x00};
  le(timings + 4, w);
  le(timings + 8, x);
  le(timings + 12, y);
  const uint8_t waves[8] = {0x30, 0x28, 0x24, 0x22, 0x0a, 0x08, 0x05, 0x04};
  memcpy(timings + 16, waves, 8);
  le(timings + 24, z);
  bool ok = Bulk(2, timings, sizeof(timings), 200);
  uint8_t c1c[6] = {kCmdWrite, 0x1c, 0x02, 0x00, 0x02, 0x00};
  ok &= Bulk(2, c1c, sizeof(c1c), 200);
  uint16_t timeout = uint16_t(hz * 4);  // frames without set_eye before the glasses go idle
  uint8_t cto[6] = {kCmdWrite, 0x1e, 0x02, 0x00, uint8_t(timeout), uint8_t(timeout >> 8)};
  ok &= Bulk(2, cto, sizeof(cto), 200);
  uint8_t c1b[5] = {kCmdWrite, 0x1b, 0x01, 0x00, 0x07};
  ok &= Bulk(2, c1b, sizeof(c1b), 200);
  if (ok) rate_ = hz;
  return ok;
}

bool Emitter::SetEye(int eye) {
  int32_t r = T2Count((1e6 / (rate_ > 0 ? rate_ : 120.0)) / 1.8);
  uint8_t buf[8] = {kCmdSetEye, uint8_t(eye ? 0xFE : 0xFF), 0, 0, uint8_t(r), uint8_t(r >> 8), uint8_t(r >> 16),
                    uint8_t(r >> 24)};
  return Bulk(1, buf, 8, 50);
}

bool Emitter::ReadKeys(int* wheel, int* wheelPressed, bool* button) {
  uint8_t cmd[4] = {uint8_t(kCmdRead | kCmdClear), 0x18, 0x03, 0x00};
  if (!Bulk(2, cmd, 4, 50)) return false;
  uint8_t rb[7] = {};
  if (!Bulk(0x84, rb, sizeof(rb), 20)) return false;
  *wheel = int8_t(rb[4]);
  *wheelPressed = int8_t(rb[5]);
  *button = (rb[6] & 1) != 0;
  return true;
}

// ------------------------------------------------------------------------------------
// Synchronisation engine
// ------------------------------------------------------------------------------------
double NowSeconds() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}

namespace {
struct Sync {
  std::mutex m;
  // vblank model: vblank number `abs0` happened at t0, period `period`; it showed eye `eye0`
  bool valid = false, locked = false;
  double t0 = 0, period = 1.0 / 120, lastObs = 0;
  int64_t abs0 = 0;
  int eye0 = 0;
  uint64_t obs = 0, slips = 0, outliers = 0;
  std::vector<double> warm;  // first observations, to measure the period
  std::atomic<double> lastObsAtomic{0};

  std::thread th;
  std::atomic<bool> run{false};
  std::atomic<uint64_t> sent{0}, usbErrors{0}, late{0}, slowUsb{0};
  std::atomic<bool> emitterOpen{false};
  std::string lastError;
  int64_t lastSentAbs = INT64_MIN;
  ~Sync() {  // process exit without vkDestroyDevice: never leave a joinable thread behind
    run = false;
    if (th.joinable()) th.join();
  }
};
Sync g_sync;
std::mutex g_startLock;

void SleepUntil(double target) {
  for (;;) {
    double now = NowSeconds();
    double left = target - now;
    if (left <= 0) return;
    if (left > 0.0015) {
      timespec ts{0, long((left - 0.001) * 1e9)};
      nanosleep(&ts, nullptr);
    } else if (left > 0.0002) {
      sched_yield();
    }
  }
}

void HandleKeys(int wheel, int wheelPressed, bool button) {
  Config& c = Cfg();
  if (wheel) {  // wheel: depth (separation), like the Windows driver
    c.separation = std::clamp(c.separation * std::pow(1.06f, float(wheel)), 0.0005f, 0.2f);
    Log(1, "emitter wheel: separation=%.4f", c.separation);
  }
  if (wheelPressed) {  // wheel while the button is held: convergence
    c.convergence = std::max(0.001f, c.convergence * std::pow(1.06f, float(wheelPressed)));
    Log(1, "emitter wheel (pressed): convergence=%.3f", c.convergence);
  }
  if (button && c.emitterButton) {
    c.stereo = !c.stereo;
    Log(1, "emitter button: 3D %s", c.stereo ? "ON" : "OFF");
  }
}

void SyncThread() {
  Sync& s = g_sync;
  // real-time priority if allowed (rtkit / limits.conf); otherwise just a normal thread
  sched_param sp{};
  sp.sched_priority = 20;
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) Log(2, "3D Vision: no real-time priority for the emitter thread");
  pthread_setname_np(pthread_self(), "3dvision-sync");
  prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);  // wake up on time (default slack is 50 us)

  Emitter em;
  double nextOpenTry = 0, nextKeys = 0, nextStat = NowSeconds() + 10, nextRate = 0;
  double rateEma = 0;
  bool warnedOpen = false;
  while (s.run.load()) {
    double now = NowSeconds();
    if (!em.IsOpen()) {
      s.emitterOpen = false;
      if (!Cfg().emitter || now < nextOpenTry) {
        SleepUntil(now + 0.05);
        continue;
      }
      std::string err;
      if (em.Open(&err)) {
        Log(0, "3D Vision emitter ready");
        s.emitterOpen = true;
        warnedOpen = false;
        double p = SyncPeriod();
        em.SetRate(p > 0 ? 1.0 / p : (Cfg().refresh > 0 ? Cfg().refresh : 120.0));
      } else {
        if (!warnedOpen || err != s.lastError) Log(0, "3D Vision: %s", err.c_str());
        warnedOpen = true;
        s.lastError = err;
        nextOpenTry = now + 3.0;
        continue;
      }
    }

    double hz = 1.0 / SyncPeriod();
    if (!SyncLocked() || now - g_sync.lastObsAtomic.load() > 0.25) {  // nothing presented: let the glasses idle
      SleepUntil(now + 0.004);
      continue;
    }
    // The emitter times the shutters with its own clock, programmed with set_rate; set_eye only
    // says which eye comes next. If that clock runs even 0.1 Hz off the display, the shutters
    // slide across the frames and the eyes swap every few seconds (then snap back). The first
    // measurement can be that far off, so follow the refined model closely (no extra latency:
    // it only reprograms the emitter's period).
    rateEma = rateEma > 0 ? rateEma + (hz - rateEma) * 0.02 : hz;  // smooth the model's jitter
    if (std::fabs(rateEma - em.Rate()) > 0.003 && now >= nextRate) {
      nextRate = now + 2.0;
      double before = em.Rate();
      if (em.SetRate(rateEma)) Log(1, "3D Vision: emitter clock %.3f -> %.3f Hz", before, rateEma);
      else ++s.usbErrors;
    }

    const double delay = double(Cfg().emitterDelayUs) * 1e-6;
    double vb;
    int eye;
    int64_t num;
    if (!SyncPredict(now + 0.0003 - delay, &vb, &eye, &num)) continue;
    if (num <= s.lastSentAbs) {  // already sent for this vblank: take the next one
      if (!SyncPredict(vb + 0.5 / hz, &vb, &eye, &num)) continue;
    }
    double target = vb + delay;
    if (target - now > 0.1) {  // model jumped; re-read it soon
      SleepUntil(now + 0.002);
      continue;
    }
    SleepUntil(target);
    // woke up too late (the system was busy): a late command would shift the shutters for one
    // refresh (visible as a blink). Skip it: the emitter keeps alternating on its own timer.
    if (NowSeconds() - target > 0.0006) {
      ++s.late;
      s.lastSentAbs = num;
      continue;
    }
    if (Cfg().emitterInvert) eye ^= 1;
    const double t0usb = NowSeconds();
    bool okEye = em.SetEye(eye);
    if (NowSeconds() - t0usb > 0.002) ++s.slowUsb;
    if (!okEye) {
      ++s.usbErrors;
      if (errno == ENODEV || errno == ENOENT || errno == EIO) {
        Log(0, "3D Vision: emitter disconnected");
        em.Close();
        nextOpenTry = NowSeconds() + 1.0;
      }
      continue;
    }
    s.lastSentAbs = num;
    ++s.sent;

    now = NowSeconds();
    if (now >= nextKeys) {  // wheel / button, a few times per second, right after a set_eye
      nextKeys = now + 0.25;
      int w = 0, wp = 0;
      bool b = false;
      if (em.ReadKeys(&w, &wp, &b)) HandleKeys(w, wp, b);
    }
    if (now >= nextStat) {
      nextStat = now + 10;
      Log(1, "3D Vision: %s", SyncStatusLine().c_str());
    }
  }
  em.Close();
  s.emitterOpen = false;
}
}  // namespace

void SyncStart() {
  std::lock_guard<std::mutex> l(g_startLock);
  if (g_sync.run.load()) return;
  g_sync.run = true;
  g_sync.th = std::thread(SyncThread);
}

void SyncStop() {
  std::lock_guard<std::mutex> l(g_startLock);
  if (!g_sync.run.load()) return;
  g_sync.run = false;
  if (g_sync.th.joinable()) g_sync.th.join();
}

void SyncReset() {
  std::lock_guard<std::mutex> l(g_sync.m);
  g_sync.valid = false;
  g_sync.locked = false;
  g_sync.warm.clear();
}

void SyncObserve(double t, int eye) {
  Sync& s = g_sync;
  std::lock_guard<std::mutex> l(s.m);
  if (!s.valid || t - s.lastObs > 0.5) {  // (re)start
    s.abs0 += 1000000;  // keep vblank numbers increasing across restarts
    s.valid = true;
    s.locked = false;
    s.warm.clear();
    s.warm.push_back(t);
    s.t0 = t;
    s.eye0 = eye;
    s.lastObs = t;
    return;
  }
  if (!s.locked) {
    // measure the refresh period: median of the intervals between consecutive reports
    s.warm.push_back(t);
    s.t0 = t;
    s.eye0 = eye;
    s.lastObs = t;
    if (s.warm.size() < 41) return;
    std::vector<double> d;
    for (size_t i = 1; i < s.warm.size(); ++i) d.push_back(s.warm[i] - s.warm[i - 1]);
    std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
    double p = d[d.size() / 2];
    if (Cfg().refresh > 0) {  // the user told us the refresh rate: trust it if close
      double pn = 1.0 / Cfg().refresh;
      if (std::fabs(p - pn) < pn * 0.1) p = pn;
    }
    if (p < 1.0 / 200 || p > 1.0 / 40) {  // nonsense (e.g. no vsync): keep measuring
      s.warm.erase(s.warm.begin(), s.warm.begin() + 20);
      return;
    }
    s.period = p;
    s.locked = true;
    s.warm.clear();
    Log(1, "3D Vision: display refresh measured %.2f Hz", 1.0 / p);
    if (1.0 / p < 95 || 1.0 / p > 125)
      Log(0, "3D Vision: WARNING the display runs at %.1f Hz; 3D Vision needs 100, 110 or 120 Hz", 1.0 / p);
    return;
  }
  s.lastObsAtomic = t;
  double x = (t - s.t0) / s.period;
  int64_t n = int64_t(std::llround(x));
  double err = t - (s.t0 + double(n) * s.period);
  if (n < 0 || std::fabs(err) > s.period * 0.4) {
    if (++s.outliers % 50 == 0) Log(2, "3D Vision: timing outlier %.2f ms", err * 1e3);
    if (std::fabs(err) > s.period * 0.4 && n >= 0) {
      // probably a big scheduling delay; ignore unless it keeps happening
      return;
    }
    return;
  }
  // eye bookkeeping: did the display repeat a frame (missed vblank)?
  int predicted = (s.eye0 ^ int(n & 1)) & 1;
  if (predicted != eye) ++s.slips;
  // reports arrive *after* the vblank (wake-up latency, never early): follow early reports
  // quickly and late ones slowly, i.e. track the lower envelope.
  double g = err < 0 ? 0.25 : 0.02;
  double corr = g * err;
  s.t0 = s.t0 + double(n) * s.period + corr;
  s.abs0 += n;
  s.eye0 = eye;
  // frequency: the phase corrections average out only when the period is right
  if (n > 0) s.period = std::clamp(s.period + 0.02 * corr / double(n), 1.0 / 200, 1.0 / 40);
  s.lastObs = t;
  ++s.obs;
}

bool SyncPredict(double t, double* vblank, int* eye, int64_t* number) {
  std::lock_guard<std::mutex> l(g_sync.m);
  const Sync& s = g_sync;
  if (!s.locked) return false;
  int64_t m = int64_t(std::ceil((t - s.t0) / s.period));
  *vblank = s.t0 + double(m) * s.period;
  *eye = (s.eye0 ^ int(m & 1)) & 1;
  *number = s.abs0 + m;
  return true;
}

bool SyncLocked() {
  std::lock_guard<std::mutex> l(g_sync.m);
  return g_sync.locked;
}
double SyncPeriod() {
  std::lock_guard<std::mutex> l(g_sync.m);
  return g_sync.locked ? g_sync.period : 0.0;
}

std::string SyncStatusLine() {
  Sync& s = g_sync;
  char b[512];
  std::lock_guard<std::mutex> l(s.m);
  snprintf(b, sizeof(b), "emitter %s | display %.3f Hz %s | eye signals sent %llu | dropped display frames %llu | "
           "usb errors %llu | late (skipped) %llu | slow usb %llu | delay %.0f us",
           s.emitterOpen ? "ON" : "NOT CONNECTED", s.locked ? 1.0 / s.period : 0.0, s.locked ? "(locked)" : "(measuring)",
           (unsigned long long)s.sent.load(), (unsigned long long)s.slips, (unsigned long long)s.usbErrors.load(),
           (unsigned long long)s.late.load(), (unsigned long long)s.slowUsb.load(),
           Cfg().emitterDelayUs);
  return b;
}

}  // namespace stereo3d
