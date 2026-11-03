// 3dvision-tool - helper for the NVIDIA 3D Vision USB emitter on Linux
//
//   3dvision-tool extract-fw <nvstusb.sys | folder> ...   find the emitter firmware in the Windows
//                                                       driver and install it for the layer
//   3dvision-tool status                                is the emitter connected / usable?
//   3dvision-tool upload                                load the firmware now
//   3dvision-tool test [seconds] [hz]                   drive the glasses without a game
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/config.h"
#include "../src/emitter.h"

using namespace stereo3d;

static bool ReadAll(const std::string& p, std::vector<uint8_t>& out) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) return false;
  out.clear();
  uint8_t buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.insert(out.end(), buf, buf + n);
  fclose(f);
  return true;
}

static void Collect(const std::string& path, std::vector<std::string>& files, int depth = 0) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) return;
  if (S_ISREG(st.st_mode)) {
    if (st.st_size > 0 && st.st_size < 64 * 1024 * 1024) files.push_back(path);
    return;
  }
  if (!S_ISDIR(st.st_mode) || depth > 8) return;
  DIR* d = opendir(path.c_str());
  if (!d) return;
  while (dirent* e = readdir(d)) {
    if (e->d_name[0] == '.') continue;
    Collect(path + "/" + e->d_name, files, depth + 1);
  }
  closedir(d);
}

static void MkdirP(const std::string& dir) {
  for (size_t p = 1; p <= dir.size(); ++p)
    if (p == dir.size() || dir[p] == '/') mkdir(dir.substr(0, p).c_str(), 0755);
}

static int ExtractFw(int argc, char** argv) {
  std::string out = ConfigDir() + "/nvstusb.fw";
  std::vector<std::string> inputs;
  for (int i = 0; i < argc; ++i) {
    if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
    else inputs.push_back(argv[i]);
  }
  if (inputs.empty()) {
    fprintf(stderr, "usage: 3dvision-tool extract-fw <nvstusb.sys | folder with the extracted driver> [-o out.fw]\n");
    return 2;
  }
  std::vector<std::string> files;
  for (auto& in : inputs) Collect(in, files);
  // nvstusb*.sys first
  std::stable_sort(files.begin(), files.end(), [](const std::string& a, const std::string& b) {
    auto score = [](const std::string& s) { return strcasestr(s.c_str(), "nvstusb") ? 0 : strcasestr(s.c_str(), ".sys") ? 1 : 2; };
    return score(a) < score(b);
  });
  for (auto& f : files) {
    std::vector<uint8_t> data, fw;
    if (!ReadAll(f, data)) continue;
    std::string info;
    if (!ExtractFirmware(data, fw, &info)) continue;
    size_t slash = out.find_last_of('/');
    if (slash != std::string::npos) MkdirP(out.substr(0, slash));
    FILE* o = fopen(out.c_str(), "wb");
    if (!o || fwrite(fw.data(), 1, fw.size(), o) != fw.size()) {
      fprintf(stderr, "cannot write %s\n", out.c_str());
      if (o) fclose(o);
      return 1;
    }
    fclose(o);
    printf("%s: %s\nfirmware written to %s (%zu bytes)\n", f.c_str(), info.c_str(), out.c_str(), fw.size());
    return 0;
  }
  fprintf(stderr, "no 3D Vision emitter firmware found in %zu file(s).\n"
                  "It is inside nvstusb.sys from the Windows 3D Vision driver (see README).\n", files.size());
  return 1;
}

static int Status() {
  EmitterInfo info = FindEmitter();
  if (!info.found) {
    printf("emitter: NOT connected (USB 0955:0007 not found)\n");
    return 1;
  }
  printf("emitter: %s, %s\n", info.devPath.c_str(),
         info.endpoints == 0 ? "no firmware loaded yet (it is loaded automatically)" : "firmware loaded");
  bool rw = access(info.devPath.c_str(), R_OK | W_OK) == 0;
  printf("permissions: %s\n", rw ? "OK" : "NO ACCESS -> run: sudo ./install-udev.sh  (then replug the emitter)");
  std::string used;
  std::vector<uint8_t> fw;
  if (LoadFirmwareFile(fw, &used)) printf("firmware file: %s (%zu bytes)\n", used.c_str(), fw.size());
  else printf("firmware file: MISSING -> 3dvision-tool extract-fw <nvstusb.sys>\n");
  return rw ? 0 : 1;
}

static int Upload() {
  Emitter em;
  std::string err;
  if (!em.Open(&err)) {
    fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  printf("emitter ready (firmware loaded)\n");
  return 0;
}

// free-running test: no display sync, just alternate the shutters at `hz`
static int Test(double seconds, double hz) {
  Emitter em;
  std::string err;
  if (!em.Open(&err)) {
    fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  if (!em.SetRate(hz)) {
    fprintf(stderr, "set rate failed\n");
    return 1;
  }
  printf("glasses running at %.1f Hz for %.0f s (they should darken / flicker slightly).\n"
         "Turn the wheel or press the button on the emitter to test them.\n", hz, seconds);
  const double period = 1.0 / hz, start = NowSeconds();
  int eye = 0;
  uint64_t n = 0, errors = 0;
  double nextKeys = start;
  while (NowSeconds() - start < seconds) {
    double target = start + double(n) * period;
    while (NowSeconds() < target) {
      double left = target - NowSeconds();
      if (left > 0.0015) usleep(useconds_t((left - 0.001) * 1e6));
    }
    if (!em.SetEye(eye)) ++errors;
    eye ^= 1;
    ++n;
    if (NowSeconds() >= nextKeys) {
      nextKeys = NowSeconds() + 0.2;
      int w = 0, wp = 0;
      bool b = false;
      if (em.ReadKeys(&w, &wp, &b) && (w || wp || b)) printf("wheel %+d  wheel(pressed) %+d  button %d\n", w, wp, b);
    }
  }
  printf("%llu eye signals sent, %llu USB errors\n", (unsigned long long)n, (unsigned long long)errors);
  return errors ? 1 : 0;
}

int main(int argc, char** argv) {
  Cfg().logLevel = 1;
  if (argc < 2) {
    printf("3dvision-tool - NVIDIA 3D Vision emitter helper\n\n"
           "  3dvision-tool extract-fw <nvstusb.sys | folder> [-o out.fw]\n"
           "  3dvision-tool status\n"
           "  3dvision-tool upload\n"
           "  3dvision-tool test [seconds=10] [hz=120]\n");
    return 2;
  }
  std::string cmd = argv[1];
  if (cmd == "extract-fw") return ExtractFw(argc - 2, argv + 2);
  if (cmd == "status") return Status();
  if (cmd == "upload") return Upload();
  if (cmd == "test") return Test(argc > 2 ? atof(argv[2]) : 10.0, argc > 3 ? atof(argv[3]) : 120.0);
  fprintf(stderr, "unknown command %s\n", cmd.c_str());
  return 2;
}
