// 3dvisionlinux - runtime configuration, per-game profiles and hotkeys
#include "config.h"

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <mutex>

namespace stereo3d {

static Config g_cfg;
static std::string g_game;
static std::string g_profile;
static time_t g_profileMtime = 0;
static FILE* g_logFile = nullptr;
static std::mutex g_logMutex;

Config& Cfg() { return g_cfg; }
float HudFactor() {
  if (g_cfg.hudDepth <= 0.0f) return 0.0f;
  return 1.0f - g_cfg.convergence / g_cfg.hudDepth;
}
const std::string& GameName() { return g_game; }
std::string ProfilePath() { return g_profile; }

void Log(int level, const char* fmt, ...) {
  if (level > g_cfg.logLevel) return;
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  std::lock_guard<std::mutex> l(g_logMutex);
  fprintf(stderr, "[3dvision] %s\n", buf);
  if (g_logFile) {
    fprintf(g_logFile, "%s\n", buf);
    fflush(g_logFile);
  }
}

// --- small parsing helpers (avoid strtol: glibc 2.38 C23 symbol versioning) -------
static bool ParseFloat(const char* s, float& out) {
  if (!s || !*s) return false;
  char* end = nullptr;
  float v = strtof(s, &end);
  if (end == s) return false;
  out = v;
  return true;
}
static bool ParseBool(const char* s, bool& out) {
  if (!s) return false;
  if (!strcmp(s, "1") || !strcasecmp(s, "true") || !strcasecmp(s, "on") || !strcasecmp(s, "yes")) {
    out = true; return true;
  }
  if (!strcmp(s, "0") || !strcasecmp(s, "false") || !strcasecmp(s, "off") || !strcasecmp(s, "no")) {
    out = false; return true;
  }
  return false;
}
static void ApplyKey(const char* key, const char* val) {
  float f;
  bool b;
  if (!strcmp(key, "separation") && ParseFloat(val, f)) g_cfg.separation = f;
  else if (!strcmp(key, "convergence") && ParseFloat(val, f)) g_cfg.convergence = f;
  else if (!strcmp(key, "swap_eyes") && ParseBool(val, b)) g_cfg.swapEyes = b;
  else if (!strcmp(key, "stereo") && ParseBool(val, b)) g_cfg.stereo = b;
  else if (!strcmp(key, "zoom") && ParseFloat(val, f)) g_cfg.zoom = f;
  else if (!strcmp(key, "hud_depth") && ParseFloat(val, f)) g_cfg.hudDepth = f;
  else if (!strcmp(key, "auto_switch") && ParseBool(val, b)) g_cfg.autoSwitch = b;
  else if (!strcmp(key, "fps_limit") && ParseFloat(val, f)) g_cfg.fpsLimit = f;
  else if (!strcmp(key, "game_sync") && ParseBool(val, b)) g_cfg.gameSync = b;
  else if (!strcmp(key, "own_window") && ParseBool(val, b)) g_cfg.ownWindow = b ? 1 : 0;
  else if (!strcmp(key, "mode"))
    g_cfg.mode = !strcasecmp(val, "replay") ? 1 : !strcasecmp(val, "depth") ? 2 : !strcasecmp(val, "hybrid") ? 3 : 0;
  else if (!strcmp(key, "replay_warmup") && ParseFloat(val, f)) g_cfg.replayWarmup = std::max(0.0f, f);
  else if (!strcmp(key, "crop")) {
    if (!strcasecmp(val, "auto")) g_cfg.crop = -1;
    else if (!strcasecmp(val, "center")) g_cfg.crop = 1;
    else if (!strcasecmp(val, "content")) g_cfg.crop = 2;
    else if (ParseBool(val, b)) g_cfg.crop = b ? 1 : 0;
  }
  else if (!strcmp(key, "hide_extensions") && ParseBool(val, b)) g_cfg.hideExtensions = b;
  else if (!strcmp(key, "patch_shaders") && ParseBool(val, b)) g_cfg.patchShaders = b;
  else if (!strcmp(key, "log_level") && ParseFloat(val, f)) g_cfg.logLevel = int(f);
  else if (!strcmp(key, "hotkeys") && ParseBool(val, b)) g_cfg.hotkeys = b;
  else if (!strcmp(key, "emitter") && ParseBool(val, b)) g_cfg.emitter = b;
  else if (!strcmp(key, "emitter_delay_us") && ParseFloat(val, f)) g_cfg.emitterDelayUs = f;
  else if (!strcmp(key, "emitter_invert") && ParseBool(val, b)) g_cfg.emitterInvert = b;
  else if (!strcmp(key, "emitter_button") && ParseBool(val, b)) g_cfg.emitterButton = b;
  else if (!strcmp(key, "refresh") && ParseFloat(val, f)) g_cfg.refresh = f;
  else if (!strcmp(key, "present_images") && ParseFloat(val, f)) g_cfg.presentImages = std::clamp(int(f), 2, 4);
  else if (!strcmp(key, "test_pattern") && ParseBool(val, b)) g_cfg.testPattern = b;
  else if (!strcmp(key, "record_width") && ParseFloat(val, f)) g_cfg.recordWidth = std::max(0, int(f));
  else if (!strcmp(key, "record_height") && ParseFloat(val, f)) g_cfg.recordHeight = std::max(0, int(f));
  else if (!strcmp(key, "record_fps") && ParseFloat(val, f)) g_cfg.recordFps = std::clamp(f, 10.0f, 240.0f);
  else if (!strcmp(key, "record") && ParseBool(val, b)) g_cfg.recordAtStart = b;
  else if (!strcmp(key, "record_dir")) {}  // used by 3dvision-recorder
  else if (!strcmp(key, "hide_mesh_shaders") && ParseBool(val, b)) g_cfg.hideMeshShaders = b;
  else if (!strcmp(key, "shift_shadows") && ParseBool(val, b)) g_cfg.shiftShadows = b;
  else if (!strcmp(key, "skip_repeats") && ParseBool(val, b)) g_cfg.skipRepeats = b;
  else if (!strcmp(key, "dump_shaders") && ParseBool(val, b)) g_cfg.dumpShaders = b;
  else if (!strcmp(key, "hud_separation") && ParseBool(val, b)) g_cfg.hudSeparation = b;
  else if (!strcmp(key, "flat_menus") && ParseBool(val, b)) g_cfg.flatMenus = b;
  else if (!strcmp(key, "depth_layers")) g_cfg.depthNearest = !strcasecmp(val, "nearest");
  else if (!strcmp(key, "hud_split")) {
    if (!strcasecmp(val, "auto")) g_cfg.hudSplit = 0;
    else if (ParseFloat(val, f)) g_cfg.hudSplit = std::clamp(int(f), 0, 16);
  }
  else if (!strcmp(key, "fix_shadows")) {
    if (!strcasecmp(val, "auto")) g_cfg.fixShadows = 2;
    else if (ParseBool(val, b)) g_cfg.fixShadows = b ? 1 : 0;
  }
  else if (!strcmp(key, "input"))  // what the application presents: a game, or already-stereo video frames
    g_cfg.inputLayout = !strcasecmp(val, "sbs") ? 1 : (!strcasecmp(val, "tab") || !strcasecmp(val, "ou")) ? 2 : 0;
}

static bool LoadFile(const std::string& path) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return false;
  char line[512];
  while (fgets(line, sizeof(line), f)) {
    char* p = line;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '#' || *p == ';' || *p == '\n' || !*p) continue;
    char* eq = strchr(p, '=');
    if (!eq) continue;
    *eq = 0;
    char* k = p;
    char* v = eq + 1;
    auto trim = [](char* s) {
      size_t n = strlen(s);
      while (n && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ' || s[n - 1] == '\t')) s[--n] = 0;
      while (*s == ' ' || *s == '\t') ++s;
      return s;
    };
    ApplyKey(trim(k), trim(v));
  }
  fclose(f);
  return true;
}

std::string ConfigDir() {
  if (const char* x = getenv("XDG_CONFIG_HOME"); x && *x) return std::string(x) + "/3dvision";
  if (const char* h = getenv("HOME"); h && *h) return std::string(h) + "/.config/3dvision";
  return "/tmp/3dvision";
}

static std::string DetectGameName() {
  // Under Proton argv[0] is usually the Windows exe path (Z:\...\Game.exe or C:\...).
  std::string name;
  FILE* f = fopen("/proc/self/cmdline", "rb");
  if (f) {
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = 0;
    // pick the first argument that ends with .exe, else argv[0]
    size_t off = 0;
    std::string first;
    while (off < n) {
      std::string arg(buf + off);
      if (first.empty()) first = arg;
      size_t len = arg.size();
      if (len > 4 && !strcasecmp(arg.c_str() + len - 4, ".exe")) { name = arg; break; }
      off += len + 1;
    }
    if (name.empty()) name = first;
  }
  size_t slash = name.find_last_of("/\\");
  if (slash != std::string::npos) name = name.substr(slash + 1);
  for (char& c : name)
    if (c == ' ' || c == ':' || c == '"' || c == '\'') c = '_';
  if (name.empty()) name = "unknown";
  return name;
}

static void LoadEnv() {
  // NV3D_<KEY>=value for any config key, e.g. NV3D_MODE=alternate NV3D_SEPARATION=0.03
  static const char* kKeys[] = {"separation", "convergence", "swap_eyes", "stereo", "zoom", "hud_depth",
                                "auto_switch", "fps_limit", "game_sync", "own_window", "mode", "crop",
                                "hide_extensions", "patch_shaders", "log_level", "hotkeys", "emitter",
                                "emitter_delay_us", "emitter_invert", "emitter_button", "refresh",
                                "present_images", "test_pattern", "input", "skip_repeats", "shift_shadows", "hide_mesh_shaders", "record_width",
                                "record_height", "record_fps", "record", "replay_warmup", "dump_shaders", "fix_shadows", "hud_separation", "hud_split", "flat_menus", "depth_layers"};
  for (const char* k : kKeys) {
    std::string env = "NV3D_";
    for (const char* c = k; *c; ++c) env += char(toupper(*c));
    if (const char* v = getenv(env.c_str()); v && *v) ApplyKey(k, v);
  }
  // short aliases
  if (const char* v = getenv("NV3D_SEP")) ApplyKey("separation", v);
  if (const char* v = getenv("NV3D_CONV")) ApplyKey("convergence", v);
  if (const char* v = getenv("NV3D_SWAP")) ApplyKey("swap_eyes", v);
  if (const char* v = getenv("NV3D_HUD")) ApplyKey("hud_depth", v);
  if (const char* v = getenv("NV3D_PATCH")) ApplyKey("patch_shaders", v);
  if (const char* v = getenv("NV3D_HIDE_EXT")) ApplyKey("hide_extensions", v);
  if (const char* v = getenv("NV3D_DELAY")) ApplyKey("emitter_delay_us", v);
}

const char* ModeName(int mode) {
  return mode == 1 ? "replay" : mode == 2 ? "depth" : mode == 3 ? "hybrid" : "alternate";
}

void LoadConfig() {
  static bool loaded = false;
  if (loaded) return;
  loaded = true;
  if (const char* lf = getenv("NV3D_LOG"); lf && *lf) g_logFile = fopen(lf, "a");
  g_game = DetectGameName();
  std::string dir = ConfigDir();
  LoadFile(dir + "/default.conf");
  if (const char* p = getenv("NV3D_CONFIG"); p && *p) g_profile = p;
  else g_profile = dir + "/" + g_game + ".conf";
  bool had = LoadFile(g_profile);
  struct stat st;
  if (stat(g_profile.c_str(), &st) == 0) g_profileMtime = st.st_mtime;
  LoadEnv();
  // launchers, overlays and Wine helpers: never touch them
  static const char* kIgnore[] = {"explorer.exe", "xalia.exe", "upc.exe", "UbisoftConnect.exe", "UplayWebCore.exe",
                                  "UbisoftGameLauncher.exe", "UbisoftGameLauncher64.exe", "EpicGamesLauncher.exe",
                                  "EpicWebHelper.exe", "EOSOverlayRenderer-Win64-Shipping.exe",
                                  "EOSOverlayRenderer-Win32-Shipping.exe", "steamwebhelper.exe", "steam.exe",
                                  "EADesktop.exe", "EABackgroundService.exe", "Link2EA.exe", "RockstarService.exe",
                                  "LauncherPatcher.exe", "Launcher.exe", "steamwebhelper", "steam", "services.exe",
                                  "winedevice.exe", "plugplay.exe", "rpcss.exe", "tabtip.exe", "conhost.exe"};
  for (const char* n : kIgnore)
    if (!strcasecmp(g_game.c_str(), n)) g_cfg.passthrough = true;
  if (const char* ig = getenv("NV3D_IGNORE"); ig && strcasestr(ig, g_game.c_str())) g_cfg.passthrough = true;
  if (!g_cfg.passthrough && !g_logFile) {  // always keep a per-game log
    std::string ldir = dir + "/logs";
    for (size_t p = 1; p <= ldir.size(); ++p)  // mkdir -p
      if (p == ldir.size() || ldir[p] == '/') mkdir(ldir.substr(0, p).c_str(), 0755);
    g_logFile = fopen((ldir + "/" + g_game + ".log").c_str(), "w");
  }
  if (g_cfg.passthrough) {
    g_cfg.patchShaders = false;
    g_cfg.ownWindow = 0;
    g_cfg.mode = 0;
    g_cfg.hideExtensions = false;
    g_cfg.fpsLimit = 0;
    g_cfg.emitter = false;
    Log(1, "%s: launcher/helper, 3dvision inactive", g_game.c_str());
    return;
  }
  Log(1, "game=%s profile=%s%s sep=%.4f conv=%.3f swap=%d mode=%s emitter=%d delay=%.0fus", g_game.c_str(),
      g_profile.c_str(), had ? "" : " (new)", g_cfg.separation, g_cfg.convergence, g_cfg.swapEyes,
      ModeName(g_cfg.mode), g_cfg.emitter, g_cfg.emitterDelayUs);
}

void SaveConfig() {
  std::string dir = ConfigDir();
  std::string cmd = dir;
  // mkdir -p (two levels is enough for ~/.config/stereo3d)
  size_t pos = 0;
  while ((pos = cmd.find('/', pos + 1)) != std::string::npos) mkdir(cmd.substr(0, pos).c_str(), 0755);
  mkdir(dir.c_str(), 0755);
  FILE* f = fopen(g_profile.c_str(), "w");
  if (!f) { Log(0, "cannot write %s", g_profile.c_str()); return; }
  fprintf(f, "# 3dvisionlinux profile for %s\n", g_game.c_str());
  fprintf(f, "separation = %.5f\nconvergence = %.5f\nswap_eyes = %d\nstereo = %d\n",
          g_cfg.separation, g_cfg.convergence, g_cfg.swapEyes, g_cfg.stereo);
  fprintf(f, "mode = %s\nzoom = %.4f\nfps_limit = %.1f\nhud_depth = %.4f\nauto_switch = %d\n",
          ModeName(g_cfg.mode), g_cfg.zoom, g_cfg.fpsLimit, g_cfg.hudDepth, g_cfg.autoSwitch);
  fprintf(f, "crop = %s\n", g_cfg.crop < 0 ? "auto" : g_cfg.crop == 1 ? "center" : g_cfg.crop == 2 ? "content" : "0");
  fprintf(f, "emitter_delay_us = %.0f\nemitter_invert = %d\n", g_cfg.emitterDelayUs, g_cfg.emitterInvert);
  fprintf(f, "skip_repeats = %d\nshift_shadows = %d\n", g_cfg.skipRepeats, g_cfg.shiftShadows);
  fprintf(f, "fix_shadows = %s\n", g_cfg.fixShadows == 2 ? "auto" : g_cfg.fixShadows ? "1" : "0");
  fprintf(f, "hud_separation = %d\n", g_cfg.hudSeparation ? 1 : 0);
  if (g_cfg.hudSplit > 0) fprintf(f, "hud_split = %d\n", g_cfg.hudSplit);
  else fprintf(f, "hud_split = auto\n");
  fprintf(f, "flat_menus = %d\n", g_cfg.flatMenus ? 1 : 0);
  fprintf(f, "depth_layers = %s\n", g_cfg.depthNearest ? "nearest" : "latest");
  fclose(f);
  struct stat st;
  if (stat(g_profile.c_str(), &st) == 0) g_profileMtime = st.st_mtime;
  Log(1, "saved %s", g_profile.c_str());
}

// --- X11 hotkeys via dlopen (no link-time dependency) ---------------------------------
namespace {
struct X11 {
  void* lib = nullptr;
  void* dpy = nullptr;
  void* (*OpenDisplay)(const char*) = nullptr;
  int (*QueryKeymap)(void*, char[32]) = nullptr;
  unsigned char (*KeysymToKeycode)(void*, unsigned long) = nullptr;
  bool tried = false;
  bool ok = false;
  unsigned char kc[32] = {};  // (one per Key below)
};
X11 g_x;

enum Key { kCtrlL, kCtrlR, kShiftL, kShiftR, kF3, kF4, kF5, kF6, kF7, kF8, kF9, kF10, kF11,
           kPgUp, kPgDn, kHome, kF1, kF2, kIns, kDel, kEnd, kF12, kPause, kScroll, kCount };
static_assert(kCount <= 32, "X11::kc holds 32 keys");
const unsigned long kSyms[kCount] = {0xFFE3, 0xFFE4, 0xFFE1, 0xFFE2, 0xFFC0, 0xFFC1, 0xFFC2, 0xFFC3,
                                     0xFFC4, 0xFFC5, 0xFFC6, 0xFFC7, 0xFFC8, 0xFF55, 0xFF56, 0xFF50,
                                     0xFFBE, 0xFFBF, 0xFF63, 0xFFFF, 0xFF57, 0xFFC9, 0xFF13, 0xFF14};
bool g_dumpRequested = false;
bool g_recordToggle = false;

bool InitX11() {
  if (g_x.tried) return g_x.ok;
  g_x.tried = true;
  g_x.lib = dlopen("libX11.so.6", RTLD_NOW | RTLD_LOCAL);
  if (!g_x.lib) { Log(1, "hotkeys disabled (libX11 not found)"); return false; }
  g_x.OpenDisplay = (void* (*)(const char*))dlsym(g_x.lib, "XOpenDisplay");
  g_x.QueryKeymap = (int (*)(void*, char[32]))dlsym(g_x.lib, "XQueryKeymap");
  g_x.KeysymToKeycode = (unsigned char (*)(void*, unsigned long))dlsym(g_x.lib, "XKeysymToKeycode");
  if (!g_x.OpenDisplay || !g_x.QueryKeymap || !g_x.KeysymToKeycode) return false;
  g_x.dpy = g_x.OpenDisplay(nullptr);
  if (!g_x.dpy) { Log(1, "hotkeys disabled (no X display)"); return false; }
  for (int i = 0; i < kCount; ++i) g_x.kc[i] = g_x.KeysymToKeycode(g_x.dpy, kSyms[i]);
  g_x.ok = true;
  Log(1, "hotkeys: Ctrl+Shift + F3/F4 sep -/+  F5/F6 conv -/+  F7 swap eyes  F8 3D on/off  F9 save  F10 test pattern  "
         "F11 capture  F12 record video  Ins/Del shutter delay +/-  End shutter invert  PgUp/PgDn zoom  Home zoom=1  F1/F2 HUD depth  Pause shadow fix on/off  "
         "Scroll Lock HUD separation (off / auto / split 1-6)");
  return true;
}

bool Down(const char km[32], int k) {
  unsigned char c = g_x.kc[k];
  return c && (km[c >> 3] & (1 << (c & 7)));
}
}  // namespace

bool TakeDumpRequest(uint64_t frame) {
  static float autoFrame = -2.0f;
  if (autoFrame == -2.0f) {
    autoFrame = -1.0f;
    if (const char* s = getenv("NV3D_DUMP"); s && *s) ParseFloat(s, autoFrame);
  }
  if (autoFrame >= 0 && frame >= uint64_t(autoFrame)) {
    autoFrame = -1.0f;
    return true;
  }
  if (g_dumpRequested) {
    g_dumpRequested = false;
    return true;
  }
  return false;
}

bool TakeRecordToggle() {
  static bool first = true;
  if (first) {  // NV3D_RECORD=1 / record = 1: start right away
    first = false;
    if (g_cfg.recordAtStart) return true;
  }
  if (!g_recordToggle) return false;
  g_recordToggle = false;
  return true;
}

std::string DumpDir() {
  if (const char* d = getenv("NV3D_DUMP_DIR"); d && *d) return d;
  return ConfigDir();
}

bool PollControls() {
  bool changed = false;
  using clk = std::chrono::steady_clock;
  static clk::time_point lastFileCheck = clk::now();
  static clk::time_point lastKeyPoll = clk::now();
  auto now = clk::now();

  // config file hot reload (once per second)
  if (now - lastFileCheck > std::chrono::seconds(1)) {
    lastFileCheck = now;
    struct stat st;
    if (stat(g_profile.c_str(), &st) == 0 && st.st_mtime != g_profileMtime) {
      g_profileMtime = st.st_mtime;
      LoadFile(g_profile);
      Log(1, "reloaded %s sep=%.4f conv=%.3f", g_profile.c_str(), g_cfg.separation, g_cfg.convergence);
      changed = true;
    }
  }

  if (!g_cfg.hotkeys || !InitX11()) return changed;
  float dt = std::chrono::duration<float>(now - lastKeyPoll).count();
  lastKeyPoll = now;
  if (dt > 0.1f) dt = 0.1f;

  char km[32];
  g_x.QueryKeymap(g_x.dpy, km);
  static bool prev[kCount] = {};
  bool cur[kCount];
  for (int i = 0; i < kCount; ++i) cur[i] = Down(km, i);
  bool mod = (cur[kCtrlL] || cur[kCtrlR]) && (cur[kShiftL] || cur[kShiftR]);
  auto pressed = [&](int k) { return mod && cur[k] && !prev[k]; };
  float rate = 1.0f + 1.2f * dt;  // ~ x2.2 per second while held
  if (mod) {
    if (cur[kF3]) { g_cfg.separation /= rate; changed = true; }
    if (cur[kF4]) { g_cfg.separation *= rate; changed = true; }
    if (cur[kF5]) { g_cfg.convergence /= rate; changed = true; }
    if (cur[kF6]) { g_cfg.convergence *= rate; changed = true; }
    float zr = 1.0f + 0.5f * dt;
    if (cur[kPgUp]) { g_cfg.zoom = std::min(4.0f, g_cfg.zoom * zr); changed = true; }
    if ((cur[kF1] || cur[kF2]) && g_cfg.hudDepth <= 0.0f) g_cfg.hudDepth = g_cfg.convergence;
    if (cur[kF1]) { g_cfg.hudDepth /= rate; changed = true; }   // HUD closer
    if (cur[kF2]) { g_cfg.hudDepth *= rate; changed = true; }   // HUD farther
    if (cur[kPgDn]) { g_cfg.zoom = std::max(0.5f, g_cfg.zoom / zr); changed = true; }
  }
  if (pressed(kF7)) { g_cfg.swapEyes = !g_cfg.swapEyes; changed = true; Log(1, "swap_eyes=%d", g_cfg.swapEyes); }
  if (pressed(kF8)) { g_cfg.stereo = !g_cfg.stereo; changed = true; Log(1, "stereo=%d", g_cfg.stereo); }
  if (pressed(kF9)) SaveConfig();
  if (pressed(kF10)) { g_cfg.testPattern = !g_cfg.testPattern; Log(1, "test_pattern=%d", g_cfg.testPattern); }
  if (pressed(kIns)) { g_cfg.emitterDelayUs += 25; Log(1, "emitter_delay_us=%.0f", g_cfg.emitterDelayUs); }
  if (pressed(kDel)) { g_cfg.emitterDelayUs -= 25; Log(1, "emitter_delay_us=%.0f", g_cfg.emitterDelayUs); }
  if (pressed(kEnd)) { g_cfg.emitterInvert = !g_cfg.emitterInvert; Log(1, "emitter_invert=%d", g_cfg.emitterInvert); }
  if (pressed(kF11)) { g_dumpRequested = true; Log(1, "capture requested"); }
  if (pressed(kF12)) g_recordToggle = true;
  if (pressed(kHome)) { g_cfg.zoom = 1.0f; changed = true; Log(1, "zoom=1"); }
  if (pressed(kPause)) {
    g_cfg.fixShadowsOn = !g_cfg.fixShadowsOn;
    changed = true;
    Log(1, "fix_shadows corrections %s", g_cfg.fixShadowsOn ? "ON" : "OFF");
  }
  if (pressed(kScroll)) {  // HUD separation: off -> automatic -> after draw 1, 2 ... 6 -> off
    if (!g_cfg.hudSeparation) g_cfg.hudSeparation = true, g_cfg.hudSplit = 0;
    else if (g_cfg.hudSplit < 6) ++g_cfg.hudSplit;
    else g_cfg.hudSeparation = false, g_cfg.hudSplit = 0;
    changed = true;
    if (!g_cfg.hudSeparation) Log(1, "HUD separation: off");
    else if (!g_cfg.hudSplit) Log(1, "HUD separation: on, automatic split point");
    else Log(1, "HUD separation: on, copy after back buffer draw %d", g_cfg.hudSplit);
  }
  static bool wasAdjusting = false;
  bool adjusting = mod && (cur[kF3] || cur[kF4] || cur[kF5] || cur[kF6] || cur[kPgUp] || cur[kPgDn] || cur[kF1] || cur[kF2]);
  if (wasAdjusting && !adjusting)
    Log(1, "separation=%.4f convergence=%.3f hud_depth=%.3f zoom=%.3f", g_cfg.separation, g_cfg.convergence,
        g_cfg.hudDepth, g_cfg.zoom);
  wasAdjusting = adjusting;
  for (int i = 0; i < kCount; ++i) prev[i] = cur[i];
  return changed;
}

}  // namespace stereo3d
