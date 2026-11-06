// 3dvisionlinux - runtime configuration, per-game profiles and hotkeys
#pragma once
#include <cstdint>
#include <string>

namespace stereo3d {

struct Config {
  float separation = 0.02f;    // per-eye shift at infinity, in NDC units (1.0 = half screen)
  float convergence = 8.095f;  // view depth (game units) that lands on the screen plane
  bool swapEyes = false;       // flip left/right
  bool stereo = true;          // false = same image to both eyes (2D)
  bool hideExtensions = true;  // hide GPL/shader-object/present-wait etc. so we see every shader
  bool patchShaders = true;    // false = only frame-sequential output (debug)
  int logLevel = 1;            // 0 errors, 1 info, 2 verbose
  bool hotkeys = true;
  bool passthrough = false;    // launcher/helper process: layer does nothing
  int ownWindow = 0;           // 0 present on the game window (default), 1 own full-screen window
  int mode = 1;                // 1 replay (both eyes every game frame, default), 0 alternate (L/R frames),
                               // 2 depth (one game frame -> both eyes, rebuilt from the depth buffer)
  float replayWarmup = 8.0f;   // replay: seconds of alternate at start (intros, shader compilation)
  float fpsLimit = 0.0f;       // extra game fps cap (pairs per second in replay). 0 = only the display sync
  bool gameSync = true;        // the game waits for the display (no frames rendered just to be dropped)
  float hudDepth = 0.0f;       // apparent depth of the HUD (same units as convergence). 0 = screen plane
  bool autoSwitch = true;      // replay -> alternate automatically when the game reads back every frame
  float zoom = 1.0f;           // 1 = whole game image in each eye; >1 zoom in
  int crop = 0;                // 0 off, -1 auto, 1 center 16:9, 2 always detect black bars
  // ---- 3D Vision
  bool emitter = true;         // drive the USB emitter
  float emitterDelayUs = 0.0f; // shift of the shutter signal relative to the vblank (crosstalk tuning)
  bool emitterInvert = false;  // swap which shutter opens (fixes eye order without touching the image)
  bool emitterButton = true;   // the emitter's front button toggles 3D
  float refresh = 0.0f;        // expected refresh rate (Hz). 0 = measure
  int presentImages = 3;       // images of the real swapchain (3 = 2 frames queued, robust)
  bool testPattern = false;    // left eye white "L" side, right eye black: sync/crosstalk test
  bool hideMeshShaders = false;  // hide VK_EXT_mesh_shader so engines use vertex shaders (patched)
  bool shiftShadows = true;    // false: depth-only pipelines (shadow maps) get no stereo shift
  bool skipRepeats = false;    // alternate: ignore presents that only repeat the previous image (emulators)
  int recordWidth = 1920;      // recording: size of each eye in the SBS video (0 = display size)
  int recordHeight = 1080;
  float recordFps = 60.0f;     // constant frame rate of the recording
  bool recordAtStart = false;  // start recording as soon as the game shows (NV3D_RECORD=1)
  int inputLayout = 0;         // 0 game (render each eye), 1 SBS input (left/right halves), 2 top-bottom input
};

Config& Cfg();
float HudFactor();            // 1 - convergence / hud_depth (0 when hud_depth = 0)
void LoadConfig();            // env + config files; call once at startup
const char* ModeName(int mode);
void SaveConfig();            // writes per-game profile
const std::string& GameName();
std::string ProfilePath();
std::string ConfigDir();      // ~/.config/3dvision

void Log(int level, const char* fmt, ...) __attribute__((format(printf, 2, 3)));

// Polls hotkeys (X11, Ctrl+Shift+F-keys) and config-file changes.
// Returns true if something changed.
bool PollControls();

// Frame capture (Ctrl+Shift+F11 or NV3D_DUMP=<frame>): returns true once per request.
bool TakeDumpRequest(uint64_t frame);
// Ctrl+Shift+F12: start / stop recording. Returns true once per key press.
bool TakeRecordToggle();
std::string DumpDir();

}  // namespace stereo3d
