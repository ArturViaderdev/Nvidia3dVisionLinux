// 3dvisionlinux - Vulkan layer: real (geometry) stereoscopic 3D for Linux / Proton games
// shown frame-sequentially (120 Hz, 60 per eye) with NVIDIA 3D Vision shutter glasses.
//
// How it works
//  * Every vertex / tess-eval / geometry shader that reaches the driver is patched so that
//    gl_Position.x += k * (gl_Position.w - convergence)   (not applied when w == 1).
//    k (= ±separation) and convergence live in a tiny GPU buffer read through a
//    buffer-device-address constant baked into the shader.
//  * The game's swapchain is virtual. Each game frame is rendered for both eyes (replay mode:
//    the frame's command buffers are submitted a second time with the other eye's camera;
//    alternate mode: even frames left, odd frames right). Both eye images are copied into a
//    "pair" slot (triple buffered).
//  * A presenter thread owns the real swapchain (FIFO, display at 100-120 Hz) and presents
//    left, right, left, right... on every refresh from the newest complete pair, on its own
//    Vulkan queue so the game's GPU work never delays a flip. If the game is slow the last
//    pair is repeated: the eye order on screen never breaks.
//  * The presenter reports which eye reached the screen at every vblank (VK_KHR_present_wait
//    when available) to the sync engine (emitter.cpp), which drives the 3D Vision USB IR
//    emitter so the right shutter is open.
#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "config.h"
#include "emitter.h"
#include "output.h"
#include "spirv_patch.h"
#include "depth3d_spv.h"

#define LAYER_NAME "VK_LAYER_3DVISION_linux"
#define EXPORT extern "C" __attribute__((visibility("default")))

using namespace stereo3d;
using Clock = std::chrono::steady_clock;

namespace {

// ------------------------------------------------------------------------------------
// Dispatch tables
// ------------------------------------------------------------------------------------
#define INST_FUNCS(X)                                                                     \
  X(DestroyInstance) X(EnumeratePhysicalDevices) X(EnumeratePhysicalDeviceGroups)         \
  X(EnumeratePhysicalDeviceGroupsKHR) X(EnumerateDeviceExtensionProperties)               \
  X(GetPhysicalDeviceSurfaceCapabilitiesKHR) X(GetPhysicalDeviceSurfaceCapabilities2KHR)  \
  X(GetPhysicalDeviceSurfacePresentModesKHR) X(GetPhysicalDeviceMemoryProperties)         \
  X(GetPhysicalDeviceQueueFamilyProperties) X(GetPhysicalDeviceFeatures2)                 \
  X(GetPhysicalDeviceFeatures2KHR) X(GetPhysicalDeviceProperties)                         \
  X(GetPhysicalDeviceFormatProperties) X(DestroySurfaceKHR) X(GetPhysicalDeviceSurfaceSupportKHR)          \
  X(GetPhysicalDeviceSurfaceFormatsKHR)

#define DEV_FUNCS(X)                                                                        \
  X(DestroyDevice) X(GetDeviceQueue) X(GetDeviceQueue2) X(CreateShaderModule)               \
  X(DestroyShaderModule) X(CreateGraphicsPipelines) X(QueueSubmit) X(QueueSubmit2)          \
  X(QueueSubmit2KHR) X(QueueWaitIdle) X(QueueBindSparse) X(DeviceWaitIdle)                  \
  X(CreateSwapchainKHR) X(DestroySwapchainKHR) X(GetSwapchainImagesKHR)                     \
  X(AcquireNextImageKHR) X(AcquireNextImage2KHR) X(QueuePresentKHR) X(CreateImage)          \
  X(DestroyImage) X(GetImageMemoryRequirements) X(AllocateMemory) X(FreeMemory)             \
  X(BindImageMemory) X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements)        \
  X(BindBufferMemory) X(MapMemory) X(UnmapMemory) X(GetBufferDeviceAddress)                 \
  X(GetBufferDeviceAddressKHR) X(CreateCommandPool) X(DestroyCommandPool)                   \
  X(AllocateCommandBuffers) X(FreeCommandBuffers) X(BeginCommandBuffer) X(EndCommandBuffer) \
  X(ResetCommandBuffer) X(CmdPipelineBarrier) X(CmdPipelineBarrier2) X(CmdPipelineBarrier2KHR)       \
  X(CmdCopyImage) X(CmdBlitImage) X(CmdCopyImage2) X(CmdCopyImage2KHR) X(CmdBlitImage2)      \
  X(CmdBlitImage2KHR) X(CmdResolveImage) X(CmdResolveImage2) X(CmdResolveImage2KHR) X(CmdExecuteCommands) \
  X(CmdUpdateBuffer) X(CmdCopyBuffer) X(CmdClearColorImage) X(CmdCopyImageToBuffer) X(CreateFence) X(DestroyFence) X(WaitForFences)  \
  X(ResetFences) X(CreateSemaphore) X(DestroySemaphore) X(WaitSemaphores) X(WaitSemaphoresKHR)  \
  X(GetFenceStatus) X(WaitForPresentKHR) X(InvalidateMappedMemoryRanges)                  \
  X(CreateImageView) X(DestroyImageView) X(CmdBeginRendering) X(CmdBeginRenderingKHR)       \
  X(CmdEndRendering) X(CmdEndRenderingKHR) X(CmdDraw) X(CmdDrawIndexed) X(CmdDrawIndirect)  \
  X(CmdDrawIndexedIndirect) X(CmdDrawIndirectCount) X(CmdDrawIndexedIndirectCount)          \
  X(CreateComputePipelines) X(DestroyPipeline) X(CreatePipelineLayout)                      \
  X(DestroyPipelineLayout) X(CmdBindPipeline) X(CmdPushConstants) X(CmdDispatch)            \
  X(CmdFillBuffer) X(CmdCopyBufferToImage) X(CmdBeginRenderPass) X(CmdBeginRenderPass2)            \
  X(CmdSetViewport) X(CmdSetViewportWithCount) X(CmdSetViewportWithCountEXT)

struct InstDispatch {
#define DECL(n) PFN_vk##n n = nullptr;
  INST_FUNCS(DECL)
};
struct DevDispatch {
  PFN_vkGetDeviceProcAddr GetDeviceProcAddr = nullptr;
  DEV_FUNCS(DECL)
#undef DECL
};

inline void* Key(const void* dispatchable) { return *(void* const*)dispatchable; }

struct InstanceData {
  VkInstance instance = VK_NULL_HANDLE;
  PFN_vkGetInstanceProcAddr gipa = nullptr;
  InstDispatch d;
  uint32_t apiVersion = VK_API_VERSION_1_0;
  PFN_vkVoidFunction createXlibSurface = nullptr;  // vkCreateXlibSurfaceKHR (own output window)
};
// VkXlibSurfaceCreateInfoKHR without pulling Xlib into this file
struct XlibSurfaceCI {
  VkStructureType sType;
  const void* pNext;
  VkFlags flags;
  void* dpy;
  unsigned long window;
};
typedef VkResult(VKAPI_PTR* PFN_CreateXlibSurface)(VkInstance, const XlibSurfaceCI*, const VkAllocationCallbacks*,
                                                    VkSurfaceKHR*);

struct QueueInfo {
  uint32_t family = UINT32_MAX;
  std::mutex m;
};

struct VImage {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkCommandBuffer cmd2 = VK_NULL_HANDLE;  // replay mode: right-eye copy
};

// A stereo pair: one image per eye (real swapchain format and size, always in GENERAL layout)
struct Slot {
  VkImage img[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  VkDeviceMemory mem[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  VkFence written = VK_NULL_HANDLE;  // signalled when the game's copy into this pair is done
};
constexpr int kSlots = 3;

// one in-flight frame of the presenter thread
struct PresEntry {
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkSemaphore acq = VK_NULL_HANDLE, done = VK_NULL_HANDLE;
  int slot = -1;         // pair read by this frame (-1 none)
  bool pending = false;  // submitted, `slot` still counted as busy
  bool used = false;     // fence has been submitted at least once
};
constexpr int kRing = 4;

struct Swapchain {
  VkSwapchainKHR handle = VK_NULL_HANDLE;  // == real swapchain handle
  VkFormat format = VK_FORMAT_UNDEFINED;      // game (virtual) format
  VkFormat realFormat = VK_FORMAT_UNDEFINED;  // real swapchain format
  bool formatDiffers = false;
  bool onOwn = false;                         // real swapchain lives on our own output window
  VkExtent2D vext{}, rext{};                  // game image size, display (one eye) size
  VkOffset2D srcOff{};      // region of the game image that holds one eye (crops Proton letterbox)
  VkExtent2D srcExt{};
  VkOffset2D dstOff{};      // where that region goes inside one eye (keeps aspect ratio)
  VkExtent2D dstExt{};
  bool detectOn = false;    // detect black bars from the image content
  int cx0 = 0, cy0 = 0, cx1 = 0, cy1 = 0;     // current game-content rectangle (end exclusive)
  int kx0 = -1, ky0 = -1, kx1 = -1, ky1 = -1;  // candidate rectangle waiting for confirmation
  int candCount = 0;
  uint64_t nextDetect = 0;
  int detections = 0;
  float lastZoom = -1.0f;
  std::vector<VkImage> realImages;
  std::vector<VImage> v;
  VkCommandPool pool = VK_NULL_HANDLE;
  uint32_t poolFamily = UINT32_MAX;
  bool canBlit = true;
  uint32_t nextAcquire = 0;
  Clock::time_point statStart = Clock::now();
  uint64_t statFrames = 0;

  // ---- pairs shared between the game (writer) and the presenter thread
  Slot slots[kSlots];
  std::mutex pm;                 // protects cur/ready/writing/busy and the ring fences
  std::condition_variable pcv;   // presenter took a pair
  int cur = -1;                  // pair on screen
  int ready = -1;                // newest complete pair, not shown yet
  int writing = -1;              // pair being filled (alternate mode: between L and R frames)
  int busy[kSlots] = {};         // presenter copies in flight that read the pair
  uint64_t pairsPublished = 0, pairsShown = 0, statPublished = 0;

  // ---- presenter thread
  VkCommandPool presPool = VK_NULL_HANDLE;
  PresEntry ring[kRing];
  std::thread presThread;
  std::atomic<bool> stop{false};
  std::atomic<int> presError{VK_SUCCESS};  // error to report to the game (out of date, lost...)
  std::atomic<uint64_t> refreshes{0};      // frames presented by the presenter
  struct Recorder* rec = nullptr;          // video recording (Ctrl+Shift+F12)

  // ---- depth mode: both eyes rebuilt from one game frame
  VkBuffer work = VK_NULL_HANDLE;          // colour | keys (2 eyes) | eyes (2) | last real eyes (2, hybrid)
  bool prevValid[2] = {false, false};      // hybrid: the last real image of that eye is in the work buffer
  VkDeviceMemory workMem = VK_NULL_HANDLE;
  VkDeviceAddress workAddr = 0;
  VkImage synth[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};  // the two eyes, game format and size
  VkDeviceMemory synthMem[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  bool depthBroken = false;
};

struct ModuleEntry {
  std::vector<uint32_t> code;
  struct Variant { uint32_t model; std::string entry; VkShaderModule patched; };
  std::vector<Variant> variants;
};

struct DeviceData {
  VkDevice device = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  InstanceData* inst = nullptr;
  DevDispatch d;
  PFN_vkSetDeviceLoaderData setLoaderData = nullptr;
  uint32_t gfxFamily = UINT32_MAX;
  uint32_t apiVersion = VK_API_VERSION_1_0;
  VkPhysicalDeviceMemoryProperties memProps{};

  std::mutex queueLock;
  std::unordered_map<VkQueue, std::unique_ptr<QueueInfo>> queues;
  VkQueue gfxQueue = VK_NULL_HANDLE;
  VkQueue presQueue = VK_NULL_HANDLE;   // our own queue for the presenter thread
  int32_t presQueueIndex = -1;          // index in gfxFamily (-1: shared with the game)
  bool presentWait = false;             // VK_KHR_present_id + VK_KHR_present_wait enabled by us
  bool recordResume = false;            // a recording was running on a destroyed swapchain

  OutputWindow* out = nullptr;     // own window on the glasses monitor
  VkSurfaceKHR outSurface = VK_NULL_HANDLE;
  int outState = 0;                // 0 undecided, 1 using own window, -1 using the game window

  bool stereoShaders = false;
  VkBuffer paramBuf = VK_NULL_HANDLE;
  VkDeviceMemory paramMem = VK_NULL_HANDLE;
  VkDeviceAddress paramAddr = 0;

  std::mutex shaderLock;
  std::unordered_map<VkShaderModule, ModuleEntry> modules;
  std::unordered_map<VkShaderModule, uint64_t> moduleHash;  // dump_shaders: module -> dumped file
  std::atomic<uint32_t> patchedCount{0}, skippedCount{0}, pipelineCount{0}, depthOnlyCount{0}, screenFixCount{0};
  bool screenFix = false;    // fix_shadows: correct the screen-space position rebuilds (fragment/compute)
  bool unshiftD16 = false;   // fix_shadows: 16-bit depth-only pipelines (Unreal shadow maps) unshifted
  uint32_t lastFixLog[2] = {UINT32_MAX, UINT32_MAX};

  std::mutex swapLock;
  std::unordered_map<VkSwapchainKHR, Swapchain*> swapchains;
  uint64_t frame = 0;  // presents to virtual swapchains; even = left eye
  Clock::time_point lastPresent{};  // for the frame limiter

  // ---- replay mode: the frame's command buffers are submitted a second time for the right eye
  std::atomic<bool> replay{false};
  std::atomic<const char*> stage{"init"};     // what the layer is doing (for the stall watchdog)
  std::thread watchdog;
  std::atomic<bool> stopWatchdog{false};
  int badReplayWindows = 0;
  std::mutex recLock;                         // held by submits on the present queue and by present
  std::vector<VkCommandBuffer> recCbs;        // this frame's command buffers, in order
  std::unordered_map<VkSemaphore, uint64_t> heldTimeline;  // signals delayed until after the replay
  std::vector<VkFence> heldFences;
  bool recMono = false;                       // replay impossible this frame (app needed a signal early)
  std::mutex semLock;
  std::unordered_set<VkSemaphore> timelineSems;
  std::atomic<uint32_t> monoFrames{0}, replayFrames{0};
  // skip_repeats: GPU work submitted since the last present (command buffers), to recognise
  // presents that only show the previous image again (emulators presenting faster than the game)
  std::atomic<uint64_t> work{0};
  double workEma = 0;
  std::atomic<uint32_t> repeatFrames{0};
  std::atomic<uint64_t> presentCount{0};
  std::atomic<int> swapCount{0};
  std::atomic<uint64_t> lastAbortPresent{0};
  // replay warm-up: the first seconds run in alternate (intros, shader compilation, loading)
  bool warmupPending = false;
  Clock::time_point firstPresent{};
  Clock::time_point replaySince{};
  void* paramMap = nullptr;  // persistently mapped parameter buffer (projection samples)

  // ---- depth mode
  bool depthMode = false;
  bool hybrid = false;       // mode = hybrid: depthMode machinery, but each frame renders one real eye
  std::atomic<uint64_t> lastDepthFrame{UINT64_MAX};  // frame of the last scene depth copy (menus: none)
  // projection samples are grouped by "stage" of the scene: how many times its depth was cleared
  // in the frame. Some games draw the far scene with one camera, clear the depth and draw the near
  // scene with another (EDF 6): only the last stage is in the depth copy, so only its camera counts.
  std::atomic<uint32_t> calibGroup{0};     // stages so far in the frame being recorded (reset every frame)
  VkImage stageImg[4] = {};                // the depth image of each stage (caller holds depthLock)
  uint32_t curStage = 0;                   // stage of the last scene pass (caller holds depthLock)
  std::atomic<uint64_t> stageFrameAt{UINT64_MAX};  // (MAX: never)   // last frame whose start was seen in the passes
  std::atomic<uint32_t> lastCopyGroup{0};  // stage of the last depth copy
  std::atomic<uint32_t> copiedStages{0};   // stages whose depth was copied in the frame being recorded
  uint64_t lastClassFrame = 0;
  uint32_t depthFlip = 0;                  // the depth is upside down compared with the image (Unity)
  int flipVotes = 0;
  uint64_t lastFlipCheck = 0;
  bool hintedDepth = false;
  bool canCalib = false;     // vertexPipelineStoresAndAtomics enabled
  std::mutex depthLock;
  struct DepthImg { VkFormat fmt; VkExtent2D ext; bool single; };
  std::unordered_map<VkImage, DepthImg> depthImgs;
  struct ViewInfo { VkImage img; uint32_t mip, layer; };
  std::unordered_map<VkImageView, ViewInfo> depthViews;
  // HUD separation (depth / hybrid): the game's own back buffer gets the finished scene first and
  // the HUD on top; a copy taken in between is the scene without HUD (under depthLock)
  struct ColorImg { VkFormat fmt; VkExtent2D ext; uint32_t samples; };
  std::unordered_map<VkImage, ColorImg> colorImgs;
  std::unordered_map<VkImageView, VkImage> colorViews;
  std::unordered_set<VkImage> presentedImgs;   // the layer's virtual swapchain images
  VkExtent2D gameExt{};
  VkFormat scFormat = VK_FORMAT_UNDEFINED;
  VkImage lastColorImg = VK_NULL_HANDLE;       // colour target of the last game-sized pass
  VkImage bbCand = VK_NULL_HANDLE, bbImg = VK_NULL_HANDLE;
  int bbVotes = 0;
  bool bbUsable = false, bbSwapRB = false, bb10 = false;
  // a multisampled back buffer is resolved into this image before the snapshot copy
  VkImage bbResolve = VK_NULL_HANDLE;
  VkDeviceMemory bbResolveMem = VK_NULL_HANDLE;
  VkFormat bbResolveFmt = VK_FORMAT_UNDEFINED;
  VkExtent2D bbResolveExt{};
  uint32_t bbSamples = 1;
  std::vector<std::pair<VkImage, VkDeviceMemory>> retiredImgs;
  VkBuffer snapBuf = VK_NULL_HANDLE;
  VkDeviceMemory snapMem = VK_NULL_HANDLE;
  VkDeviceAddress snapAddr = 0;
  VkDeviceSize snapSize = 0;
  // HUD separation: kSnapSlots copies (the last frames'); the present picks the one of its frame
  VkDeviceSize snapSlotBytes = 0;
  std::atomic<uint32_t> snapNext{0};
  std::atomic<uint64_t> slotFrame[3] = {{0}, {0}, {0}};
  std::atomic<bool> snapDone{false}, bbFilled{false};
  std::atomic<uint32_t> bbDraws{0};  // draws (and copies) into the back buffer in this frame so far
  std::atomic<int64_t> lastSnapNs{0};
  std::atomic<uint64_t> depthFrameAt{UINT64_MAX};  // (MAX: never)  // last frame the scene depth was cleared (HUD frame boundary)
  std::atomic<uint32_t> snapTaken{0};  // copies taken since the last report
  VkCommandBuffer lastBBcb = VK_NULL_HANDLE;
  bool snapBad = false, loggedSnap = false;
  // which draw into the back buffer ends the scene: the copy is taken after draw number 'split'.
  // Automatic (hud_split = 0): starts at 1 and moves on while the copies do not fit the game image
  uint32_t autoSplit = 1, lastSplit = 0, snapTried = 0, snapMatched = 0;
  int lastHudCfg = -1;
  std::atomic<int> traceFrames{0};  // Ctrl+Shift+F11 in depth / hybrid: log the passes of the next frame
  uint64_t snapBadUntil = 0;
  VkBuffer depthBuf = VK_NULL_HANDLE;       // raw depth copy, written inside the game's command buffers
  VkDeviceMemory depthMem = VK_NULL_HANDLE;
  VkDeviceAddress depthAddr = 0;
  VkDeviceSize depthCap = 0;
  VkBuffer metaBuf = VK_NULL_HANDLE;        // [Dw, Dh, format, copies] of the last copy
  VkDeviceMemory metaMem = VK_NULL_HANDLE;
  VkDeviceAddress metaAddr = 0;
  std::vector<std::pair<VkBuffer, VkDeviceMemory>> retiredBufs;  // may still be in recorded command buffers
  std::atomic<float> depthAspect{0.0f};     // game image aspect ratio (swapchain)
  std::atomic<float> depthArea{0.0f};       // game image pixels
  std::atomic<uint32_t> depthCopies{0};
  std::atomic<bool> loggedRenderPass{false};
  VkPipelineLayout synthLayout = VK_NULL_HANDLE;
  VkPipeline synthPipe[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
  // projection of each scene stage (calibGroup): z_ndc = A + B / w
  struct Proj {
    float A = 0, B = 0;
    bool ok = false;
    int switchVotes = 0;                    // fits in a row that disagree with the current projection
    uint32_t inliers = 0, samples = 0;
  } proj[4];
  // the scene's depth buffers: the size/format class with the most draws (ping-pong buffers of
  // the same class all count); other depth targets (reflections, particles...) are not copied
  std::unordered_map<uint64_t, uint64_t> classDraws;  // under depthLock
  std::atomic<uint64_t> mainClass{0};

  uint64_t lastFitFrame = 0;
};

std::mutex g_lock;
std::unordered_map<void*, InstanceData*> g_instByKey;
std::unordered_map<VkPhysicalDevice, InstanceData*> g_instByPhys;
std::unordered_map<void*, DeviceData*> g_devByKey;

InstanceData* Inst(const void* h) {
  std::lock_guard<std::mutex> l(g_lock);
  auto it = g_instByKey.find(Key(h));
  return it == g_instByKey.end() ? nullptr : it->second;
}
InstanceData* InstFromPhys(VkPhysicalDevice p) {
  std::lock_guard<std::mutex> l(g_lock);
  auto it = g_instByPhys.find(p);
  if (it != g_instByPhys.end()) return it->second;
  auto it2 = g_instByKey.find(Key(p));
  if (it2 != g_instByKey.end()) return it2->second;
  if (g_instByKey.size() == 1) return g_instByKey.begin()->second;
  return nullptr;
}
DeviceData* Dev(const void* h) {
  std::lock_guard<std::mutex> l(g_lock);
  auto it = g_devByKey.find(Key(h));
  return it == g_devByKey.end() ? nullptr : it->second;
}

QueueInfo* GetQueueInfo(DeviceData* dd, VkQueue q) {
  std::lock_guard<std::mutex> l(dd->queueLock);
  auto& p = dd->queues[q];
  if (!p) p.reset(new QueueInfo());
  return p.get();
}

// Presentation extensions can never work with the virtual swapchain (the left-eye frame is
// never presented on its own, present IDs / waits would never complete): always hidden.
const char* const kAlwaysHidden[] = {
    "VK_KHR_present_wait",           "VK_KHR_present_wait2",          "VK_KHR_present_id",
    "VK_KHR_present_id2",            "VK_EXT_swapchain_maintenance1", "VK_KHR_swapchain_maintenance1",
    "VK_NV_low_latency2",            "VK_AMD_anti_lag",               "VK_EXT_full_screen_exclusive",
    "VK_GOOGLE_display_timing",      "VK_EXT_present_timing",
    // pipelines built from cached shader identifiers or shader objects carry no SPIR-V to patch
    "VK_EXT_shader_module_identifier", "VK_EXT_shader_object",
};
// Hidden by default (hide_extensions = 0 shows it): library pipelines are patched too, but less tested
const char* const kShaderPathHidden[] = {
    "VK_EXT_graphics_pipeline_library",
};
// Hidden with hide_mesh_shaders = 1: mesh-shader geometry gets no stereo shift (e.g. UE5 Nanite),
// without these extensions VKD3D-Proton reports no mesh shader support and engines fall back
const char* const kMeshHidden[] = {"VK_EXT_mesh_shader", "VK_NV_mesh_shader"};
bool IsHidden(const char* name) {
  for (const char* h : kAlwaysHidden)
    if (!strcmp(h, name)) return true;
  if (Cfg().hideMeshShaders)
    for (const char* h : kMeshHidden)
      if (!strcmp(h, name)) return true;
  if (Cfg().hideExtensions)
    for (const char* h : kShaderPathHidden)
      if (!strcmp(h, name)) return true;
  return false;
}

float EyeK(uint64_t frame) {
  const Config& c = Cfg();
  if (!c.stereo) return 0.0f;
  float sign = (frame & 1) ? 1.0f : -1.0f;  // even frame = left eye
  if (c.swapEyes) sign = -sign;
  return sign * c.separation;
}

// params[4]: the screen-space corrections on (1) or off (0, Ctrl+Shift+Pause to compare)
float ScreenFixFactor() { return (Cfg().fixShadows != 0 && Cfg().fixShadowsOn) ? 1.0f : 0.0f; }

// Unreal Engine games ship as <Project>-Win64-Shipping.exe (also -WinGDK-, -Linux-...)
bool IsUnrealGame() {
  const std::string& g = GameName();
  return g.size() > 13 && strcasecmp(g.c_str() + g.size() - 13, "-Shipping.exe") == 0;
}

uint32_t FindMemType(DeviceData* dd, uint32_t bits, VkMemoryPropertyFlags want,
                     VkMemoryPropertyFlags prefer = 0) {
  for (int pass = 0; pass < 2; ++pass) {
    VkMemoryPropertyFlags f = want | (pass == 0 ? prefer : 0);
    for (uint32_t i = 0; i < dd->memProps.memoryTypeCount; ++i)
      if ((bits & (1u << i)) && (dd->memProps.memoryTypes[i].propertyFlags & f) == f) return i;
  }
  for (uint32_t i = 0; i < dd->memProps.memoryTypeCount; ++i)
    if (bits & (1u << i)) return i;
  return UINT32_MAX;
}

// ------------------------------------------------------------------------------------
// Instance hooks
// ------------------------------------------------------------------------------------
VKAPI_ATTR VkResult VKAPI_CALL hook_CreateInstance(const VkInstanceCreateInfo* ci,
                                                   const VkAllocationCallbacks* alloc,
                                                   VkInstance* pInstance) {
  auto* chain = (VkLayerInstanceCreateInfo*)ci->pNext;
  while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                    chain->function == VK_LAYER_LINK_INFO))
    chain = (VkLayerInstanceCreateInfo*)chain->pNext;
  if (!chain) return VK_ERROR_INITIALIZATION_FAILED;
  PFN_vkGetInstanceProcAddr gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
  auto create = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
  LoadConfig();
  // enable the Xlib surface extension so we can open our own output window on the glasses
  VkResult r = VK_ERROR_EXTENSION_NOT_PRESENT;
  if (Cfg().ownWindow == 1 && getenv("NV3D_ADD_XLIB")) {  // opt-in: games normally already have it
    std::vector<const char*> exts(ci->ppEnabledExtensionNames, ci->ppEnabledExtensionNames + ci->enabledExtensionCount);
    for (const char* want : {"VK_KHR_surface", "VK_KHR_xlib_surface"}) {
      bool have = false;
      for (const char* e : exts) have |= !strcmp(e, want);
      if (!have) exts.push_back(want);
    }
    VkInstanceCreateInfo ci2 = *ci;
    ci2.enabledExtensionCount = uint32_t(exts.size());
    ci2.ppEnabledExtensionNames = exts.data();
    r = create(&ci2, alloc, pInstance);
  }
  if (r != VK_SUCCESS) r = create(ci, alloc, pInstance);
  if (r != VK_SUCCESS) return r;

  auto* id = new InstanceData();
  id->instance = *pInstance;
  id->gipa = gipa;
  if (ci->pApplicationInfo && ci->pApplicationInfo->apiVersion)
    id->apiVersion = ci->pApplicationInfo->apiVersion;
#define LOAD(n) id->d.n = (PFN_vk##n)gipa(*pInstance, "vk" #n);
  INST_FUNCS(LOAD)
#undef LOAD
  id->createXlibSurface = gipa(*pInstance, "vkCreateXlibSurfaceKHR");
  std::lock_guard<std::mutex> l(g_lock);
  g_instByKey[Key(*pInstance)] = id;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL hook_DestroyInstance(VkInstance instance, const VkAllocationCallbacks* a) {
  InstanceData* id = Inst(instance);
  if (!id) return;
  id->d.DestroyInstance(instance, a);
  std::lock_guard<std::mutex> l(g_lock);
  g_instByKey.erase(Key(instance));
  for (auto it = g_instByPhys.begin(); it != g_instByPhys.end();)
    it = (it->second == id) ? g_instByPhys.erase(it) : std::next(it);
  delete id;
}

VKAPI_ATTR VkResult VKAPI_CALL hook_EnumeratePhysicalDevices(VkInstance instance, uint32_t* count,
                                                             VkPhysicalDevice* devs) {
  InstanceData* id = Inst(instance);
  VkResult r = id->d.EnumeratePhysicalDevices(instance, count, devs);
  if (devs && (r == VK_SUCCESS || r == VK_INCOMPLETE)) {
    std::lock_guard<std::mutex> l(g_lock);
    for (uint32_t i = 0; i < *count; ++i) g_instByPhys[devs[i]] = id;
  }
  return r;
}

VkResult EnumGroups(PFN_vkEnumeratePhysicalDeviceGroups fn, InstanceData* id, VkInstance instance,
                    uint32_t* count, VkPhysicalDeviceGroupProperties* props) {
  VkResult r = fn(instance, count, props);
  if (props && (r == VK_SUCCESS || r == VK_INCOMPLETE)) {
    std::lock_guard<std::mutex> l(g_lock);
    for (uint32_t i = 0; i < *count; ++i)
      for (uint32_t j = 0; j < props[i].physicalDeviceCount; ++j)
        g_instByPhys[props[i].physicalDevices[j]] = id;
  }
  return r;
}
VKAPI_ATTR VkResult VKAPI_CALL hook_EnumeratePhysicalDeviceGroups(
    VkInstance instance, uint32_t* count, VkPhysicalDeviceGroupProperties* props) {
  InstanceData* id = Inst(instance);
  return EnumGroups(id->d.EnumeratePhysicalDeviceGroups, id, instance, count, props);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_EnumeratePhysicalDeviceGroupsKHR(
    VkInstance instance, uint32_t* count, VkPhysicalDeviceGroupProperties* props) {
  InstanceData* id = Inst(instance);
  return EnumGroups(id->d.EnumeratePhysicalDeviceGroupsKHR, id, instance, count, props);
}

VKAPI_ATTR VkResult VKAPI_CALL hook_EnumerateDeviceExtensionProperties(
    VkPhysicalDevice phys, const char* layerName, uint32_t* count, VkExtensionProperties* props) {
  if (layerName && !strcmp(layerName, LAYER_NAME)) {
    *count = 0;
    return VK_SUCCESS;
  }
  InstanceData* id = InstFromPhys(phys);
  if (!id) return VK_ERROR_INITIALIZATION_FAILED;
  if (layerName || Cfg().passthrough)
    return id->d.EnumerateDeviceExtensionProperties(phys, layerName, count, props);

  uint32_t n = 0;
  VkResult r = id->d.EnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr);
  if (r != VK_SUCCESS) return r;
  std::vector<VkExtensionProperties> all(n);
  r = id->d.EnumerateDeviceExtensionProperties(phys, nullptr, &n, all.data());
  if (r != VK_SUCCESS && r != VK_INCOMPLETE) return r;
  all.resize(n);
  std::vector<VkExtensionProperties> kept;
  for (auto& e : all)
    if (!IsHidden(e.extensionName)) kept.push_back(e);
  if (!props) {
    *count = uint32_t(kept.size());
    return VK_SUCCESS;
  }
  uint32_t c = std::min<uint32_t>(*count, uint32_t(kept.size()));
  std::copy(kept.begin(), kept.begin() + c, props);
  *count = c;
  return c < kept.size() ? VK_INCOMPLETE : VK_SUCCESS;
}

// ------------------------------------------------------------------------------------
// Stall watchdog: if the game stops presenting, say where we are and release held signals
// ------------------------------------------------------------------------------------
void AbortReplayLocked(DeviceData* dd, const char* why);

// What every thread of the process is doing (name, state, kernel wait channel)
void LogThreads() {
  DIR* d = opendir("/proc/self/task");
  if (!d) return;
  int n = 0;
  while (dirent* e = readdir(d)) {
    if (e->d_name[0] == '.') continue;
    std::string base = std::string("/proc/self/task/") + e->d_name;
    auto readFile = [](const std::string& p) {
      std::string s;
      if (FILE* f = fopen(p.c_str(), "r")) {
        char buf[256];
        size_t k = fread(buf, 1, sizeof(buf) - 1, f);
        buf[k] = 0;
        s = buf;
        fclose(f);
      }
      while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
      return s;
    };
    std::string comm = readFile(base + "/comm"), wchan = readFile(base + "/wchan"), stat = readFile(base + "/stat");
    std::string sys = readFile(base + "/syscall");
    char state = '?';
    size_t rp = stat.rfind(')');
    if (rp != std::string::npos && rp + 2 < stat.size()) state = stat[rp + 2];
    if (sys.size() > 12) sys = sys.substr(0, sys.find(' '));
    Log(0, "  thread %s %-16s state %c wchan %-24s syscall %s", e->d_name, comm.c_str(), state, wchan.c_str(),
        sys.c_str());
    if (++n >= 64) break;
  }
  closedir(d);
}

void WatchdogMain(DeviceData* dd) {
  uint64_t last = dd->presentCount.load();
  auto lastChange = Clock::now();
  int level = 0;
  while (!dd->stopWatchdog.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    uint64_t pc = dd->presentCount.load();
    auto now = Clock::now();
    if (pc != last || dd->swapCount.load() == 0) {
      if (level > 0 && pc != last) Log(0, "stall over, presenting again");
      last = pc;
      lastChange = now;
      level = 0;
      continue;
    }
    double st = std::chrono::duration<double>(now - lastChange).count();
    if ((level == 0 && st > 2.0) || (level == 1 && st > 6.0)) {
      ++level;
      Log(0, "STALL %.0f s without presenting (frame %llu). Layer state: %s. mode=%s", st,
          (unsigned long long)pc, dd->stage.load(), dd->replay ? "replay" : "alternate");
      LogThreads();
      if (dd->replay) {
        if (dd->recLock.try_lock()) {
          size_t held = dd->heldFences.size() + dd->heldTimeline.size() + dd->recCbs.size();
          Log(0, "  held: %zu fences, %zu timeline semaphores, %zu command buffers -> releasing",
              dd->heldFences.size(), dd->heldTimeline.size(), dd->recCbs.size());
          AbortReplayLocked(dd, "watchdog");
          // nothing held back by replay: the game is just busy (loading, compiling shaders);
          // replay is not the cause, keep it
          if (level == 2 && held == 0) Log(0, "  nothing held by replay: the game is busy (loading?), staying in replay");
          bool justStarted = lastChange - dd->replaySince < std::chrono::seconds(3);
          if (level == 2 && (held > 0 || justStarted)) {
            Log(0, "  still stalled: switching to alternate mode");
            dd->replay = false;
            dd->recMono = false;
          }
          dd->recLock.unlock();
        } else {
          Log(0, "  replay lock busy (the present thread is blocked: %s)", dd->stage.load());
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------
// Depth mode: the game renders one mono image per frame; both eyes are rebuilt from it and
// its depth buffer (copied right after the main scene pass inside the game's own command
// buffers). The depth -> distance mapping comes from (clip z, clip w) samples that the
// patched vertex shaders store into the parameter buffer.
// ------------------------------------------------------------------------------------
VkDeviceAddress BufAddr(DeviceData* dd, VkBuffer buf) {
  VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  bai.buffer = buf;
  auto gbda = (dd->apiVersion >= VK_API_VERSION_1_2 && dd->d.GetBufferDeviceAddress)
                  ? dd->d.GetBufferDeviceAddress : dd->d.GetBufferDeviceAddressKHR;
  if (!gbda) gbda = dd->d.GetBufferDeviceAddress;
  return gbda ? gbda(dd->device, &bai) : 0;
}

bool CreateDevBuf(DeviceData* dd, VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer* buf, VkDeviceMemory* mem,
                  VkDeviceAddress* addr) {
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  bci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (dd->d.CreateBuffer(dd->device, &bci, nullptr, buf) != VK_SUCCESS) return false;
  VkMemoryRequirements req;
  dd->d.GetBufferMemoryRequirements(dd->device, *buf, &req);
  VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fi};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = FindMemType(dd, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (ai.memoryTypeIndex == UINT32_MAX || dd->d.AllocateMemory(dd->device, &ai, nullptr, mem) != VK_SUCCESS) {
    dd->d.DestroyBuffer(dd->device, *buf, nullptr);
    *buf = VK_NULL_HANDLE;
    return false;
  }
  dd->d.BindBufferMemory(dd->device, *buf, *mem, 0);
  if (addr) *addr = BufAddr(dd, *buf);
  return true;
}

bool IsDepthFormat(VkFormat f) {
  switch (f) {
    case VK_FORMAT_D16_UNORM: case VK_FORMAT_X8_D24_UNORM_PACK32: case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D16_UNORM_S8_UINT: case VK_FORMAT_D24_UNORM_S8_UINT: case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return true;
    default:
      return false;
  }
}
bool HasStencil(VkFormat f) {
  return f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
}
// how the depth aspect looks in a buffer copy: 0 float32, 1 unorm24 in 32 bits, 2 unorm16
uint32_t DepthCode(VkFormat f) {
  if (f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D16_UNORM_S8_UINT) return 2;
  if (f == VK_FORMAT_X8_D24_UNORM_PACK32 || f == VK_FORMAT_D24_UNORM_S8_UINT) return 1;
  return 0;
}

struct SynthPC {
  uint64_t color, depth, keys, outc, meta;
  uint32_t W, H;
  float a, b, kL, kR, conv, maxShift;
  uint64_t prev;   // hybrid: last real image of each eye (2*W*H), holes are filled from it
  uint32_t hyb;    // 1 = fill holes from prev
  uint64_t stats;  // orientation test counters (host-visible, in the parameter buffer)
  uint64_t snap;   // HUD separation: the game's back buffer before the HUD (W*H)
  uint32_t snapFlags;  // 1 = copies available, 2 = red / blue swapped, 4 = 10-bit copies
  uint32_t snapStride; // words between copies
  uint32_t snapMask;   // slots holding a recent copy
};
constexpr uint32_t kSnapSlots = 3;
constexpr uint32_t kStatsOffset = 3072;  // in the parameter buffer, after the projection samples

bool InitDepthMode(DeviceData* dd) {
  if (!CreateDevBuf(dd, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dd->metaBuf,
                    &dd->metaMem, &dd->metaAddr) || !dd->metaAddr)
    return false;
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SynthPC)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  if (dd->d.CreatePipelineLayout(dd->device, &plci, nullptr, &dd->synthLayout) != VK_SUCCESS) return false;
  const uint32_t* code[3] = {kDepthSplatSpv, kDepthResolveSpv, kDepthMatchSpv};
  const size_t size[3] = {sizeof(kDepthSplatSpv), sizeof(kDepthResolveSpv), sizeof(kDepthMatchSpv)};
  for (int i = 0; i < 3; ++i) {
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = size[i];
    smci.pCode = code[i];
    VkShaderModule m;
    if (dd->d.CreateShaderModule(dd->device, &smci, nullptr, &m) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = m;
    cpci.stage.pName = "main";
    cpci.layout = dd->synthLayout;
    VkResult r = dd->d.CreateComputePipelines(dd->device, VK_NULL_HANDLE, 1, &cpci, nullptr, &dd->synthPipe[i]);
    dd->d.DestroyShaderModule(dd->device, m, nullptr);
    if (r != VK_SUCCESS) return false;
  }
  return true;
}

void DestroyDepthMode(DeviceData* dd) {
  VkDevice dev = dd->device;
  for (auto& p : dd->synthPipe)
    if (p) dd->d.DestroyPipeline(dev, p, nullptr);
  if (dd->synthLayout) dd->d.DestroyPipelineLayout(dev, dd->synthLayout, nullptr);
  auto freeBuf = [&](VkBuffer b, VkDeviceMemory m) {
    if (b) dd->d.DestroyBuffer(dev, b, nullptr);
    if (m) dd->d.FreeMemory(dev, m, nullptr);
  };
  freeBuf(dd->depthBuf, dd->depthMem);
  freeBuf(dd->snapBuf, dd->snapMem);
  freeBuf(dd->metaBuf, dd->metaMem);
  for (auto& r : dd->retiredBufs) freeBuf(r.first, r.second);
  dd->retiredBufs.clear();
  if (dd->bbResolve) dd->retiredImgs.push_back({dd->bbResolve, dd->bbResolveMem});
  dd->bbResolve = VK_NULL_HANDLE;
  dd->bbResolveMem = VK_NULL_HANDLE;
  for (auto& r : dd->retiredImgs) {
    dd->d.DestroyImage(dev, r.first, nullptr);
    if (r.second) dd->d.FreeMemory(dev, r.second, nullptr);
  }
  dd->retiredImgs.clear();
}

// the depth copy target, big enough for the game image at up to 1.5x its size in each direction
void EnsureDepthBuffer(DeviceData* dd, VkExtent2D game) {
  // one copy per scene stage (4); each big enough for up to 1.5x the game image in each direction
  VkDeviceSize need = (VkDeviceSize(game.width) * game.height * 4 * 9 / 4 + 255) & ~VkDeviceSize(255);
  std::lock_guard<std::mutex> l(dd->depthLock);
  dd->depthAspect = float(game.width) / float(std::max(1u, game.height));
  dd->depthArea = float(game.width) * float(game.height);
  if (need <= dd->depthCap) return;
  VkBuffer b;
  VkDeviceMemory m;
  VkDeviceAddress a;
  if (!CreateDevBuf(dd, need * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &b, &m, &a)) {
    Log(0, "depth mode: cannot allocate %llu bytes for the depth copies", (unsigned long long)need * 4);
    return;
  }
  // the old one may be referenced by command buffers the game already recorded
  if (dd->depthBuf) dd->retiredBufs.push_back({dd->depthBuf, dd->depthMem});
  dd->depthBuf = b;
  dd->depthMem = m;
  dd->depthAddr = a;
  dd->depthCap = need;
}

// per command buffer: the depth attachment of the current dynamic rendering pass
struct CbDepth {
  VkImage img = VK_NULL_HANDLE;
  VkFormat fmt = VK_FORMAT_UNDEFINED;
  VkExtent2D ext{};
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  uint32_t mip = 0, layer = 0, draws = 0;
  bool active = false, suspending = false, secondary = false;
  bool clears = false;  // the pass clears the depth (loadOp CLEAR)
  int flag = -1;        // params.flags last set in this command buffer (-1: not yet)
  // the scene depth to copy: copied once per stage, just before the game clears it for the next stage
  // or when the command buffer ends (not after every pass: EDF 6 has ~30 scene passes per frame)
  bool pending = false;
  VkImage pImg = VK_NULL_HANDLE;
  VkFormat pFmt = VK_FORMAT_UNDEFINED;
  VkExtent2D pExt{};
  VkImageLayout pLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  uint32_t pMip = 0, pLayer = 0, pStage = 0;
  uint32_t stage = 0;  // scene stage of the current pass
  // the current pass draws into the game's back buffer (HUD separation)
  bool bbPass = false, bbKhr = false, bbCanSplit = false, bbSecondary = false;
  VkImage bbImg = VK_NULL_HANDLE;
  VkImageLayout bbLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkRenderingInfo bbInfo{};
  std::vector<VkRenderingAttachmentInfo> bbColors;
  VkRenderingAttachmentInfo bbDepth{}, bbStencil{};
};
std::mutex g_cbLock;
std::unordered_map<VkCommandBuffer, CbDepth> g_cbDepth;
struct TlsCb { VkCommandBuffer cb = VK_NULL_HANDLE; CbDepth* st = nullptr; DeviceData* dd = nullptr; };
thread_local TlsCb t_cb;

inline TlsCb& CbState(VkCommandBuffer cb) {
  if (t_cb.cb != cb) {
    DeviceData* dd = Dev(cb);
    std::lock_guard<std::mutex> l(g_cbLock);
    // (entries of freed command buffers stay: games reuse their command buffers, the map stays small,
    // and other threads may hold a pointer to an entry)
    t_cb.st = &g_cbDepth[cb];
    t_cb.cb = cb;
    t_cb.dd = dd;
  }
  return t_cb;
}

VKAPI_ATTR VkResult VKAPI_CALL hook_CreateImage(VkDevice device, const VkImageCreateInfo* ci,
                                                const VkAllocationCallbacks* a, VkImage* img) {
  DeviceData* dd = Dev(device);
  bool depth = dd->depthMode && IsDepthFormat(ci->format) && ci->imageType == VK_IMAGE_TYPE_2D &&
               (ci->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  bool color = dd->depthMode && !depth && ci->imageType == VK_IMAGE_TYPE_2D &&
               (ci->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) && ci->mipLevels >= 1;
  if (color) {  // possibly the game's back buffer: the layer may copy it (scene before the HUD)
    VkImageCreateInfo c = *ci;
    c.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;  // (multisampled: the source of a resolve)
    VkResult r = dd->d.CreateImage(device, &c, a, img);
    if (r != VK_SUCCESS) r = dd->d.CreateImage(device, ci, a, img);
    if (r != VK_SUCCESS) return r;
    std::lock_guard<std::mutex> l(dd->depthLock);
    dd->colorImgs[*img] = {ci->format, {ci->extent.width, ci->extent.height}, uint32_t(ci->samples)};
    return r;
  }
  if (!depth) return dd->d.CreateImage(device, ci, a, img);
  VkImageCreateInfo c = *ci;
  c.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;  // so the layer can copy it
  VkResult r = dd->d.CreateImage(device, &c, a, img);
  if (r != VK_SUCCESS) return dd->d.CreateImage(device, ci, a, img);
  std::lock_guard<std::mutex> l(dd->depthLock);
  dd->depthImgs[*img] = {ci->format, {ci->extent.width, ci->extent.height}, ci->samples == VK_SAMPLE_COUNT_1_BIT};
  return r;
}
VKAPI_ATTR void VKAPI_CALL hook_DestroyImage(VkDevice device, VkImage img, const VkAllocationCallbacks* a) {
  DeviceData* dd = Dev(device);
  if (dd->depthMode && img) {
    std::lock_guard<std::mutex> l(dd->depthLock);
    dd->depthImgs.erase(img);
    dd->colorImgs.erase(img);
    if (img == dd->bbImg) dd->bbImg = VK_NULL_HANDLE, dd->bbUsable = false;
    if (img == dd->bbCand) dd->bbCand = VK_NULL_HANDLE;
    if (img == dd->lastColorImg) dd->lastColorImg = VK_NULL_HANDLE;
  }
  dd->d.DestroyImage(device, img, a);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_CreateImageView(VkDevice device, const VkImageViewCreateInfo* ci,
                                                    const VkAllocationCallbacks* a, VkImageView* view) {
  DeviceData* dd = Dev(device);
  VkResult r = dd->d.CreateImageView(device, ci, a, view);
  if (r == VK_SUCCESS && dd->depthMode) {
    std::lock_guard<std::mutex> l(dd->depthLock);
    if (dd->depthImgs.count(ci->image))
      dd->depthViews[*view] = {ci->image, ci->subresourceRange.baseMipLevel, ci->subresourceRange.baseArrayLayer};
    else if (dd->colorImgs.count(ci->image) || dd->presentedImgs.count(ci->image))
      dd->colorViews[*view] = ci->image;
  }
  return r;
}
VKAPI_ATTR void VKAPI_CALL hook_DestroyImageView(VkDevice device, VkImageView view, const VkAllocationCallbacks* a) {
  DeviceData* dd = Dev(device);
  if (dd->depthMode && view) {
    std::lock_guard<std::mutex> l(dd->depthLock);
    dd->depthViews.erase(view);
    dd->colorViews.erase(view);
  }
  dd->d.DestroyImageView(device, view, a);
}

inline uint64_t DepthClass(const CbDepth& st) {
  return (uint64_t(st.ext.width) << 40) | (uint64_t(st.ext.height) << 20) | uint64_t(st.fmt & 0xFFFFF);
}
void BeginRenderingCommon(TlsCb& t, const VkRenderingInfo* ri) {
  CbDepth& st = *t.st;
  DeviceData* dd = t.dd;
  bool resuming = (ri->flags & VK_RENDERING_RESUMING_BIT) != 0;
  if (!resuming) {
    st.active = false;
    st.draws = 0;
  }
  st.suspending = (ri->flags & VK_RENDERING_SUSPENDING_BIT) != 0;
  st.secondary = (ri->flags & VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT) != 0;
  if (resuming || !dd || !dd->depthMode || !ri->pDepthAttachment || !ri->pDepthAttachment->imageView) return;
  std::lock_guard<std::mutex> l(dd->depthLock);
  auto v = dd->depthViews.find(ri->pDepthAttachment->imageView);
  if (v == dd->depthViews.end()) return;
  auto im = dd->depthImgs.find(v->second.img);
  if (im == dd->depthImgs.end() || !im->second.single) return;
  VkExtent2D e{std::max(1u, im->second.ext.width >> v->second.mip), std::max(1u, im->second.ext.height >> v->second.mip)};
  float asp = float(e.width) / float(e.height), want = dd->depthAspect.load();
  // the scene's depth has (about) the shape and size of the game image; shadow maps, reflections
  // and other small passes do not
  // (up to 30% off: a 16:10 or 4:3 scene shown with black bars in a 16:9 window; square shadow maps are 44% off)
  if (want <= 0 || std::fabs(asp / want - 1.0f) > 0.30f) return;
  if (float(e.width) * float(e.height) < 0.4f * dd->depthArea.load()) return;
  VkDeviceSize bytes = VkDeviceSize(e.width) * e.height * (DepthCode(im->second.fmt) == 2 ? 2 : 4);
  if (bytes > dd->depthCap || e.width < 320) return;
  st.img = v->second.img;
  st.fmt = im->second.fmt;
  st.ext = e;
  st.layout = ri->pDepthAttachment->imageLayout;
  st.clears = ri->pDepthAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
  st.mip = v->second.mip;
  st.layer = v->second.layer;
  st.active = true;
  // The game clears its scene depth: a new frame begins (recorded in the game's order, unlike the
  // present: DXVK records that on its own thread, often after the next frame's first draws). The
  // HUD separation takes its next copy in this frame.
  const uint64_t main = dd->mainClass.load();
  if (st.clears && main && main == DepthClass(st)) dd->depthFrameAt = dd->frame;
  if (st.clears && main && main == DepthClass(st) && dd->snapDone.load()) {
    dd->snapDone = false;
    dd->bbFilled = false;
    dd->bbDraws = 0;
  }
}

// right after a scene pass: copy its depth into the layer's buffer, inside the game's command buffer
void RecordDepthCopy(DeviceData* dd, VkCommandBuffer cb, const CbDepth& st, uint32_t stage) {
  VkBuffer dst;
  VkDeviceSize stride;
  {
    std::lock_guard<std::mutex> l(dd->depthLock);
    dst = dd->depthBuf;
    stride = dd->depthCap;
  }
  if (!dst) return;
  const VkImageAspectFlags aspects = VK_IMAGE_ASPECT_DEPTH_BIT | (HasStencil(st.fmt) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
  const bool general = st.layout == VK_IMAGE_LAYOUT_GENERAL;
  const VkImageLayout copyLayout = general ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  const VkPipelineStageFlags DEPTH_STAGES =
      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  ib.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  ib.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  ib.oldLayout = st.layout;
  ib.newLayout = copyLayout;
  ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  ib.image = st.img;
  ib.subresourceRange = {aspects, st.mip, 1, st.layer, 1};
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};  // earlier copies / readers of our buffers
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
  mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  dd->d.CmdPipelineBarrier(cb, DEPTH_STAGES | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 1, &ib);
  // each scene stage (calibGroup) has its own copy: the eyes combine them pixel by pixel
  stage &= 3;
  VkBufferImageCopy r{};
  r.bufferOffset = stride * stage;
  r.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, st.mip, st.layer, 1};
  r.imageExtent = {st.ext.width, st.ext.height, 1};
  dd->d.CmdCopyImageToBuffer(cb, st.img, copyLayout, dst, 1, &r);
  uint32_t meta[3] = {st.ext.width, st.ext.height, DepthCode(st.fmt)};
  dd->d.CmdUpdateBuffer(cb, dd->metaBuf, 16 * stage, sizeof(meta), meta);
  dd->copiedStages |= 1u << stage;
  ib.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  ib.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  ib.oldLayout = copyLayout;
  ib.newLayout = st.layout;
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                           nullptr, 1, &ib);
  ++dd->depthCopies;
  dd->lastDepthFrame = dd->frame;
  dd->lastCopyGroup = stage;
}

// copies the pending scene depth of this command buffer (in its current layout)
void FlushDepthCopy(DeviceData* dd, VkCommandBuffer cb, CbDepth& st) {
  if (!st.pending) return;
  st.pending = false;
  if (dd->traceFrames.load() > 0) Log(1, "  depth copy (stage %u, cb %p)", st.pStage, (void*)cb);
  CbDepth c;
  c.img = st.pImg;
  c.fmt = st.pFmt;
  c.ext = st.pExt;
  c.layout = st.pLayout;
  c.mip = st.pMip;
  c.layer = st.pLayer;
  RecordDepthCopy(dd, cb, c, st.pStage);
}
// the game changes the layout of the pending depth image: follow it (or drop it if discarded)
void NoteDepthBarrier(CbDepth& st, VkImage img, VkImageLayout oldL, VkImageLayout newL) {
  if (!st.pending || img != st.pImg) return;
  if (oldL == VK_IMAGE_LAYOUT_UNDEFINED) st.pending = false;  // contents discarded
  else st.pLayout = newL;
}

// params.flags (projection samples): 1 + stage inside the scene's passes, 0 elsewhere (shadow maps,
// reflections... have other cameras). Recorded in the game's command buffer, outside render passes.
void SetCalibFlag(DeviceData* dd, VkCommandBuffer cb, float v) {
  VkMemoryBarrier pre{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  pre.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  pre.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &pre, 0,
                           nullptr, 0, nullptr);
  dd->d.CmdUpdateBuffer(cb, dd->paramBuf, 12, 4, &v);
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                           nullptr, 0, nullptr);
}

// params.flags = -2: the passes drawing the HUD over the copied scene (HUD separation) render without
// stereo shift (the patched vertex shaders read it), so the HUD has no disparity in the real eyes
// either: the rebuilt eye shows it where the real one does. (outside render passes)
constexpr int kFlagNoShift = -2;
void SetHudFlag(DeviceData* dd, VkCommandBuffer cb, CbDepth& st, bool on) {
  if (!dd->paramBuf || !dd->canCalib) return;
  const int want = on ? kFlagNoShift : 0;
  if (st.flag == want || (!on && st.flag != kFlagNoShift)) return;
  SetCalibFlag(dd, cb, float(want));
  st.flag = want;
}

void EndRenderingCommon(TlsCb& t, VkCommandBuffer cb) {
  CbDepth& st = *t.st;
  if (!st.active || st.suspending) return;
  st.active = false;
  DeviceData* dd = t.dd;
  const uint64_t cls = DepthClass(st);
  {
    std::lock_guard<std::mutex> l(dd->depthLock);
    dd->classDraws[cls] += st.draws + (st.secondary ? 8 : 0);
  }
  const uint64_t main = dd->mainClass.load();
  // a pass with the depth in a read-only layout cannot have changed it: no new copy needed
  const bool readOnly = st.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL ||
                        st.layout == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL ||
                        st.layout == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL ||
                        st.layout == VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
  if ((main == 0 || main == cls) && (st.draws >= 8 || st.secondary) && !readOnly) {
    if (st.pending && st.pImg != st.img) FlushDepthCopy(dd, cb, st);  // another image: keep that one too
    st.pending = true;
    st.pImg = st.img;
    st.pFmt = st.fmt;
    st.pExt = st.ext;
    st.pLayout = st.layout;
    st.pMip = st.mip;
    st.pLayer = st.layer;
    st.pStage = st.stage;
  }
}
// called by the presenter side every few frames: the class with the most draws is the scene
void UpdateMainDepthClass(DeviceData* dd) {
  std::lock_guard<std::mutex> l(dd->depthLock);
  uint64_t best = 0, bestDraws = 0;
  for (auto& kv : dd->classDraws)
    if (kv.second > bestDraws) { best = kv.first; bestDraws = kv.second; }
  if (best && best != dd->mainClass.load()) {
    if (dd->mainClass.load())
      Log(1, "depth mode: scene depth is now %llux%llu", (unsigned long long)(best >> 40),
          (unsigned long long)((best >> 20) & 0xFFFFF));
    dd->mainClass = best;
  }
  dd->classDraws.clear();
}


// ------------------------------------------------------------------------------------
// HUD separation (depth / hybrid). DXVK / VKD3D games draw into their own back buffer: first the
// finished scene (one full-screen draw or a copy), then the HUD on top; at present that back buffer
// is copied into the image the layer presents. The layer learns which image that is, copies it
// right after the scene arrived (splitting the game's pass after its first draw if needed), and
// the eyes are rebuilt from that copy, with what was drawn after it (HUD, crosshair, markers,
// menus) put back on the screen plane.
// ------------------------------------------------------------------------------------
inline bool IsRGBA8(VkFormat f) {
  return f == VK_FORMAT_R8G8B8A8_UNORM || f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_A8B8G8R8_UNORM_PACK32 ||
         f == VK_FORMAT_A8B8G8R8_SRGB_PACK32;
}
inline bool IsBGRA8(VkFormat f) { return f == VK_FORMAT_B8G8R8A8_UNORM || f == VK_FORMAT_B8G8R8A8_SRGB; }
inline bool Is10Bit(VkFormat f) {
  return f == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || f == VK_FORMAT_A2R10G10B10_UNORM_PACK32;
}
// red in the lowest bits of the 32-bit pixel
inline bool LowIsRed(VkFormat f) { return IsRGBA8(f) || f == VK_FORMAT_A2B10G10R10_UNORM_PACK32; }
// a format a back buffer can have (what is shown); HDR scene targets (16-bit float, B10G11R11) are not
inline bool IsDisplayFormat(VkFormat f) {
  return IsRGBA8(f) || IsBGRA8(f) || f == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || f == VK_FORMAT_A2R10G10B10_UNORM_PACK32;
}

bool CreateImageMem(DeviceData* dd, VkImageCreateInfo ici, VkImage* img, VkDeviceMemory* mem);
// an image whose content went into the presented image (caller holds depthLock)
void VoteBackbufferLocked(DeviceData* dd, VkImage cand) {
  if (!cand || dd->presentedImgs.count(cand)) return;
  auto cf = dd->colorImgs.find(cand);
  if (cf == dd->colorImgs.end() || !IsDisplayFormat(cf->second.fmt)) return;
  if (cand == dd->bbCand) ++dd->bbVotes;
  else dd->bbCand = cand, dd->bbVotes = 1;
  if (dd->bbVotes < 3 || cand == dd->bbImg) return;
  dd->bbImg = cand;
  auto it = dd->colorImgs.find(cand);
  if (it == dd->colorImgs.end()) { dd->bbUsable = false; return; }
  const DeviceData::ColorImg& ci = it->second;
  // (8-bit copies, or 10-bit ones turned into 8 bits by the shader, against an 8-bit presented image)
  bool fmtOk = (IsDisplayFormat(ci.fmt) && (IsRGBA8(dd->scFormat) || IsBGRA8(dd->scFormat))) ||
               (ci.fmt == dd->scFormat && IsDisplayFormat(ci.fmt));
  bool extOk = ci.ext.width == dd->gameExt.width && ci.ext.height == dd->gameExt.height;
  dd->bbSamples = ci.samples;
  if (fmtOk && extOk && ci.samples > 1 &&
      (!dd->bbResolve || dd->bbResolveFmt != ci.fmt || dd->bbResolveExt.width != ci.ext.width ||
       dd->bbResolveExt.height != ci.ext.height)) {
    if (dd->bbResolve) dd->retiredImgs.push_back({dd->bbResolve, dd->bbResolveMem});
    dd->bbResolve = VK_NULL_HANDLE;
    dd->bbResolveMem = VK_NULL_HANDLE;
    VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ic.imageType = VK_IMAGE_TYPE_2D;
    ic.format = ci.fmt;
    ic.extent = {ci.ext.width, ci.ext.height, 1};
    ic.mipLevels = 1;
    ic.arrayLayers = 1;
    ic.samples = VK_SAMPLE_COUNT_1_BIT;
    ic.tiling = VK_IMAGE_TILING_OPTIMAL;
    ic.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkImage im = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (CreateImageMem(dd, ic, &im, &mem)) {
      dd->bbResolve = im, dd->bbResolveMem = mem, dd->bbResolveFmt = ci.fmt, dd->bbResolveExt = ci.ext;
    } else {
      if (im) dd->retiredImgs.push_back({im, mem});
    }
  }
  bool resolveOk = ci.samples == 1 || dd->bbResolve;
  dd->bbUsable = fmtOk && extOk && resolveOk && dd->snapBuf;
  dd->bb10 = Is10Bit(ci.fmt) && !Is10Bit(dd->scFormat);
  dd->bbSwapRB = fmtOk && LowIsRed(ci.fmt) != LowIsRed(dd->scFormat);
  Log(1, "HUD separation: the game's back buffer is a %ux%u image (format %d, %u samples)%s", ci.ext.width,
      ci.ext.height, int(ci.fmt), ci.samples,
      !dd->bbUsable ? (!fmtOk ? ": format not supported" : !extOk ? ": another size than the game image, not used"
                                                             : ": could not make its resolve image, not used")
      : !Cfg().hudSeparation ? " (hud_separation = 0: not used)" : "");
}

// copies the back buffer (= the scene without HUD) into snapBuf, outside any render pass
void RecordSnapshot(DeviceData* dd, VkCommandBuffer cb, VkImage img, VkImageLayout layout) {
  VkBuffer dst;
  VkExtent2D ext;
  VkImage rimg;
  uint32_t samples;
  VkDeviceSize slotBytes;
  {
    std::lock_guard<std::mutex> l(dd->depthLock);
    dst = dd->snapBuf;
    ext = dd->gameExt;
    rimg = dd->bbResolve;
    samples = dd->bbSamples;
    slotBytes = dd->snapSlotBytes;
  }
  if (!dst || !slotBytes || layout == VK_IMAGE_LAYOUT_UNDEFINED || (samples > 1 && !rimg) ||
      dd->snapDone.exchange(true))
    return;
  const uint32_t slot = dd->snapNext.fetch_add(1) % kSnapSlots;
  ++dd->snapTaken;
  dd->lastSnapNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch()).count();
  dd->slotFrame[slot] = dd->frame + 1;  // (+1: 0 = never written)
  const VkDeviceSize dstOff = slot * slotBytes;
  if (dd->traceFrames.load() > 0)
    Log(1, "  -> scene copy %u for the HUD separation, after back buffer draw %u%s", slot, dd->bbDraws.load(),
        samples > 1 ? " (resolved)" : "");
  if (samples > 1) {  // multisampled: resolve into the layer's image, then copy that one
    VkImageMemoryBarrier bs[2] = {{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}, {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}};
    const VkImageLayout sl = layout == VK_IMAGE_LAYOUT_GENERAL ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    for (auto& b : bs) {
      b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    bs[0].image = img;
    bs[0].oldLayout = layout;
    bs[0].newLayout = sl;
    bs[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    bs[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bs[1].image = rimg;
    bs[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    bs[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bs[1].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bs[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, bs);
    VkImageResolve rr{};
    rr.srcSubresource = rr.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    rr.extent = {ext.width, ext.height, 1};
    dd->d.CmdResolveImage(cb, img, sl, rimg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rr);
    bs[0].oldLayout = sl;
    bs[0].newLayout = layout;
    bs[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bs[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    bs[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bs[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    bs[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bs[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};  // earlier readers of snapBuf (the eyes' compute)
    mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                             nullptr, 2, bs);
    VkBufferImageCopy r{};
    r.bufferOffset = dstOff;
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {ext.width, ext.height, 1};
    dd->d.CmdCopyImageToBuffer(cb, rimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, 1, &r);
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
    return;
  }
  const VkImageLayout cl = layout == VK_IMAGE_LAYOUT_GENERAL ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  ib.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  ib.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  ib.oldLayout = layout;
  ib.newLayout = cl;
  ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  ib.image = img;
  ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};  // earlier readers of snapBuf (the eyes' compute)
  mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 1, &ib);
  VkBufferImageCopy r{};
  r.bufferOffset = dstOff;
  r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  r.imageExtent = {ext.width, ext.height, 1};
  dd->d.CmdCopyImageToBuffer(cb, img, cl, dst, 1, &r);
  ib.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  ib.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  ib.oldLayout = cl;
  ib.newLayout = layout;
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                           nullptr, 1, &ib);
}
// the copy is taken once 'split' draws went into the back buffer this frame
uint32_t SplitDraw(DeviceData* dd) { return Cfg().hudSplit > 0 ? uint32_t(Cfg().hudSplit) : dd->autoSplit; }
inline bool SnapDue(DeviceData* dd) { return !dd->snapDone.load() && dd->bbDraws.load() >= SplitDraw(dd); }

// before the game's pass begins: does it draw into the presented image (then the last game-sized
// target is the back buffer candidate) or into the back buffer?
void NoteColorPass(TlsCb& t, VkCommandBuffer cb, const VkRenderingInfo* ri, bool khr) {
  CbDepth& st = *t.st;
  DeviceData* dd = t.dd;
  if (ri->flags & VK_RENDERING_RESUMING_BIT) return;  // a resumed pass keeps its state
  st.bbPass = false;
  if (!dd->depthMode || !ri->colorAttachmentCount || !ri->pColorAttachments || !ri->pColorAttachments[0].imageView)
    return;
  const VkRenderingAttachmentInfo& ca = ri->pColorAttachments[0];
  VkImage img;
  bool isBB;
  {
    std::lock_guard<std::mutex> l(dd->depthLock);
    // a multisampled target is resolved into a normal image at the end of the pass: that one is the output
    VkImageView outView = (ca.resolveMode != VK_RESOLVE_MODE_NONE && ca.resolveImageView) ? ca.resolveImageView
                                                                                       : ca.imageView;
    auto v = dd->colorViews.find(outView);
    if (v == dd->colorViews.end()) return;
    img = v->second;
    if (dd->presentedImgs.count(img)) {
      VoteBackbufferLocked(dd, dd->lastColorImg);
      // the game's frame is finished (when there is no scene depth to tell: menus, 2D games)
      if (dd->depthFrameAt.load() == UINT64_MAX || dd->frame > dd->depthFrameAt.load() + 10) {
        dd->snapDone = false;
        dd->bbFilled = false;
        dd->bbDraws = 0;
      }
      return;
    }
    // (only targets in a display format: the game may already be drawing the next frame's HDR scene
    // when the present copy is recorded on another thread)
    auto c = dd->colorImgs.find(img);
    if (c != dd->colorImgs.end() && c->second.ext.width == dd->gameExt.width &&
        c->second.ext.height == dd->gameExt.height && IsDisplayFormat(c->second.fmt))
      dd->lastColorImg = img;
    isBB = img == dd->bbImg && dd->bbUsable && !dd->snapBad && Cfg().hudSeparation;
  }
  if (!isBB) return;
  st.bbPass = true;
  st.bbKhr = khr;
  st.bbImg = img;
  st.bbLayout = ca.imageLayout;
  st.bbCanSplit = !(ri->flags & VK_RENDERING_SUSPENDING_BIT);
  st.bbSecondary = (ri->flags & VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT) != 0;
  // the same pass, continued (for the split after the scene's draw)
  st.bbInfo = *ri;
  st.bbInfo.pNext = nullptr;
  st.bbColors.assign(ri->pColorAttachments, ri->pColorAttachments + ri->colorAttachmentCount);
  for (auto& a : st.bbColors) a.pNext = nullptr, a.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  st.bbInfo.pColorAttachments = st.bbColors.data();
  if (ri->pDepthAttachment) {
    st.bbDepth = *ri->pDepthAttachment;
    st.bbDepth.pNext = nullptr;
    st.bbDepth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    st.bbInfo.pDepthAttachment = &st.bbDepth;
  }
  if (ri->pStencilAttachment) {
    st.bbStencil = *ri->pStencilAttachment;
    st.bbStencil.pNext = nullptr;
    st.bbStencil.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    st.bbInfo.pStencilAttachment = &st.bbStencil;
  }
  // enough of the frame is already in the back buffer (earlier passes, or the scene came in by a copy:
  // CopyResource, a blit) and this pass adds to it: copy now
  if (SnapDue(dd) && ca.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD) RecordSnapshot(dd, cb, img, ca.imageLayout);
  if (dd->snapDone.load()) SetHudFlag(dd, cb, st, true);  // this pass draws HUD only
}

// before a draw: the back buffer pass already has its first draw (the finished scene): copy it now,
// before the HUD, by ending the pass and continuing it
void BeforeDraw(TlsCb& t, VkCommandBuffer cb) {
  CbDepth& st = *t.st;
  if (!st.bbPass || st.bbSecondary) return;
  DeviceData* dd = t.dd;
  if (!st.bbCanSplit || !SnapDue(dd)) {
    ++dd->bbDraws;
    return;
  }
  (st.bbKhr && dd->d.CmdEndRenderingKHR ? dd->d.CmdEndRenderingKHR : dd->d.CmdEndRendering)(cb);
  RecordSnapshot(dd, cb, st.bbImg, st.bbLayout);
  SetHudFlag(dd, cb, st, true);
  (st.bbKhr && dd->d.CmdBeginRenderingKHR ? dd->d.CmdBeginRenderingKHR : dd->d.CmdBeginRendering)(cb, &st.bbInfo);
  ++dd->bbDraws;
}
void AfterColorPass(TlsCb& t, VkCommandBuffer cb) {
  CbDepth& st = *t.st;
  if (!st.bbPass) return;
  st.bbPass = false;
  SetHudFlag(t.dd, cb, st, false);  // (after the pass: the next passes are the game's again)
  if (st.bbCanSplit && SnapDue(t.dd)) RecordSnapshot(t.dd, cb, st.bbImg, st.bbLayout);
}
// the pass's draws are in secondary command buffers (D3D11 games on DXVK may record them on worker
// threads): the first one with draws is the scene; the pass is split after it, as after a first draw
VKAPI_ATTR void VKAPI_CALL hook_CmdExecuteCommands(VkCommandBuffer cb, uint32_t n, const VkCommandBuffer* cbs) {
  TlsCb& t = CbState(cb);
  DeviceData* dd = t.dd;
  CbDepth* st = t.st;
  std::vector<uint32_t> draws(n, 0);
  {
    std::lock_guard<std::mutex> l(g_cbLock);
    for (uint32_t i = 0; i < n; ++i) {
      auto it = g_cbDepth.find(cbs[i]);
      if (it != g_cbDepth.end()) draws[i] = it->second.draws;
    }
  }
  if (dd->traceFrames.load() > 0) {
    std::string line = "    execute " + std::to_string(n) + " secondary command buffer(s), draws:";
    for (uint32_t i = 0; i < n; ++i) line += " " + std::to_string(draws[i]);
    Log(1, "%s", line.c_str());
  }
  // the copy is taken between two of them, once 'split' draws went in (it cannot be inside one)
  uint32_t start = 0;
  for (uint32_t i = 0; i < n; ++i) {
    st->draws += draws[i];
    if (!st->bbPass || !st->bbSecondary) continue;
    if (st->bbCanSplit && draws[i] && SnapDue(dd)) {
      if (i > start) dd->d.CmdExecuteCommands(cb, i - start, cbs + start);
      start = i;
      (st->bbKhr && dd->d.CmdEndRenderingKHR ? dd->d.CmdEndRenderingKHR : dd->d.CmdEndRendering)(cb);
      RecordSnapshot(dd, cb, st->bbImg, st->bbLayout);
      SetHudFlag(dd, cb, *st, true);
      (st->bbKhr && dd->d.CmdBeginRenderingKHR ? dd->d.CmdBeginRenderingKHR : dd->d.CmdBeginRendering)(cb, &st->bbInfo);
    }
    dd->bbDraws += draws[i];
  }
  if (start < n) dd->d.CmdExecuteCommands(cb, n - start, cbs + start);
}
void NoteTransfer(DeviceData* dd, VkImage src, VkImage dst) {
  if (!dd->depthMode) return;
  std::lock_guard<std::mutex> l(dd->depthLock);
  if (dd->traceFrames.load() > 0)
    Log(1, "  transfer %p -> %p%s", (void*)src, (void*)dst, dd->presentedImgs.count(dst) ? " (PRESENTED)" : "");
  if (dd->presentedImgs.count(dst)) {
    VoteBackbufferLocked(dd, src);
    if (dd->depthFrameAt.load() == UINT64_MAX || dd->frame > dd->depthFrameAt.load() + 10) {
      dd->snapDone = false;
      dd->bbFilled = false;
      dd->bbDraws = 0;
    }
  }
  if (dst && dst == dd->bbImg) {  // a copy into the back buffer counts as a draw
    dd->bbFilled = true;
    ++dd->bbDraws;
  }
}
VKAPI_ATTR void VKAPI_CALL hook_CmdCopyImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
                                             VkImageLayout dl, uint32_t n, const VkImageCopy* r) {
  TlsCb& t = CbState(cb);
  NoteTransfer(t.dd, src, dst);
  t.dd->d.CmdCopyImage(cb, src, sl, dst, dl, n, r);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdBlitImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
                                             VkImageLayout dl, uint32_t n, const VkImageBlit* r, VkFilter f) {
  TlsCb& t = CbState(cb);
  NoteTransfer(t.dd, src, dst);
  t.dd->d.CmdBlitImage(cb, src, sl, dst, dl, n, r, f);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdResolveImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
                                                VkImageLayout dl, uint32_t n, const VkImageResolve* r) {
  TlsCb& t = CbState(cb);
  if (t.dd->traceFrames.load() > 0) Log(1, "  resolve %p -> %p", (void*)src, (void*)dst);
  NoteTransfer(t.dd, src, dst);
  t.dd->d.CmdResolveImage(cb, src, sl, dst, dl, n, r);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdResolveImage2(VkCommandBuffer cb, const VkResolveImageInfo2* i) {
  TlsCb& t = CbState(cb);
  if (t.dd->traceFrames.load() > 0) Log(1, "  resolve %p -> %p", (void*)i->srcImage, (void*)i->dstImage);
  NoteTransfer(t.dd, i->srcImage, i->dstImage);
  (t.dd->d.CmdResolveImage2 ? t.dd->d.CmdResolveImage2 : t.dd->d.CmdResolveImage2KHR)(cb, i);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdResolveImage2KHR(VkCommandBuffer cb, const VkResolveImageInfo2* i) {
  TlsCb& t = CbState(cb);
  if (t.dd->traceFrames.load() > 0) Log(1, "  resolve %p -> %p", (void*)i->srcImage, (void*)i->dstImage);
  NoteTransfer(t.dd, i->srcImage, i->dstImage);
  (t.dd->d.CmdResolveImage2KHR ? t.dd->d.CmdResolveImage2KHR : t.dd->d.CmdResolveImage2)(cb, i);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdCopyImage2(VkCommandBuffer cb, const VkCopyImageInfo2* i) {
  TlsCb& t = CbState(cb);
  NoteTransfer(t.dd, i->srcImage, i->dstImage);
  (t.dd->d.CmdCopyImage2 ? t.dd->d.CmdCopyImage2 : t.dd->d.CmdCopyImage2KHR)(cb, i);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdCopyImage2KHR(VkCommandBuffer cb, const VkCopyImageInfo2* i) {
  TlsCb& t = CbState(cb);
  NoteTransfer(t.dd, i->srcImage, i->dstImage);
  (t.dd->d.CmdCopyImage2KHR ? t.dd->d.CmdCopyImage2KHR : t.dd->d.CmdCopyImage2)(cb, i);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdBlitImage2(VkCommandBuffer cb, const VkBlitImageInfo2* i) {
  TlsCb& t = CbState(cb);
  NoteTransfer(t.dd, i->srcImage, i->dstImage);
  (t.dd->d.CmdBlitImage2 ? t.dd->d.CmdBlitImage2 : t.dd->d.CmdBlitImage2KHR)(cb, i);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdBlitImage2KHR(VkCommandBuffer cb, const VkBlitImageInfo2* i) {
  TlsCb& t = CbState(cb);
  NoteTransfer(t.dd, i->srcImage, i->dstImage);
  (t.dd->d.CmdBlitImage2KHR ? t.dd->d.CmdBlitImage2KHR : t.dd->d.CmdBlitImage2)(cb, i);
}


// diagnostics (Ctrl+Shift+F11 in depth / hybrid): the passes of one frame in the log
std::string DescribeImage(DeviceData* dd, VkImageView view, bool depth) {
  char buf[160];
  std::lock_guard<std::mutex> l(dd->depthLock);
  if (depth) {
    auto v = dd->depthViews.find(view);
    if (v == dd->depthViews.end()) return "depth ?";
    auto im = dd->depthImgs.find(v->second.img);
    if (im == dd->depthImgs.end()) return "depth ?";
    snprintf(buf, sizeof(buf), "depth %ux%u fmt %d img %p", im->second.ext.width, im->second.ext.height,
             int(im->second.fmt), (void*)v->second.img);
    return buf;
  }
  auto v = dd->colorViews.find(view);
  if (v == dd->colorViews.end()) {
    snprintf(buf, sizeof(buf), "colour ? (view %p)", (void*)view);
    return buf;
  }
  if (dd->presentedImgs.count(v->second)) return "PRESENTED";
  auto im = dd->colorImgs.find(v->second);
  if (im == dd->colorImgs.end()) return "colour ?";
  snprintf(buf, sizeof(buf), "colour %ux%u fmt %d x%u img %p%s", im->second.ext.width, im->second.ext.height,
           int(im->second.fmt), im->second.samples, (void*)v->second, v->second == dd->bbImg ? " (back buffer)" : "");
  return buf;
}
void TracePass(TlsCb& t, VkCommandBuffer cb, const VkRenderingInfo* ri) {
  DeviceData* dd = t.dd;
  if (dd->traceFrames.load() <= 0) return;
  std::string line = "  pass";
  char buf[64];
  snprintf(buf, sizeof(buf), " cb %p flags %u:", (void*)cb, ri->flags);
  line += buf;
  for (uint32_t i = 0; i < ri->colorAttachmentCount; ++i)
    if (ri->pColorAttachments[i].imageView)
      line += " [" + DescribeImage(dd, ri->pColorAttachments[i].imageView, false) + " load " +
              std::to_string(int(ri->pColorAttachments[i].loadOp)) +
              (ri->pColorAttachments[i].resolveMode && ri->pColorAttachments[i].resolveImageView
                   ? " -> resolved into " + DescribeImage(dd, ri->pColorAttachments[i].resolveImageView, false)
                   : std::string()) + "]";
  if (ri->pDepthAttachment && ri->pDepthAttachment->imageView)
    line += " [" + DescribeImage(dd, ri->pDepthAttachment->imageView, true) + " load " +
            std::to_string(int(ri->pDepthAttachment->loadOp)) + " layout " +
            std::to_string(int(ri->pDepthAttachment->imageLayout)) + "]";
  if (t.st->active) line += " SCENE-DEPTH stage " + std::to_string(t.st->stage);
  Log(1, "%s", line.c_str());
}
void TraceEnd(TlsCb& t) {
  if (t.dd->traceFrames.load() > 0) Log(1, "    end: %u draws", t.st->draws);
}
// before the game's pass begins: projection samples on (tagged with the stage) for the scene's passes,
// off for the other passes with a depth buffer (shadow maps, reflections: other cameras). Written only
// when the value changes in this command buffer: every write costs a full pipeline barrier.
void CalibBeforePass(TlsCb& t, VkCommandBuffer cb, const VkRenderingInfo* ri) {
  CbDepth& st = *t.st;
  DeviceData* dd = t.dd;
  // the game is about to clear (or replace) the scene depth: copy the previous stage first
  if (st.pending && ri->pDepthAttachment && ri->pDepthAttachment->loadOp != VK_ATTACHMENT_LOAD_OP_LOAD &&
      !(ri->flags & VK_RENDERING_RESUMING_BIT)) {
    bool same;
    {
      std::lock_guard<std::mutex> l(dd->depthLock);
      auto v = dd->depthViews.find(ri->pDepthAttachment->imageView);  // (same image?)
      same = v != dd->depthViews.end() && v->second.img == st.pImg;
    }
    if (same) FlushDepthCopy(dd, cb, st);
  }
  if ((ri->flags & VK_RENDERING_RESUMING_BIT) || !dd->canCalib || !dd->paramBuf) return;
  int want;
  const uint64_t main = dd->mainClass.load();
  if (st.active && (main == 0 || main == DepthClass(st))) {
    // A new stage each time the scene depth is cleared (another camera), or when the scene goes on in
    // another depth image. A pass that goes back to an image of an earlier stage without clearing it
    // continues that stage (EDF 6: the main scene, a far-away layer in another image with its own
    // projection, then more of the main scene): its samples and its depth belong to that camera.
    uint32_t g;
    {
      std::lock_guard<std::mutex> l(dd->depthLock);
      g = dd->calibGroup.load();
      // The first stage's image cleared again after the frame went on to later stages and came back:
      // the next frame begins (seen in the game's recording order; the present is recorded on
      // another thread and can come late, in the middle of the next frame's stages)
      if (st.clears && g > dd->curStage && dd->stageImg[1] == st.img) {
        dd->calibGroup = g = 0;
        for (auto& im : dd->stageImg) im = VK_NULL_HANDLE;
        dd->stageFrameAt = dd->frame;
      }
      if (!dd->stageImg[g]) dd->stageImg[g] = st.img;
      if (st.clears || dd->stageImg[g] != st.img) {
        int back = -1;
        if (!st.clears)
          for (int s = int(g); s >= 0; --s)
            if (dd->stageImg[s] == st.img) { back = s; break; }
        if (back >= 0) {
          g = uint32_t(back);
        } else if (g < 3) {
          dd->calibGroup = ++g;
          dd->stageImg[g] = st.img;
        }
      }
    }
    st.stage = g;
    {
      std::lock_guard<std::mutex> l(dd->depthLock);
      dd->curStage = g;
    }
    want = int(1 + g);
  } else if (ri->pDepthAttachment && ri->pDepthAttachment->imageView) {
    want = 0;
  } else {
    return;
  }
  if (st.flag == want) return;
  SetCalibFlag(dd, cb, float(want));
  st.flag = want;
}

VKAPI_ATTR void VKAPI_CALL hook_CmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags src,
                                                   VkPipelineStageFlags dst, VkDependencyFlags dep, uint32_t mc,
                                                   const VkMemoryBarrier* mb, uint32_t bc,
                                                   const VkBufferMemoryBarrier* bb, uint32_t ic,
                                                   const VkImageMemoryBarrier* ib) {
  TlsCb& t = CbState(cb);
  for (uint32_t i = 0; i < ic; ++i) NoteDepthBarrier(*t.st, ib[i].image, ib[i].oldLayout, ib[i].newLayout);
  t.dd->d.CmdPipelineBarrier(cb, src, dst, dep, mc, mb, bc, bb, ic, ib);
}
void NoteDependency(TlsCb& t, const VkDependencyInfo* di) {
  for (uint32_t i = 0; di && i < di->imageMemoryBarrierCount; ++i)
    NoteDepthBarrier(*t.st, di->pImageMemoryBarriers[i].image, di->pImageMemoryBarriers[i].oldLayout,
                     di->pImageMemoryBarriers[i].newLayout);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo* di) {
  TlsCb& t = CbState(cb);
  NoteDependency(t, di);
  (t.dd->d.CmdPipelineBarrier2 ? t.dd->d.CmdPipelineBarrier2 : t.dd->d.CmdPipelineBarrier2KHR)(cb, di);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdPipelineBarrier2KHR(VkCommandBuffer cb, const VkDependencyInfo* di) {
  TlsCb& t = CbState(cb);
  NoteDependency(t, di);
  (t.dd->d.CmdPipelineBarrier2KHR ? t.dd->d.CmdPipelineBarrier2KHR : t.dd->d.CmdPipelineBarrier2)(cb, di);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_EndCommandBuffer(VkCommandBuffer cb) {
  TlsCb& t = CbState(cb);
  if (t.dd->depthMode) FlushDepthCopy(t.dd, cb, *t.st);  // the last stage of the frame
  return t.dd->d.EndCommandBuffer(cb);
}

VKAPI_ATTR void VKAPI_CALL hook_CmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* ri) {
  TlsCb& t = CbState(cb);
  BeginRenderingCommon(t, ri);
  CalibBeforePass(t, cb, ri);
  NoteColorPass(t, cb, ri, false);
  TracePass(t, cb, ri);
  t.dd->d.CmdBeginRendering(cb, ri);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdBeginRenderingKHR(VkCommandBuffer cb, const VkRenderingInfo* ri) {
  TlsCb& t = CbState(cb);
  BeginRenderingCommon(t, ri);
  CalibBeforePass(t, cb, ri);
  NoteColorPass(t, cb, ri, true);
  TracePass(t, cb, ri);
  (t.dd->d.CmdBeginRenderingKHR ? t.dd->d.CmdBeginRenderingKHR : t.dd->d.CmdBeginRendering)(cb, ri);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdEndRendering(VkCommandBuffer cb) {
  TlsCb& t = CbState(cb);
  t.dd->d.CmdEndRendering(cb);
  TraceEnd(t);
  EndRenderingCommon(t, cb);
  AfterColorPass(t, cb);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdEndRenderingKHR(VkCommandBuffer cb) {
  TlsCb& t = CbState(cb);
  (t.dd->d.CmdEndRenderingKHR ? t.dd->d.CmdEndRenderingKHR : t.dd->d.CmdEndRendering)(cb);
  TraceEnd(t);
  EndRenderingCommon(t, cb);
  AfterColorPass(t, cb);
}
void NoteRenderPass(DeviceData* dd) {
  if (!dd->loggedRenderPass.exchange(true))
    Log(0, "depth mode: this game uses classic render passes; only dynamic rendering (DXVK, VKD3D-Proton) "
        "is supported, the eyes will have no depth");
}
VKAPI_ATTR void VKAPI_CALL hook_CmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo* bi,
                                                   VkSubpassContents c) {
  TlsCb& t = CbState(cb);
  NoteRenderPass(t.dd);
  t.dd->d.CmdBeginRenderPass(cb, bi, c);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdBeginRenderPass2(VkCommandBuffer cb, const VkRenderPassBeginInfo* bi,
                                                    const VkSubpassBeginInfo* sbi) {
  TlsCb& t = CbState(cb);
  NoteRenderPass(t.dd);
  t.dd->d.CmdBeginRenderPass2(cb, bi, sbi);
}
// depth / hybrid diagnostics: depth ranges other than [0, 1] put objects (often the player or the
// weapon, so they never go through walls) on another depth scale than the rest of the scene
std::mutex g_vpLock;
std::vector<std::pair<float, float>> g_vpRanges;
void NoteViewports(uint32_t n, const VkViewport* v) {
  for (uint32_t i = 0; i < n; ++i) {
    float a = v[i].minDepth, b = v[i].maxDepth;
    if (a == 0.0f && b == 1.0f) continue;
    std::lock_guard<std::mutex> l(g_vpLock);
    bool seen = false;
    for (auto& r : g_vpRanges) seen |= r.first == a && r.second == b;
    if (seen || g_vpRanges.size() >= 16) continue;
    g_vpRanges.push_back({a, b});
    Log(1, "depth mode: viewport with depth range [%g, %g] (%gx%g)", a, b, v[i].width, std::fabs(v[i].height));
  }
}
VKAPI_ATTR void VKAPI_CALL hook_CmdSetViewport(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkViewport* v) {
  TlsCb& t = CbState(cb);
  NoteViewports(n, v);
  t.dd->d.CmdSetViewport(cb, first, n, v);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdSetViewportWithCount(VkCommandBuffer cb, uint32_t n, const VkViewport* v) {
  TlsCb& t = CbState(cb);
  NoteViewports(n, v);
  (t.dd->d.CmdSetViewportWithCount ? t.dd->d.CmdSetViewportWithCount : t.dd->d.CmdSetViewportWithCountEXT)(cb, n, v);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdSetViewportWithCountEXT(VkCommandBuffer cb, uint32_t n, const VkViewport* v) {
  TlsCb& t = CbState(cb);
  NoteViewports(n, v);
  (t.dd->d.CmdSetViewportWithCountEXT ? t.dd->d.CmdSetViewportWithCountEXT : t.dd->d.CmdSetViewportWithCount)(cb, n, v);
}

VKAPI_ATTR void VKAPI_CALL hook_CmdDraw(VkCommandBuffer cb, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi) {
  TlsCb& t = CbState(cb);
  BeforeDraw(t, cb);
  ++t.st->draws;
  t.dd->d.CmdDraw(cb, vc, ic, fv, fi);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdDrawIndexed(VkCommandBuffer cb, uint32_t ic, uint32_t inst, uint32_t fi, int32_t vo,
                                               uint32_t fin) {
  TlsCb& t = CbState(cb);
  BeforeDraw(t, cb);
  ++t.st->draws;
  t.dd->d.CmdDrawIndexed(cb, ic, inst, fi, vo, fin);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdDrawIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s) {
  TlsCb& t = CbState(cb);
  BeforeDraw(t, cb);
  t.st->draws += std::max(1u, n);
  t.dd->d.CmdDrawIndirect(cb, b, o, n, s);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n,
                                                       uint32_t s) {
  TlsCb& t = CbState(cb);
  BeforeDraw(t, cb);
  t.st->draws += std::max(1u, n);
  t.dd->d.CmdDrawIndexedIndirect(cb, b, o, n, s);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdDrawIndirectCount(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkBuffer cbuf,
                                                     VkDeviceSize co, uint32_t mx, uint32_t s) {
  TlsCb& t = CbState(cb);
  BeforeDraw(t, cb);
  t.st->draws += 8;  // count unknown on the CPU
  t.dd->d.CmdDrawIndirectCount(cb, b, o, cbuf, co, mx, s);
}
VKAPI_ATTR void VKAPI_CALL hook_CmdDrawIndexedIndirectCount(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o,
                                                            VkBuffer cbuf, VkDeviceSize co, uint32_t mx, uint32_t s) {
  TlsCb& t = CbState(cb);
  BeforeDraw(t, cb);
  t.st->draws += 8;
  t.dd->d.CmdDrawIndexedIndirectCount(cb, b, o, cbuf, co, mx, s);
}

// Fit clip z = A' * w + B' to the samples the patched shaders stored (RANSAC: shadow-map or
// other cameras' samples are outliers); then z_ndc = A' + B' / w.
// fits the projection of one scene stage from its samples
void FitStage(DeviceData* dd, DeviceData::Proj& P, std::vector<std::pair<double, double>>& pts, uint32_t stage) {
  P.samples = uint32_t(pts.size());
  if (pts.size() < 8) return;
  static std::mt19937 rng(1234);
  std::uniform_int_distribution<size_t> pick(0, pts.size() - 1);
  auto inl = [&](double A, double B, std::vector<size_t>* out) {
    size_t n = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
      double pred = A * pts[i].first + B;
      if (std::fabs(pts[i].second - pred) <= 2e-4 * (std::fabs(pts[i].second) + std::fabs(B)) + 1e-7) {
        ++n;
        if (out) out->push_back(i);
      }
    }
    return n;
  };
  size_t best = 0;
  double bA = 0, bB = 0;
  for (int it = 0; it < 200; ++it) {
    size_t i = pick(rng), j = pick(rng);
    double w1 = pts[i].first, w2 = pts[j].first;
    if (std::fabs(w1 - w2) <= 1e-5 * (w1 + w2)) continue;
    double A = (pts[i].second - pts[j].second) / (w1 - w2), B = pts[i].second - A * w1;
    size_t n = inl(A, B, nullptr);
    if (n > best) { best = n; bA = A; bB = B; }
  }
  // Several cameras can feed samples (a scope, reflections, a spot light's shadow): keep the
  // current projection while it is about as good as the best, switch only when another one is
  // clearly better several fits in a row. Otherwise the depth scale (the 3D) would jump around.
  if (P.ok) {
    double cA = double(P.A), cB = double(P.B);
    size_t cur = inl(cA, cB, nullptr);
    P.inliers = uint32_t(cur);
    // keep it unless another projection clearly explains more samples (the main camera draws
    // most of the vertices)
    if (cur * 3 >= best * 2 && cur >= 4) {  // still the main one: refine it a little
      P.switchVotes = 0;
      std::vector<size_t> idx;
      inl(cA, cB, &idx);
      double sw = 0, sz = 0, sww = 0, swz = 0, n = double(idx.size());
      for (size_t i : idx) {
        sw += pts[i].first; sz += pts[i].second;
        sww += pts[i].first * pts[i].first; swz += pts[i].first * pts[i].second;
      }
      double den = n * sww - sw * sw;
      if (n >= 6 && std::fabs(den) > 1e-12 * n * sww) {
        double A = (n * swz - sw * sz) / den, B = (sz - A * sw) / n;
        if (std::fabs(B) > 1e-9 && std::fabs(B / cB - 1.0) < 0.01) {
          P.A = float(cA + (A - cA) * 0.2);
          P.B = float(cB + (B - cB) * 0.2);
        }
      }
      return;
    }
    // the camera of a stage can change from one frame to the next (EDF 6 does not clear its depth
    // the same number of times every frame): switch at once when the samples clearly say so
    bool clear = best >= std::max<size_t>(8, pts.size() * 6 / 10) && cur * 4 < pts.size();
    if (!clear && (best < std::max<size_t>(6, pts.size() * 3 / 10) || ++P.switchVotes < 3)) return;
  } else if (best < std::max<size_t>(6, pts.size() * 3 / 10) || std::fabs(bB) < 1e-9) {
    return;
  }
  std::vector<size_t> idx;
  inl(bA, bB, &idx);
  double sw = 0, sz = 0, sww = 0, swz = 0, n = double(idx.size());
  for (size_t i : idx) {
    sw += pts[i].first; sz += pts[i].second;
    sww += pts[i].first * pts[i].first; swz += pts[i].first * pts[i].second;
  }
  double den = n * sww - sw * sw;
  if (std::fabs(den) > 1e-12 * n * sww) {
    double A = (n * swz - sw * sz) / den, B = (sz - A * sw) / n;
    if (std::fabs(B) > 1e-9) { bA = A; bB = B; }
  }
  if (std::fabs(bB) < 1e-9) return;
  Log(1, "depth mode: projection of scene stage %u %s (z = %.6g + %.6g / w, %zu of %zu samples)", stage,
      P.ok ? "changed" : "found", bA, bB, best, pts.size());
  P.A = float(bA);
  P.B = float(bB);
  P.ok = true;
  P.inliers = uint32_t(best);
  P.switchVotes = 0;
}

// samples of every stage are read and fitted separately; the eyes use the stage of each frame's depth copy
void FitProjection(DeviceData* dd) {
  if (!dd->paramMap) return;
  volatile uint64_t* slots = (volatile uint64_t*)((char*)dd->paramMap + kCalibOffset);
  std::vector<std::pair<double, double>> pts[4];  // (w, z) per stage
  for (uint32_t i = 0; i < kCalibSlots; ++i) {
    uint64_t v = slots[i];
    if (!v) continue;
    slots[i] = 0;
    float w, z;
    uint32_t wb = uint32_t(v), zb = uint32_t(v >> 32);
    memcpy(&w, &wb, 4);
    memcpy(&z, &zb, 4);
    if (std::isfinite(w) && std::isfinite(z) && w > 1e-6f) pts[(i / kCalibGroupSlots) & 3].push_back({w, z});
  }
  for (uint32_t g = 0; g < 4; ++g)
    if (!pts[g].empty()) FitStage(dd, dd->proj[g], pts[g], g);
}

// ------------------------------------------------------------------------------------
// Device creation
// ------------------------------------------------------------------------------------
bool CreateParamBuffer(DeviceData* dd) {
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = 4096;  // params + projection sample slots (kCalibOffset)
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (dd->d.CreateBuffer(dd->device, &bci, nullptr, &dd->paramBuf) != VK_SUCCESS) return false;
  VkMemoryRequirements req;
  dd->d.GetBufferMemoryRequirements(dd->device, dd->paramBuf, &req);
  uint32_t type = FindMemType(dd, req.memoryTypeBits,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fi};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = type;
  if (type == UINT32_MAX || dd->d.AllocateMemory(dd->device, &ai, nullptr, &dd->paramMem) != VK_SUCCESS)
    return false;
  dd->d.BindBufferMemory(dd->device, dd->paramBuf, dd->paramMem, 0);
  void* map = nullptr;
  if (dd->d.MapMemory(dd->device, dd->paramMem, 0, VK_WHOLE_SIZE, 0, &map) == VK_SUCCESS) {
    const bool mono = dd->depthMode && !dd->hybrid;  // depth mode renders the game unshifted
    float init[5] = {mono ? 0.0f : EyeK(0), Cfg().convergence, (mono || EyeK(0) == 0) ? 0.0f : HudFactor(),
                     dd->depthMode ? 1.0f : 0.0f, ScreenFixFactor()};
    memset(map, 0, 4096);
    memcpy(map, init, sizeof(init));
    dd->paramMap = map;  // stays mapped: the projection samples are read from it
  }
  VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  bai.buffer = dd->paramBuf;
  auto gbda = (dd->apiVersion >= VK_API_VERSION_1_2 && dd->d.GetBufferDeviceAddress)
                  ? dd->d.GetBufferDeviceAddress : dd->d.GetBufferDeviceAddressKHR;
  if (!gbda) gbda = dd->d.GetBufferDeviceAddress;
  if (!gbda) return false;
  dd->paramAddr = gbda(dd->device, &bai);
  return dd->paramAddr != 0;
}

VKAPI_ATTR VkResult VKAPI_CALL hook_CreateDevice(VkPhysicalDevice phys, const VkDeviceCreateInfo* pci,
                                                 const VkAllocationCallbacks* alloc, VkDevice* pDevice) {
  InstanceData* inst = InstFromPhys(phys);
  auto* chain = (VkLayerDeviceCreateInfo*)pci->pNext;
  while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                    chain->function == VK_LAYER_LINK_INFO))
    chain = (VkLayerDeviceCreateInfo*)chain->pNext;
  if (!chain || !inst) return VK_ERROR_INITIALIZATION_FAILED;
  PFN_vkSetDeviceLoaderData setLoaderData = nullptr;
  for (auto* c = (VkLayerDeviceCreateInfo*)pci->pNext; c; c = (VkLayerDeviceCreateInfo*)c->pNext)
    if (c->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && c->function == VK_LOADER_DATA_CALLBACK)
      setLoaderData = c->u.pfnSetDeviceLoaderData;

  PFN_vkGetInstanceProcAddr gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  PFN_vkGetDeviceProcAddr gdpa = chain->u.pLayerInfo->pfnNextGetDeviceProcAddr;
  chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
  auto create = (PFN_vkCreateDevice)gipa(inst->instance, "vkCreateDevice");

  // ---- can we do shader stereo? (buffer device address + int64) ----
  VkPhysicalDeviceProperties props{};
  inst->d.GetPhysicalDeviceProperties(phys, &props);
  uint32_t api = std::min(props.apiVersion, inst->apiVersion);
  bool supported = false, vtxStores = false;
  auto gpf2 = inst->d.GetPhysicalDeviceFeatures2 ? inst->d.GetPhysicalDeviceFeatures2
                                                 : inst->d.GetPhysicalDeviceFeatures2KHR;
  if (gpf2) {
    VkPhysicalDeviceBufferDeviceAddressFeatures bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &bda};
    gpf2(phys, &f2);
    supported = bda.bufferDeviceAddress && f2.features.shaderInt64;
    vtxStores = f2.features.vertexPipelineStoresAndAtomics;
  }
  // Proton (DXVK / VKD3D-Proton) always uses Vulkan 1.3. For 1.0 apps BDA would need extra
  // instance extensions, so only Vulkan >= 1.1 is handled (NV3D_LEGACY=1 forces it).
  const char* legacy = getenv("NV3D_LEGACY");
  bool apiOk = api >= VK_API_VERSION_1_1 || (legacy && legacy[0] == '1');
  bool wantStereo = Cfg().patchShaders && supported && apiOk;
  // depth mode: patched shaders store projection samples (vertex-stage stores) and the eyes are
  // built by compute shaders (SPIR-V 1.5: Vulkan 1.2)
  const bool depthLike = Cfg().mode == 2 || Cfg().mode == 3;  // depth / hybrid
  bool wantDepth = wantStereo && depthLike && vtxStores && api >= VK_API_VERSION_1_2;
  if (wantStereo && depthLike && !wantDepth)
    Log(0, "%s mode needs Vulkan 1.2 and vertexPipelineStoresAndAtomics: using alternate", ModeName(Cfg().mode));

  VkDeviceCreateInfo ci = *pci;
  std::vector<const char*> exts;
  for (uint32_t i = 0; i < pci->enabledExtensionCount; ++i) {
    bool always = false;
    for (const char* h : kAlwaysHidden) always |= !strcmp(h, pci->ppEnabledExtensionNames[i]);
    if (always && !Cfg().passthrough) {
      Log(1, "not enabling %s (incompatible with the stereo swapchain)", pci->ppEnabledExtensionNames[i]);
      continue;
    }
    exts.push_back(pci->ppEnabledExtensionNames[i]);
  }
  ci.enabledExtensionCount = uint32_t(exts.size());
  ci.ppEnabledExtensionNames = exts.data();
  VkPhysicalDeviceFeatures featCopy{};
  VkPhysicalDeviceBufferDeviceAddressFeatures ourBda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
  if (wantStereo) {
    if (api < VK_API_VERSION_1_2) {
      bool have = false;
      for (const char* e : exts) have |= !strcmp(e, "VK_KHR_buffer_device_address");
      if (!have) exts.push_back("VK_KHR_buffer_device_address");
      ci.enabledExtensionCount = uint32_t(exts.size());
      ci.ppEnabledExtensionNames = exts.data();
    }
    bool haveBda = false, haveI64 = false;
    for (auto* p = (VkBaseOutStructure*)ci.pNext; p; p = p->pNext) {
      switch (p->sType) {
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
          ((VkPhysicalDeviceVulkan12Features*)p)->bufferDeviceAddress = VK_TRUE;
          haveBda = true;
          break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES:
          ((VkPhysicalDeviceBufferDeviceAddressFeatures*)p)->bufferDeviceAddress = VK_TRUE;
          haveBda = true;
          break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2:
          ((VkPhysicalDeviceFeatures2*)p)->features.shaderInt64 = VK_TRUE;
          if (wantDepth) ((VkPhysicalDeviceFeatures2*)p)->features.vertexPipelineStoresAndAtomics = VK_TRUE;
          haveI64 = true;
          break;
        default: break;
      }
    }
    if (!haveI64) {
      if (ci.pEnabledFeatures) featCopy = *ci.pEnabledFeatures;
      featCopy.shaderInt64 = VK_TRUE;
      if (wantDepth) featCopy.vertexPipelineStoresAndAtomics = VK_TRUE;
      ci.pEnabledFeatures = &featCopy;
    }
    if (!haveBda) {
      ourBda.bufferDeviceAddress = VK_TRUE;
      ourBda.pNext = (void*)ci.pNext;
      ci.pNext = &ourBda;
    }
  } else {
    Log(0, "shader stereo NOT available (bda+int64 supported=%d, api>=1.1=%d, patch=%d)",
        supported, apiOk, Cfg().patchShaders);
  }

  // ---- presenter: one extra queue in the graphics family (its own GPU channel, so flips are
  //      never stuck behind the game's rendering) + present id / present wait for timing ----
  uint32_t qfCount = 0;
  inst->d.GetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, nullptr);
  std::vector<VkQueueFamilyProperties> qf(qfCount);
  inst->d.GetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, qf.data());
  std::vector<VkDeviceQueueCreateInfo> qcis(pci->pQueueCreateInfos, pci->pQueueCreateInfos + pci->queueCreateInfoCount);
  std::vector<float> prios;
  uint32_t gfxFam = UINT32_MAX;
  int32_t presIdx = -1;
  for (auto& q : qcis) {
    if (q.queueFamilyIndex < qfCount && (qf[q.queueFamilyIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
      gfxFam = q.queueFamilyIndex;
      if (!Cfg().passthrough && q.flags == 0 && q.queueCount < qf[gfxFam].queueCount && !getenv("NV3D_SHARED_QUEUE")) {
        prios.assign(q.pQueuePriorities, q.pQueuePriorities + q.queueCount);
        prios.push_back(1.0f);
        presIdx = int32_t(q.queueCount);
        q.queueCount += 1;
        q.pQueuePriorities = prios.data();
        ci.pQueueCreateInfos = qcis.data();
      }
      break;
    }
  }
  bool wantSwapchain = false;
  for (const char* e : exts) wantSwapchain |= !strcmp(e, "VK_KHR_swapchain");
  VkPhysicalDevicePresentIdFeaturesKHR pidF{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR};
  VkPhysicalDevicePresentWaitFeaturesKHR pwF{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR};
  bool presentWait = false;
  if (wantSwapchain && !Cfg().passthrough && gpf2 && inst->apiVersion >= VK_API_VERSION_1_1 &&
      !getenv("NV3D_NO_PRESENT_WAIT")) {
    uint32_t n = 0;
    inst->d.EnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> av(n);
    inst->d.EnumerateDeviceExtensionProperties(phys, nullptr, &n, av.data());
    bool hasId = false, hasWait = false;
    for (auto& e : av) {
      hasId |= !strcmp(e.extensionName, "VK_KHR_present_id");
      hasWait |= !strcmp(e.extensionName, "VK_KHR_present_wait");
    }
    if (hasId && hasWait) {
      VkPhysicalDevicePresentWaitFeaturesKHR qw{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR};
      VkPhysicalDevicePresentIdFeaturesKHR qi{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR, &qw};
      VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &qi};
      gpf2(phys, &f2);
      presentWait = qi.presentId && qw.presentWait;
    }
    if (presentWait) {
      for (const char* want : {"VK_KHR_present_id", "VK_KHR_present_wait"}) {
        bool have = false;
        for (const char* e : exts) have |= !strcmp(e, want);
        if (!have) exts.push_back(want);
      }
      bool haveId = false, haveWait = false;  // the app may already chain these structs
      for (auto* p = (VkBaseOutStructure*)ci.pNext; p; p = p->pNext) {
        if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR) {
          ((VkPhysicalDevicePresentIdFeaturesKHR*)p)->presentId = VK_TRUE;
          haveId = true;
        } else if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR) {
          ((VkPhysicalDevicePresentWaitFeaturesKHR*)p)->presentWait = VK_TRUE;
          haveWait = true;
        }
      }
      if (!haveId) {
        pidF.presentId = VK_TRUE;
        pidF.pNext = (void*)ci.pNext;
        ci.pNext = &pidF;
      }
      if (!haveWait) {
        pwF.presentWait = VK_TRUE;
        pwF.pNext = (void*)ci.pNext;
        ci.pNext = &pwF;
      }
    }
  }
  ci.enabledExtensionCount = uint32_t(exts.size());
  ci.ppEnabledExtensionNames = exts.data();

  VkResult r = create(phys, &ci, alloc, pDevice);
  if (r != VK_SUCCESS && (presentWait || presIdx >= 0)) {
    Log(0, "device creation with the presenter queue / present wait failed (%d), retrying without", r);
    presentWait = false;
    presIdx = -1;
    VkDeviceCreateInfo ci2 = ci;
    ci2.pQueueCreateInfos = pci->pQueueCreateInfos;
    std::vector<const char*> e2;
    for (const char* e : exts)
      if (strcmp(e, "VK_KHR_present_id") && strcmp(e, "VK_KHR_present_wait")) e2.push_back(e);
    ci2.enabledExtensionCount = uint32_t(e2.size());
    ci2.ppEnabledExtensionNames = e2.data();
    // drop our feature structs (they are at the front of the chain)
    while (ci2.pNext && (ci2.pNext == &pidF || ci2.pNext == &pwF)) ci2.pNext = ((VkBaseInStructure*)ci2.pNext)->pNext;
    r = create(phys, &ci2, alloc, pDevice);
  }
  if (r != VK_SUCCESS) return r;

  auto* dd = new DeviceData();
  dd->device = *pDevice;
  dd->phys = phys;
  dd->inst = inst;
  dd->setLoaderData = setLoaderData;
  dd->apiVersion = api;
  dd->d.GetDeviceProcAddr = gdpa;
#define LOAD(n) dd->d.n = (PFN_vk##n)gdpa(*pDevice, "vk" #n);
  DEV_FUNCS(LOAD)
#undef LOAD
  inst->d.GetPhysicalDeviceMemoryProperties(phys, &dd->memProps);

  dd->gfxFamily = gfxFam;
  dd->presentWait = presentWait && dd->d.WaitForPresentKHR;
  if (presIdx >= 0) {
    VkQueue q = VK_NULL_HANDLE;
    dd->d.GetDeviceQueue(*pDevice, gfxFam, uint32_t(presIdx), &q);
    if (q) {
      if (setLoaderData) setLoaderData(*pDevice, q);
      else *(void**)q = *(void**)*pDevice;
      dd->presQueue = q;
      dd->presQueueIndex = presIdx;
      GetQueueInfo(dd, q)->family = gfxFam;
    }
  }
  if (!Cfg().passthrough)
    Log(1, "presenter queue: %s, present wait: %s", dd->presQueue ? "own queue" : "shared with the game (less robust)",
        dd->presentWait ? "yes (exact eye sync)" : "no (eye sync from vsync timing)");

  if (wantStereo) {
    dd->depthMode = wantDepth;
    dd->hybrid = wantDepth && Cfg().mode == 3;
    dd->canCalib = wantDepth;
    dd->stereoShaders = CreateParamBuffer(dd);
    // fix_shadows: screen-space rebuilds corrected in every game (the pattern is specific); shadow
    // maps left unshifted by their 16-bit depth format only in Unreal Engine games (or fix_shadows = 1)
    dd->screenFix = dd->stereoShaders && Cfg().fixShadows != 0;
    dd->unshiftD16 = dd->screenFix && (Cfg().fixShadows == 1 || IsUnrealGame());
    if (dd->unshiftD16) Log(1, "fix_shadows: Unreal Engine shadow maps (16-bit depth) without stereo shift");
    bool replay = dd->stereoShaders && Cfg().mode == 1;
    // replay starts after a warm-up in alternate: games that would hang at start in replay
    // (intros, shader compilation, loading screens that wait for the GPU) get going first
    dd->warmupPending = replay && Cfg().replayWarmup > 0;
    dd->replay = replay && !dd->warmupPending;
    if (dd->replay) dd->replaySince = Clock::now();
    if (!dd->stereoShaders) { Log(0, "failed to create stereo parameter buffer"); dd->depthMode = false; }
    if (dd->depthMode && !InitDepthMode(dd)) {
      Log(0, "depth mode: setup failed, using alternate");
      dd->depthMode = false;
    }
  }
  Log(1, "device '%s' api %u.%u: shader stereo %s, mode %s, param @0x%llx", props.deviceName,
      VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api), dd->stereoShaders ? "ON" : "OFF",
      dd->hybrid ? "hybrid (alternate L/R frames, the other eye of each frame rebuilt from its depth)"
      : dd->depthMode ? "depth (both eyes from each game frame + its depth buffer)"
      : dd->replay ? "replay (both eyes every frame)"
      : dd->warmupPending ? "replay (after a warm-up in alternate)" : "alternate (L/R frames)",
      (unsigned long long)dd->paramAddr);

  if (!Cfg().passthrough) dd->watchdog = std::thread(WatchdogMain, dd);
  std::lock_guard<std::mutex> l(g_lock);
  g_devByKey[Key(*pDevice)] = dd;
  return VK_SUCCESS;
}

struct Swapchain;
void StopPresenter(DeviceData* dd, Swapchain* sc);
struct Recorder;
void DestroyRecorder(Recorder* r);
Recorder* CreateRecorder(DeviceData* dd, Swapchain* sc);

VKAPI_ATTR void VKAPI_CALL hook_DestroyDevice(VkDevice device, const VkAllocationCallbacks* a) {
  DeviceData* dd = Dev(device);
  if (!dd) return;
  dd->stopWatchdog = true;
  if (dd->watchdog.joinable()) dd->watchdog.join();
  {  // swapchains the app did not destroy: stop their presenter threads
    std::lock_guard<std::mutex> l(dd->swapLock);
    for (auto& kv : dd->swapchains) {
      DestroyRecorder(kv.second->rec);
      kv.second->rec = nullptr;
      StopPresenter(dd, kv.second);
    }
  }
  {
    std::lock_guard<std::mutex> l(dd->shaderLock);
    for (auto& m : dd->modules)
      for (auto& v : m.second.variants)
        if (v.patched) dd->d.DestroyShaderModule(device, v.patched, nullptr);
    dd->modules.clear();
  }
  if (dd->outSurface) dd->inst->d.DestroySurfaceKHR(dd->inst->instance, dd->outSurface, nullptr);
  if (dd->out) DestroyOutputWindow(dd->out);
  DestroyDepthMode(dd);
  if (dd->paramBuf) dd->d.DestroyBuffer(device, dd->paramBuf, nullptr);
  if (dd->paramMem) dd->d.FreeMemory(device, dd->paramMem, nullptr);
  dd->d.DestroyDevice(device, a);
  bool last;
  {
    std::lock_guard<std::mutex> l(g_lock);
    g_devByKey.erase(Key(device));
    last = g_devByKey.empty();
  }
  delete dd;
  if (last) SyncStop();
}

void RegisterQueue(DeviceData* dd, VkQueue q, uint32_t family) {
  QueueInfo* qi = GetQueueInfo(dd, q);
  qi->family = family;
  if (family == dd->gfxFamily && !dd->gfxQueue) dd->gfxQueue = q;
}

VKAPI_ATTR void VKAPI_CALL hook_GetDeviceQueue(VkDevice device, uint32_t family, uint32_t index, VkQueue* q) {
  DeviceData* dd = Dev(device);
  dd->d.GetDeviceQueue(device, family, index, q);
  if (*q) RegisterQueue(dd, *q, family);
}
VKAPI_ATTR void VKAPI_CALL hook_GetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2* info, VkQueue* q) {
  DeviceData* dd = Dev(device);
  dd->d.GetDeviceQueue2(device, info, q);
  if (*q) RegisterQueue(dd, *q, info->queueFamilyIndex);
}

// ------------------------------------------------------------------------------------
// Queue ops. In replay mode the frame's command buffers are recorded and the timeline
// signals / fences of the present queue are held back until the right-eye replay is done.
// ------------------------------------------------------------------------------------
bool IsTimeline(DeviceData* dd, VkSemaphore sem) {
  std::lock_guard<std::mutex> l(dd->semLock);
  return dd->timelineSems.count(sem) != 0;
}

// caller holds dd->recLock and the queue lock of q
void ReleaseHeldLocked(DeviceData* dd, VkQueue q) {
  if (dd->heldTimeline.empty() && dd->heldFences.empty()) return;
  std::vector<VkSemaphore> sems;
  std::vector<uint64_t> vals;
  for (auto& kv : dd->heldTimeline) { sems.push_back(kv.first); vals.push_back(kv.second); }
  VkTimelineSemaphoreSubmitInfo tl{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  tl.signalSemaphoreValueCount = uint32_t(vals.size());
  tl.pSignalSemaphoreValues = vals.data();
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, sems.empty() ? nullptr : &tl};
  si.signalSemaphoreCount = uint32_t(sems.size());
  si.pSignalSemaphores = sems.data();
  size_t fi = 0;
  VkFence f0 = dd->heldFences.empty() ? VK_NULL_HANDLE : dd->heldFences[fi++];
  dd->d.QueueSubmit(q, 1, &si, f0);
  for (; fi < dd->heldFences.size(); ++fi) {
    VkSubmitInfo e{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    dd->d.QueueSubmit(q, 1, &e, dd->heldFences[fi]);
  }
  dd->heldTimeline.clear();
  dd->heldFences.clear();
}

// The app needs a held signal before the frame is complete: no replay for this frame.
// caller holds dd->recLock (not the queue lock)
void AbortReplayLocked(DeviceData* dd, const char* why) {
  if (!dd->recMono) {
    dd->recMono = true;
    ++dd->monoFrames;
    dd->lastAbortPresent = dd->presentCount.load();
    Log(2, "replay skipped this frame: %s", why);
  }
  dd->recCbs.clear();
  if (dd->gfxQueue) {
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, dd->gfxQueue)->m);
    ReleaseHeldLocked(dd, dd->gfxQueue);
  }
}

bool HeldLocked(DeviceData* dd, VkSemaphore sem, uint64_t value) {
  auto it = dd->heldTimeline.find(sem);
  return it != dd->heldTimeline.end() && value <= it->second;
}
bool WaitsOnHeldLocked(DeviceData* dd, uint32_t n, const VkSubmitInfo* s) {
  if (dd->heldTimeline.empty()) return false;
  for (uint32_t i = 0; i < n; ++i) {
    const VkTimelineSemaphoreSubmitInfo* tl = nullptr;
    for (auto* p = (const VkBaseInStructure*)s[i].pNext; p; p = p->pNext)
      if (p->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) tl = (const VkTimelineSemaphoreSubmitInfo*)p;
    if (!tl || !tl->pWaitSemaphoreValues) continue;
    for (uint32_t w = 0; w < s[i].waitSemaphoreCount && w < tl->waitSemaphoreValueCount; ++w)
      if (HeldLocked(dd, s[i].pWaitSemaphores[w], tl->pWaitSemaphoreValues[w])) return true;
  }
  return false;
}
bool WaitsOnHeldLocked(DeviceData* dd, uint32_t n, const VkSubmitInfo2* s) {
  if (dd->heldTimeline.empty()) return false;
  for (uint32_t i = 0; i < n; ++i)
    for (uint32_t w = 0; w < s[i].waitSemaphoreInfoCount; ++w)
      if (HeldLocked(dd, s[i].pWaitSemaphoreInfos[w].semaphore, s[i].pWaitSemaphoreInfos[w].value)) return true;
  return false;
}

bool RecordingQueue(DeviceData* dd, VkQueue q) {
  return dd->replay && q == dd->gfxQueue && dd->swapCount.load() > 0;
}

VKAPI_ATTR VkResult VKAPI_CALL hook_QueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo* s, VkFence f) {
  DeviceData* dd = Dev(q);
  for (uint32_t i = 0; i < n; ++i) dd->work += s[i].commandBufferCount;
  if (!dd->replay) {
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
    return dd->d.QueueSubmit(q, n, s, f);
  }
  std::unique_lock<std::mutex> rl(dd->recLock);
  if (WaitsOnHeldLocked(dd, n, s)) AbortReplayLocked(dd, "submit waits on a held signal");
  if (!RecordingQueue(dd, q) || dd->recMono) {
    rl.unlock();
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
    return dd->d.QueueSubmit(q, n, s, f);
  }
  std::vector<VkSubmitInfo> si(s, s + n);
  std::vector<VkTimelineSemaphoreSubmitInfo> tls(n);
  std::vector<std::vector<VkSemaphore>> sigs(n);
  std::vector<std::vector<uint64_t>> vals(n);
  for (uint32_t i = 0; i < n; ++i) {
    dd->recCbs.insert(dd->recCbs.end(), s[i].pCommandBuffers, s[i].pCommandBuffers + s[i].commandBufferCount);
    const VkTimelineSemaphoreSubmitInfo* tl = nullptr;
    for (auto* p = (const VkBaseInStructure*)s[i].pNext; p; p = p->pNext)
      if (p->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) tl = (const VkTimelineSemaphoreSubmitInfo*)p;
    bool changed = false;
    for (uint32_t k = 0; k < s[i].signalSemaphoreCount; ++k) {
      VkSemaphore sem = s[i].pSignalSemaphores[k];
      uint64_t v = (tl && tl->pSignalSemaphoreValues && k < tl->signalSemaphoreValueCount)
                       ? tl->pSignalSemaphoreValues[k] : 0;
      if (IsTimeline(dd, sem)) {
        uint64_t& h = dd->heldTimeline[sem];
        h = std::max(h, v);
        changed = true;
      } else {
        sigs[i].push_back(sem);
        vals[i].push_back(v);
      }
    }
    if (changed) {
      si[i].signalSemaphoreCount = uint32_t(sigs[i].size());
      si[i].pSignalSemaphores = sigs[i].data();
      if (tl) {
        tls[i] = *tl;
        tls[i].pNext = nullptr;
        tls[i].signalSemaphoreValueCount = uint32_t(vals[i].size());
        tls[i].pSignalSemaphoreValues = vals[i].data();
        si[i].pNext = &tls[i];
      }
    }
  }
  if (f) dd->heldFences.push_back(f);
  std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
  return dd->d.QueueSubmit(q, n, si.data(), VK_NULL_HANDLE);
}

VkResult Submit2Common(DeviceData* dd, VkQueue q, uint32_t n, const VkSubmitInfo2* s, VkFence f,
                       PFN_vkQueueSubmit2 fn) {
  for (uint32_t i = 0; i < n; ++i) dd->work += s[i].commandBufferInfoCount;
  if (!dd->replay) {
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
    return fn(q, n, s, f);
  }
  std::unique_lock<std::mutex> rl(dd->recLock);
  if (WaitsOnHeldLocked(dd, n, s)) AbortReplayLocked(dd, "submit waits on a held signal");
  if (!RecordingQueue(dd, q) || dd->recMono) {
    rl.unlock();
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
    return fn(q, n, s, f);
  }
  std::vector<VkSubmitInfo2> si(s, s + n);
  std::vector<std::vector<VkSemaphoreSubmitInfo>> sigs(n);
  for (uint32_t i = 0; i < n; ++i) {
    for (uint32_t c = 0; c < s[i].commandBufferInfoCount; ++c)
      dd->recCbs.push_back(s[i].pCommandBufferInfos[c].commandBuffer);
    bool changed = false;
    for (uint32_t k = 0; k < s[i].signalSemaphoreInfoCount; ++k) {
      const VkSemaphoreSubmitInfo& ss = s[i].pSignalSemaphoreInfos[k];
      if (IsTimeline(dd, ss.semaphore)) {
        uint64_t& h = dd->heldTimeline[ss.semaphore];
        h = std::max(h, ss.value);
        changed = true;
      } else {
        sigs[i].push_back(ss);
      }
    }
    if (changed) {
      si[i].signalSemaphoreInfoCount = uint32_t(sigs[i].size());
      si[i].pSignalSemaphoreInfos = sigs[i].data();
    }
  }
  if (f) dd->heldFences.push_back(f);
  std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
  return fn(q, n, si.data(), VK_NULL_HANDLE);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_QueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2* s, VkFence f) {
  DeviceData* dd = Dev(q);
  return Submit2Common(dd, q, n, s, f, dd->d.QueueSubmit2);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_QueueSubmit2KHR(VkQueue q, uint32_t n, const VkSubmitInfo2* s, VkFence f) {
  DeviceData* dd = Dev(q);
  return Submit2Common(dd, q, n, s, f, dd->d.QueueSubmit2KHR);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_QueueWaitIdle(VkQueue q) {
  DeviceData* dd = Dev(q);
  if (dd->replay && q == dd->gfxQueue) {
    std::lock_guard<std::mutex> rl(dd->recLock);
    if (!dd->heldFences.empty() || !dd->heldTimeline.empty() || !dd->recCbs.empty())
      AbortReplayLocked(dd, "vkQueueWaitIdle");
  }
  std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
  return dd->d.QueueWaitIdle(q);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_QueueBindSparse(VkQueue q, uint32_t n, const VkBindSparseInfo* b, VkFence f) {
  DeviceData* dd = Dev(q);
  std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
  return dd->d.QueueBindSparse(q, n, b, f);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_DeviceWaitIdle(VkDevice device) {
  DeviceData* dd = Dev(device);
  if (!dd->gfxQueue) return dd->d.DeviceWaitIdle(device);
  if (dd->replay) {
    std::lock_guard<std::mutex> rl(dd->recLock);
    if (!dd->heldFences.empty() || !dd->heldTimeline.empty() || !dd->recCbs.empty())
      AbortReplayLocked(dd, "vkDeviceWaitIdle");
  }
  // every queue must be idle from the host side: also lock our presenter queue
  std::unique_lock<std::mutex> lp;
  if (dd->presQueue && dd->presQueue != dd->gfxQueue) lp = std::unique_lock<std::mutex>(GetQueueInfo(dd, dd->presQueue)->m);
  std::lock_guard<std::mutex> l(GetQueueInfo(dd, dd->gfxQueue)->m);
  return dd->d.DeviceWaitIdle(device);
}

// CPU waits: if the app blocks on a held signal and no present arrives, it is waiting for
// something mid-frame (e.g. a GPU readback) -> release and skip the replay for this frame.
template <class WaitFn, class HeldFn>
VkResult WaitWithReplay(DeviceData* dd, uint64_t timeout, WaitFn waitFn, HeldFn heldFn) {
  if (!dd->replay || timeout == 0) return waitFn(timeout);
  const auto begin = Clock::now();
  auto stallStart = begin;
  uint64_t seenPresent = dd->presentCount.load();
  for (;;) {
    auto now = Clock::now();
    uint64_t total = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now - begin).count());
    uint64_t slice = 2000000ull;  // 2 ms
    if (timeout != UINT64_MAX) {
      if (total >= timeout) return VK_TIMEOUT;
      slice = std::min(slice, timeout - total);
    }
    VkResult r = waitFn(slice);
    if (r != VK_TIMEOUT) return r;
    uint64_t pc = dd->presentCount.load();
    if (pc != seenPresent) {  // frames are still being presented: a normal wait
      seenPresent = pc;
      stallStart = Clock::now();
      continue;
    }
    uint64_t stalled = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - stallStart).count());
    bool recentAbort = pc - dd->lastAbortPresent.load() < 120;
    if (stalled > (recentAbort ? 3000000ull : 40000000ull)) {
      std::lock_guard<std::mutex> rl(dd->recLock);
      if (heldFn()) AbortReplayLocked(dd, "CPU waits for a held signal mid-frame");
      stallStart = Clock::now();
    }
  }
}

VKAPI_ATTR VkResult VKAPI_CALL hook_WaitForFences(VkDevice device, uint32_t n, const VkFence* fences, VkBool32 all,
                                                  uint64_t timeout) {
  DeviceData* dd = Dev(device);
  return WaitWithReplay(
      dd, timeout, [&](uint64_t t) { return dd->d.WaitForFences(device, n, fences, all, t); },
      [&]() {
        for (uint32_t i = 0; i < n; ++i)
          if (std::find(dd->heldFences.begin(), dd->heldFences.end(), fences[i]) != dd->heldFences.end())
            return true;
        return false;
      });
}
VkResult WaitSemCommon(DeviceData* dd, VkDevice device, const VkSemaphoreWaitInfo* info, uint64_t timeout,
                       PFN_vkWaitSemaphores fn) {
  return WaitWithReplay(
      dd, timeout, [&](uint64_t t) { return fn(device, info, t); },
      [&]() {
        for (uint32_t i = 0; i < info->semaphoreCount; ++i)
          if (HeldLocked(dd, info->pSemaphores[i], info->pValues[i])) return true;
        return false;
      });
}
VKAPI_ATTR VkResult VKAPI_CALL hook_WaitSemaphores(VkDevice device, const VkSemaphoreWaitInfo* info, uint64_t t) {
  DeviceData* dd = Dev(device);
  return WaitSemCommon(dd, device, info, t, dd->d.WaitSemaphores);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_WaitSemaphoresKHR(VkDevice device, const VkSemaphoreWaitInfo* info, uint64_t t) {
  DeviceData* dd = Dev(device);
  return WaitSemCommon(dd, device, info, t, dd->d.WaitSemaphoresKHR);
}

VKAPI_ATTR VkResult VKAPI_CALL hook_CreateSemaphore(VkDevice device, const VkSemaphoreCreateInfo* ci,
                                                    const VkAllocationCallbacks* a, VkSemaphore* sem) {
  DeviceData* dd = Dev(device);
  VkResult r = dd->d.CreateSemaphore(device, ci, a, sem);
  if (r == VK_SUCCESS)
    for (auto* p = (const VkBaseInStructure*)ci->pNext; p; p = p->pNext)
      if (p->sType == VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO &&
          ((const VkSemaphoreTypeCreateInfo*)p)->semaphoreType == VK_SEMAPHORE_TYPE_TIMELINE) {
        std::lock_guard<std::mutex> l(dd->semLock);
        dd->timelineSems.insert(*sem);
      }
  return r;
}
VKAPI_ATTR void VKAPI_CALL hook_DestroySemaphore(VkDevice device, VkSemaphore sem, const VkAllocationCallbacks* a) {
  DeviceData* dd = Dev(device);
  {
    std::lock_guard<std::mutex> l(dd->semLock);
    dd->timelineSems.erase(sem);
  }
  dd->d.DestroySemaphore(device, sem, a);
}

VKAPI_ATTR VkResult VKAPI_CALL hook_BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo* bi) {
  DeviceData* dd = Dev(cb);
  if (dd->depthMode) {  // a new recording: what params.flags holds when it runs is not known
    CbState(cb).st->flag = -1;
    CbState(cb).st->pending = false;
    if (bi->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT) CbState(cb).st->draws = 0;  // (secondary)
    return dd->d.BeginCommandBuffer(cb, bi);
  }
  if (!dd->replay && !dd->warmupPending) return dd->d.BeginCommandBuffer(cb, bi);
  VkCommandBufferBeginInfo b = *bi;  // the frame is submitted twice (also during the warm-up:
                                     // command buffers recorded then may be submitted in replay)
  b.flags &= ~VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  b.flags |= VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  return dd->d.BeginCommandBuffer(cb, &b);
}

// ------------------------------------------------------------------------------------
// Shader dump (dump_shaders = 1 / NV3D_DUMP_SHADERS=1): every SPIR-V module the game creates is saved
// once as ~/.cache/3dvision/shaders/<game>/<hash>.<stage>.spv, and every pipeline as a line of
// pipelines.txt (which shaders it joins, its vertex inputs and attachments). Offline material for
// tools/3dvision-shader-scan (finding the screen-space passes that need a stereo correction).
// ------------------------------------------------------------------------------------
std::mutex g_dumpLock;
std::unordered_set<uint64_t> g_dumped;
std::unordered_set<std::string> g_dumpedPipes;
std::string g_shaderDumpDir;

const std::string& ShaderDumpDir() {
  if (g_shaderDumpDir.empty()) {
    std::string base;
    if (const char* x = getenv("XDG_CACHE_HOME"); x && *x) base = std::string(x) + "/3dvision";
    else if (const char* h = getenv("HOME"); h && *h) base = std::string(h) + "/.cache/3dvision";
    else base = "/tmp/3dvision";
    std::string game = GameName().empty() ? "game" : GameName();
    for (char& c : game)
      if (c == '/' || c == ' ') c = '_';
    std::string path;
    for (const std::string& part : {base, base + "/shaders", base + "/shaders/" + game}) {
      mkdir(part.c_str(), 0755);
      path = part;
    }
    g_shaderDumpDir = path;
    Log(1, "dump_shaders: saving shaders to %s", g_shaderDumpDir.c_str());
  }
  return g_shaderDumpDir;
}

const char* SpirvStageName(const uint32_t* code, size_t words) {
  if (words < 5 || code[0] != 0x07230203) return "bad";
  for (size_t i = 5; i < words;) {
    uint32_t op = code[i] & 0xFFFF, n = code[i] >> 16;
    if (n == 0) break;
    if (op == 15 && i + 1 < words) {  // OpEntryPoint: the first one names the stage
      static const char* kNames[] = {"vs", "tcs", "tes", "gs", "fs", "cs"};
      uint32_t model = code[i + 1];
      if (model < 6) return kNames[model];
      if (model == 5267) return "task";
      if (model == 5268) return "mesh";
      return "other";
    }
    i += n;
  }
  return "none";
}

uint64_t DumpShader(const uint32_t* code, size_t words) {
  uint64_t h = 1469598103934665603ull;  // FNV-1a over the words
  for (size_t i = 0; i < words; ++i) {
    h ^= code[i];
    h *= 1099511628211ull;
  }
  std::lock_guard<std::mutex> l(g_dumpLock);
  if (!g_dumped.insert(h).second) return h;
  char name[64];
  snprintf(name, sizeof(name), "/%016llx.%s.spv", (unsigned long long)h, SpirvStageName(code, words));
  std::string path = ShaderDumpDir() + name;
  if (FILE* f = fopen(path.c_str(), "wb")) {
    fwrite(code, 4, words, f);
    fclose(f);
  }
  return h;
}

void DumpPipelineLine(const std::string& line) {
  std::lock_guard<std::mutex> l(g_dumpLock);
  if (!g_dumpedPipes.insert(line).second) return;
  std::string path = ShaderDumpDir() + "/pipelines.txt";
  if (FILE* f = fopen(path.c_str(), "a")) {
    fprintf(f, "%s\n", line.c_str());
    fclose(f);
  }
}

uint64_t StageHash(DeviceData* dd, const VkPipelineShaderStageCreateInfo& st) {
  if (st.module != VK_NULL_HANDLE) {
    std::lock_guard<std::mutex> l(dd->shaderLock);
    auto it = dd->moduleHash.find(st.module);
    return it == dd->moduleHash.end() ? 0 : it->second;
  }
  for (auto* p = (const VkBaseInStructure*)st.pNext; p; p = p->pNext)
    if (p->sType == VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO) {
      auto* smci = (const VkShaderModuleCreateInfo*)p;
      return DumpShader(smci->pCode, smci->codeSize / 4);
    }
  return 0;
}

void DumpGraphicsPipeline(DeviceData* dd, const VkGraphicsPipelineCreateInfo& ci) {
  char buf[128];
  std::string line = "G";
  for (uint32_t s = 0; s < ci.stageCount; ++s) {
    const VkPipelineShaderStageCreateInfo& st = ci.pStages[s];
    const char* n = st.stage == VK_SHADER_STAGE_VERTEX_BIT ? "vs" : st.stage == VK_SHADER_STAGE_FRAGMENT_BIT ? "fs"
                  : st.stage == VK_SHADER_STAGE_GEOMETRY_BIT ? "gs" : st.stage == VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT
                  ? "tes" : st.stage == VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT ? "tcs" : "other";
    snprintf(buf, sizeof(buf), " %s=%016llx", n, (unsigned long long)StageHash(dd, st));
    line += buf;
  }
  // a full-screen pass has no vertex buffers (UE: lighting, shadow projection, post process)
  uint32_t attribs = ci.pVertexInputState ? ci.pVertexInputState->vertexAttributeDescriptionCount : 0;
  int depthTest = ci.pDepthStencilState ? int(ci.pDepthStencilState->depthTestEnable) : -1;
  int depthWrite = ci.pDepthStencilState ? int(ci.pDepthStencilState->depthWriteEnable) : -1;
  int colors = -1, depthFmt = -1;
  if (ci.renderPass == VK_NULL_HANDLE) {
    for (auto* p = (const VkBaseInStructure*)ci.pNext; p; p = p->pNext)
      if (p->sType == VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO) {
        auto* r = (const VkPipelineRenderingCreateInfo*)p;
        colors = int(r->colorAttachmentCount);
        depthFmt = int(r->depthAttachmentFormat);
      }
  }
  int blend = 0;
  if (ci.pColorBlendState)
    for (uint32_t k = 0; k < ci.pColorBlendState->attachmentCount; ++k)
      blend |= ci.pColorBlendState->pAttachments[k].blendEnable ? 1 : 0;
  snprintf(buf, sizeof(buf), " attribs=%u colors=%d depthfmt=%d ztest=%d zwrite=%d blend=%d", attribs, colors, depthFmt,
           depthTest, depthWrite, blend);
  line += buf;
  DumpPipelineLine(line);
}

bool PatchCode(DeviceData* dd, const uint32_t* code, size_t words, uint32_t model, const char* entry,
               std::vector<uint32_t>& out);
VkShaderModule GetPatchedModule(DeviceData* dd, VkShaderModule m, uint32_t model, const char* entry);

VKAPI_ATTR VkResult VKAPI_CALL hook_CreateComputePipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
                                                           const VkComputePipelineCreateInfo* infos,
                                                           const VkAllocationCallbacks* a, VkPipeline* out) {
  DeviceData* dd = Dev(device);
  if (Cfg().dumpShaders)
    for (uint32_t i = 0; i < count; ++i) {
      char buf[64];
      snprintf(buf, sizeof(buf), "C cs=%016llx", (unsigned long long)StageHash(dd, infos[i].stage));
      DumpPipelineLine(buf);
    }
  if (!dd->screenFix || !dd->stereoShaders) return dd->d.CreateComputePipelines(device, cache, count, infos, a, out);
  // screen-space position rebuilds done in compute (tiled lighting, SSAO...)
  std::vector<VkComputePipelineCreateInfo> ci(infos, infos + count);
  struct Restore { VkShaderModuleCreateInfo* smci; const uint32_t* code; size_t size; };
  std::vector<Restore> restores;
  std::vector<std::unique_ptr<std::vector<uint32_t>>> inlineCode;
  for (uint32_t i = 0; i < count; ++i) {
    VkPipelineShaderStageCreateInfo& st = ci[i].stage;
    if (st.stage != VK_SHADER_STAGE_COMPUTE_BIT) continue;
    if (st.module != VK_NULL_HANDLE) {
      if (VkShaderModule pm = GetPatchedModule(dd, st.module, 5, st.pName)) st.module = pm;
      continue;
    }
    for (auto* p = (VkBaseOutStructure*)st.pNext; p; p = p->pNext) {
      if (p->sType != VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO) continue;
      auto* smci = (VkShaderModuleCreateInfo*)p;
      auto buf = std::make_unique<std::vector<uint32_t>>();
      if (PatchCode(dd, smci->pCode, smci->codeSize / 4, 5, st.pName, *buf)) {
        restores.push_back({smci, smci->pCode, smci->codeSize});
        smci->pCode = buf->data();
        smci->codeSize = buf->size() * 4;
        inlineCode.push_back(std::move(buf));
      }
      break;
    }
  }
  VkResult r = dd->d.CreateComputePipelines(device, cache, count, ci.data(), a, out);
  for (auto& rs : restores) {
    rs.smci->pCode = rs.code;
    rs.smci->codeSize = rs.size;
  }
  return r;
}

// ------------------------------------------------------------------------------------
// Shader patching
// ------------------------------------------------------------------------------------
VKAPI_ATTR VkResult VKAPI_CALL hook_CreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* ci,
                                                       const VkAllocationCallbacks* a, VkShaderModule* m) {
  DeviceData* dd = Dev(device);
  VkResult r = dd->d.CreateShaderModule(device, ci, a, m);
  if (r == VK_SUCCESS && Cfg().dumpShaders && ci->codeSize % 4 == 0) {
    uint64_t h = DumpShader(ci->pCode, ci->codeSize / 4);
    std::lock_guard<std::mutex> l(dd->shaderLock);
    dd->moduleHash[*m] = h;
  }
  if (r == VK_SUCCESS && dd->stereoShaders && ci->codeSize % 4 == 0 &&
      (dd->screenFix || HasPreRasterEntryPoint(ci->pCode, ci->codeSize / 4))) {
    std::lock_guard<std::mutex> l(dd->shaderLock);
    dd->modules[*m].code.assign(ci->pCode, ci->pCode + ci->codeSize / 4);
  }
  return r;
}

VKAPI_ATTR void VKAPI_CALL hook_DestroyShaderModule(VkDevice device, VkShaderModule m, const VkAllocationCallbacks* a) {
  DeviceData* dd = Dev(device);
  {
    std::lock_guard<std::mutex> l(dd->shaderLock);
    auto it = dd->modules.find(m);
    if (it != dd->modules.end()) {
      for (auto& v : it->second.variants)
        if (v.patched) dd->d.DestroyShaderModule(device, v.patched, nullptr);
      dd->modules.erase(it);
    }
    dd->moduleHash.erase(m);
  }
  dd->d.DestroyShaderModule(device, m, a);
}

uint32_t StageToModel(VkShaderStageFlagBits s) {
  switch (s) {
    case VK_SHADER_STAGE_VERTEX_BIT: return 0;
    case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: return 2;
    case VK_SHADER_STAGE_GEOMETRY_BIT: return 3;
    default: return UINT32_MAX;
  }
}

bool PatchCode(DeviceData* dd, const uint32_t* code, size_t words, uint32_t model, const char* entry,
               std::vector<uint32_t>& out) {
  std::string log;
  if (model == 4 || model == 5) {  // fragment / compute: screen-space position rebuilds
    PatchResult pr = PatchScreenSpace(code, words, dd->paramAddr, out, &log);
    if (pr != PatchResult::Patched) return false;
    uint32_t n = ++dd->screenFixCount;
    Log(1, "fix_shadows: %s shader #%u corrected (%s)", model == 4 ? "fragment" : "compute", n, log.c_str());
    return true;
  }
  PatchResult pr = PatchStereo(code, words, entry, model, dd->paramAddr, out, &log, dd->canCalib && dd->depthMode);
  if (pr == PatchResult::Patched) {
    uint32_t n = ++dd->patchedCount;
    Log(2, "patched shader #%u (model %u, %s)", n, model, log.c_str());
    return true;
  }
  ++dd->skippedCount;
  Log(2, "shader not patched: %s (model %u)", PatchResultName(pr), model);
  return false;
}

VkShaderModule GetPatchedModule(DeviceData* dd, VkShaderModule m, uint32_t model, const char* entry) {
  std::lock_guard<std::mutex> l(dd->shaderLock);
  auto it = dd->modules.find(m);
  if (it == dd->modules.end()) return VK_NULL_HANDLE;
  for (auto& v : it->second.variants)
    if (v.model == model && v.entry == entry) return v.patched;
  std::vector<uint32_t> out;
  VkShaderModule pm = VK_NULL_HANDLE;
  if (PatchCode(dd, it->second.code.data(), it->second.code.size(), model, entry, out)) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = out.size() * 4;
    ci.pCode = out.data();
    if (dd->d.CreateShaderModule(dd->device, &ci, nullptr, &pm) != VK_SUCCESS) pm = VK_NULL_HANDLE;
  }
  it->second.variants.push_back({model, entry, pm});
  return pm;
}

VKAPI_ATTR VkResult VKAPI_CALL hook_CreateGraphicsPipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
                                                            const VkGraphicsPipelineCreateInfo* infos,
                                                            const VkAllocationCallbacks* a, VkPipeline* out) {
  DeviceData* dd = Dev(device);
  if (Cfg().dumpShaders)
    for (uint32_t i = 0; i < count; ++i) DumpGraphicsPipeline(dd, infos[i]);
  if (!dd->stereoShaders) return dd->d.CreateGraphicsPipelines(device, cache, count, infos, a, out);

  std::vector<VkGraphicsPipelineCreateInfo> ci(infos, infos + count);
  std::vector<std::vector<VkPipelineShaderStageCreateInfo>> stages(count);
  struct Restore { VkShaderModuleCreateInfo* smci; const uint32_t* code; size_t size; };
  std::vector<Restore> restores;
  std::vector<std::unique_ptr<std::vector<uint32_t>>> inlineCode;

  // patches stage s of pipeline i (shader module or VK_KHR_maintenance5 inline code)
  auto patchStage = [&](uint32_t i, uint32_t s, uint32_t model) {
    const VkPipelineShaderStageCreateInfo& st = ci[i].pStages[s];
    if (st.module != VK_NULL_HANDLE) {
      VkShaderModule pm = GetPatchedModule(dd, st.module, model, st.pName);
      if (pm) {
        if (stages[i].empty()) {
          stages[i].assign(ci[i].pStages, ci[i].pStages + ci[i].stageCount);
          ci[i].pStages = stages[i].data();
        }
        stages[i][s].module = pm;
      }
      return;
    }
    for (auto* p = (VkBaseOutStructure*)st.pNext; p; p = p->pNext) {
      if (p->sType != VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO) continue;
      auto* smci = (VkShaderModuleCreateInfo*)p;
      auto buf = std::make_unique<std::vector<uint32_t>>();
      if (PatchCode(dd, smci->pCode, smci->codeSize / 4, model, st.pName, *buf)) {
        restores.push_back({smci, smci->pCode, smci->codeSize});
        smci->pCode = buf->data();
        smci->codeSize = buf->size() * 4;
        inlineCode.push_back(std::move(buf));
      }
      break;
    }
  };

  for (uint32_t i = 0; i < count; ++i) {
    if (!ci[i].pStages || !ci[i].stageCount) continue;
    int best = -1, bestRank = -1, frag = -1;
    for (uint32_t s = 0; s < ci[i].stageCount; ++s) {
      int rank = -1;
      switch (ci[i].pStages[s].stage) {
        case VK_SHADER_STAGE_VERTEX_BIT: rank = 0; break;
        case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: rank = 1; break;
        case VK_SHADER_STAGE_GEOMETRY_BIT: rank = 2; break;
        case VK_SHADER_STAGE_FRAGMENT_BIT: frag = int(s); break;
        default: break;
      }
      if (rank > bestRank) { bestRank = rank; best = int(s); }
    }
    if (best < 0) continue;
    // screen-space position rebuilds (Unreal shadow projection, lights...): see PatchScreenSpace
    if (frag >= 0 && dd->screenFix) patchStage(i, uint32_t(frag), 4);

    bool noColor = false;
    int depthFmt = -1;
    if (ci[i].renderPass == VK_NULL_HANDLE) {  // dynamic rendering: the attachment formats are in pNext
      noColor = true;
      for (auto* p = (const VkBaseInStructure*)ci[i].pNext; p; p = p->pNext)
        if (p->sType == VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO) {
          noColor = ((const VkPipelineRenderingCreateInfo*)p)->colorAttachmentCount == 0;
          depthFmt = int(((const VkPipelineRenderingCreateInfo*)p)->depthAttachmentFormat);
        }
    }
    if (!Cfg().shiftShadows) {
      // depth-only pipelines (shadow maps...): no stereo shift. Shadow maps are seen from the light,
      // they must be the same for both eyes. (A depth pre-pass would also be skipped: per-game option.)
      if (frag < 0 || noColor) {
        ++dd->depthOnlyCount;
        continue;
      }
    } else if (dd->unshiftD16 && noColor && depthFmt == VK_FORMAT_D16_UNORM) {
      // Unreal Engine draws its shadow maps in 16-bit depth and the scene's depth pre-pass in
      // D32S8 / D24S8: the shadow maps stay unshifted, the pre-pass keeps the stereo shift
      ++dd->depthOnlyCount;
      continue;
    }
    ++dd->pipelineCount;
    patchStage(i, uint32_t(best), StageToModel(ci[i].pStages[best].stage));
  }
  VkResult r = dd->d.CreateGraphicsPipelines(device, cache, count, ci.data(), a, out);
  for (auto& rs : restores) {
    rs.smci->pCode = rs.code;
    rs.smci->codeSize = rs.size;
  }
  return r;
}

// ------------------------------------------------------------------------------------
// Virtual swapchain + frame-sequential presenter
// ------------------------------------------------------------------------------------
void ComputeRects(Swapchain* sc);
Swapchain* FindSwapchain(DeviceData* dd, VkSwapchainKHR h);

inline VkQueue PresQueue(DeviceData* dd) { return dd->presQueue ? dd->presQueue : dd->gfxQueue; }

bool CreateImageMem(DeviceData* dd, VkImageCreateInfo ici, VkImage* img, VkDeviceMemory* mem) {
  VkResult r = dd->d.CreateImage(dd->device, &ici, nullptr, img);
  if (r != VK_SUCCESS && (ici.usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
    ici.usage &= ~VK_IMAGE_USAGE_STORAGE_BIT;
    r = dd->d.CreateImage(dd->device, &ici, nullptr, img);
  }
  if (r != VK_SUCCESS) return false;
  VkMemoryRequirements req;
  dd->d.GetImageMemoryRequirements(dd->device, *img, &req);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = FindMemType(dd, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (dd->d.AllocateMemory(dd->device, &ai, nullptr, mem) != VK_SUCCESS) return false;
  return dd->d.BindImageMemory(dd->device, *img, *mem, 0) == VK_SUCCESS;
}

void Barrier(DeviceData* dd, VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
             VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage,
             VkAccessFlags dstAccess) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.srcAccessMask = srcAccess;
  b.dstAccessMask = dstAccess;
  b.oldLayout = from;
  b.newLayout = to;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  dd->d.CmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// copy src rect into dst rectangle at (dx,dy) with size dext (dst in `dstLayout`); scale if needed
void CopyOrBlit(DeviceData* dd, Swapchain* sc, VkCommandBuffer cb, VkImage src, VkOffset2D so, VkExtent2D sext,
                VkImage dst, VkImageLayout dstLayout, int32_t dx, int32_t dy, VkExtent2D dext) {
  if (((sext.width == dext.width && sext.height == dext.height) && !sc->formatDiffers) || !sc->canBlit) {
    VkImageCopy c{};
    c.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.srcOffset = {so.x, so.y, 0};
    c.dstOffset = {dx, dy, 0};
    c.extent = {std::min(sext.width, dext.width), std::min(sext.height, dext.height), 1};
    dd->d.CmdCopyImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, dstLayout, 1, &c);
  } else {
    VkImageBlit b{};
    b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.srcOffsets[0] = {so.x, so.y, 0};
    b.srcOffsets[1] = {so.x + int32_t(sext.width), so.y + int32_t(sext.height), 1};
    b.dstOffsets[0] = {dx, dy, 0};
    b.dstOffsets[1] = {dx + int32_t(dext.width), dy + int32_t(dext.height), 1};
    dd->d.CmdBlitImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, dstLayout, 1, &b, VK_FILTER_LINEAR);
  }
}

VkResult Worse(VkResult a, VkResult b) {
  if (a < 0) return a;
  if (b < 0) return b;
  if (a == VK_SUBOPTIMAL_KHR || b == VK_SUBOPTIMAL_KHR) return VK_SUBOPTIMAL_KHR;
  return VK_SUCCESS;
}

// ---- frame capture (diagnostics) ----------------------------------------------------
struct HostBuf {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
};
bool CreateHostBuf(DeviceData* dd, VkDeviceSize size, HostBuf& hb) {
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (dd->d.CreateBuffer(dd->device, &bci, nullptr, &hb.buf) != VK_SUCCESS) return false;
  VkMemoryRequirements req;
  dd->d.GetBufferMemoryRequirements(dd->device, hb.buf, &req);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = FindMemType(dd, req.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                   VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
  if (dd->d.AllocateMemory(dd->device, &ai, nullptr, &hb.mem) != VK_SUCCESS) return false;
  dd->d.BindBufferMemory(dd->device, hb.buf, hb.mem, 0);
  hb.size = size;
  return true;
}
void FreeHostBuf(DeviceData* dd, HostBuf& hb) {
  if (hb.buf) dd->d.DestroyBuffer(dd->device, hb.buf, nullptr);
  if (hb.mem) dd->d.FreeMemory(dd->device, hb.mem, nullptr);
  hb = HostBuf{};
}
uint32_t BytesPerPixel(VkFormat f) {
  switch (f) {
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
      return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM:
      return 8;
    default:
      return 0;
  }
}
bool Is32bpp(VkFormat f) { return BytesPerPixel(f) != 0; }
void CopyToBuf(DeviceData* dd, VkCommandBuffer cb, VkImage img, VkImageLayout layout, VkExtent2D e, HostBuf& hb) {
  VkBufferImageCopy r{};
  r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  r.imageExtent = {e.width, e.height, 1};
  dd->d.CmdCopyImageToBuffer(cb, img, layout, hb.buf, 1, &r);
}
inline unsigned char HalfTo8(uint16_t h) {
  uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF;
  if (h & 0x8000) return 0;
  float f = e == 0 ? m / 16777216.0f : e == 31 ? 1.0f : (1.0f + m / 1024.0f) * float(1u << e) / 32768.0f;
  if (f >= 1.0f) return 255;
  return (unsigned char)(f * 255.0f + 0.5f);
}
inline void Decode(const unsigned char* px, VkFormat fmt, unsigned char& r, unsigned char& g, unsigned char& b) {
  uint32_t v;
  memcpy(&v, px, 4);
  switch (fmt) {
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
      b = v & 0xFF; g = (v >> 8) & 0xFF; r = (v >> 16) & 0xFF; break;
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
      b = (v & 0x3FF) >> 2; g = ((v >> 10) & 0x3FF) >> 2; r = ((v >> 20) & 0x3FF) >> 2; break;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
      r = (v & 0x3FF) >> 2; g = ((v >> 10) & 0x3FF) >> 2; b = ((v >> 20) & 0x3FF) >> 2; break;
    case VK_FORMAT_R16G16B16A16_SFLOAT: {
      uint16_t c[3]; memcpy(c, px, 6);
      r = HalfTo8(c[0]); g = HalfTo8(c[1]); b = HalfTo8(c[2]); break;
    }
    case VK_FORMAT_R16G16B16A16_UNORM: {
      uint16_t c[3]; memcpy(c, px, 6);
      r = c[0] >> 8; g = c[1] >> 8; b = c[2] >> 8; break;
    }
    default:
      r = v & 0xFF; g = (v >> 8) & 0xFF; b = (v >> 16) & 0xFF; break;
  }
}

// Maps the current content rectangle + zoom to the source/destination rectangles of one
// eye (the whole display). zoom = 1 shows the whole content (aspect kept, may leave bars);
// larger values zoom in around the centre until the eye is filled.
void ComputeRects(Swapchain* sc) {
  const double ew = sc->rext.width, eh = sc->rext.height, A = ew / eh;
  double zoom = std::clamp(double(Cfg().zoom), 0.5, 4.0);
  sc->lastZoom = Cfg().zoom;
  double cx0 = sc->cx0, cy0 = sc->cy0, cx1 = sc->cx1, cy1 = sc->cy1;
  double cw = cx1 - cx0, ch = cy1 - cy0;
  if (cw < 16 || ch < 16) return;
  // eye-aspect window that contains the content, shrunk by zoom
  double ww = std::max(cw, ch * A) / zoom, wh = ww / A;
  double mx = (cx0 + cx1) / 2, my = (cy0 + cy1) / 2;
  double wx0 = mx - ww / 2, wy0 = my - wh / 2;
  double sx0 = std::max(wx0, cx0), sy0 = std::max(wy0, cy0);
  double sx1 = std::min(wx0 + ww, cx1), sy1 = std::min(wy0 + wh, cy1);
  double scx = ew / ww, scy = eh / wh;
  int ix0 = int(sx0 + 0.5), iy0 = int(sy0 + 0.5), ix1 = int(sx1 + 0.5), iy1 = int(sy1 + 0.5);
  int dx0 = int((sx0 - wx0) * scx + 0.5), dy0 = int((sy0 - wy0) * scy + 0.5);
  int dx1 = int((sx1 - wx0) * scx + 0.5), dy1 = int((sy1 - wy0) * scy + 0.5);
  if (dx0 <= 3) dx0 = 0;
  if (dy0 <= 3) dy0 = 0;
  if (dx1 >= int(ew) - 3) dx1 = int(ew);
  if (dy1 >= int(eh) - 3) dy1 = int(eh);
  if (ix1 <= ix0 || iy1 <= iy0 || dx1 <= dx0 || dy1 <= dy0) return;
  sc->srcOff = {ix0, iy0};
  sc->srcExt = {uint32_t(ix1 - ix0), uint32_t(iy1 - iy0)};
  sc->dstOff = {dx0, dy0};
  sc->dstExt = {uint32_t(dx1 - dx0), uint32_t(dy1 - dy0)};
}

// Finds the non-black region of the game image. A new region is adopted only after it
// has been seen twice in a row (so splash screens / fades don't stick).
void DetectBars(DeviceData* dd, Swapchain* sc, HostBuf& hb) {
  void* p = nullptr;
  if (dd->d.MapMemory(dd->device, hb.mem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS) return;
  const unsigned char* px = (const unsigned char*)p;
  const uint32_t bpp = BytesPerPixel(sc->format);
  const int W = int(sc->vext.width), H = int(sc->vext.height);
  int x0 = W, x1 = -1, y0 = H, y1 = -1;
  for (int y = 0; y < H; y += 2) {
    const unsigned char* row = px + size_t(y) * W * bpp;
    for (int x = 0; x < W; ++x) {
      unsigned char r, g, b;
      Decode(row + size_t(x) * bpp, sc->format, r, g, b);
      if (r > 12 || g > 12 || b > 12) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  dd->d.UnmapMemory(dd->device, hb.mem);
  ++sc->detections;
  if (x1 < 0 || (x1 - x0) < W / 4 || (y1 - y0) < H / 4) return;  // black / loading screen
  // end-exclusive, snap near-edges
  int nx0 = x0 <= 4 ? 0 : x0, nx1 = x1 >= W - 5 ? W : x1 + 1;
  int ny0 = y0 <= 4 ? 0 : y0, ny1 = y1 >= H - 6 ? H : y1 + 2;
  auto near = [](int a, int b) { return std::abs(a - b) <= 8; };
  if (near(nx0, sc->kx0) && near(nx1, sc->kx1) && near(ny0, sc->ky0) && near(ny1, sc->ky1)) {
    ++sc->candCount;
  } else {
    sc->kx0 = nx0; sc->kx1 = nx1; sc->ky0 = ny0; sc->ky1 = ny1;
    sc->candCount = 1;
  }
  bool same = near(nx0, sc->cx0) && near(nx1, sc->cx1) && near(ny0, sc->cy0) && near(ny1, sc->cy1);
  if (sc->candCount >= 2 && !same) {
    sc->cx0 = nx0; sc->cx1 = nx1; sc->cy0 = ny0; sc->cy1 = ny1;
    ComputeRects(sc);
    Log(1, "game content %dx%d at (%d,%d) of %dx%d -> shown %ux%u in each eye (zoom %.2f, Ctrl+Shift+PgUp/PgDn)",
        nx1 - nx0, ny1 - ny0, nx0, ny0, W, H, sc->dstExt.width, sc->dstExt.height, Cfg().zoom);
  }
}

bool WritePPM(DeviceData* dd, HostBuf& hb, VkFormat fmt, VkExtent2D e, const std::string& path) {
  void* p = nullptr;
  if (dd->d.MapMemory(dd->device, hb.mem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS) return false;
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) { dd->d.UnmapMemory(dd->device, hb.mem); return false; }
  fprintf(f, "P6\n%u %u\n255\n", e.width, e.height);
  const unsigned char* px = (const unsigned char*)p;
  const uint32_t bpp = BytesPerPixel(fmt);
  std::vector<unsigned char> row(e.width * 3);
  for (uint32_t y = 0; y < e.height; ++y) {
    for (uint32_t x = 0; x < e.width; ++x) {
      const unsigned char* v = px + (size_t(y) * e.width + x) * bpp;
      unsigned char r, g, b;
      Decode(v, fmt, r, g, b);
      row[x * 3] = r; row[x * 3 + 1] = g; row[x * 3 + 2] = b;
    }
    fwrite(row.data(), 1, row.size(), f);
  }
  fclose(f);
  dd->d.UnmapMemory(dd->device, hb.mem);
  return true;
}

void UpdateParams(DeviceData* dd, VkCommandBuffer cb, float k) {
  const VkPipelineStageFlags ALL = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, XFER = VK_PIPELINE_STAGE_TRANSFER_BIT;
  // (params[3], the projection-sample flag, belongs to the scene's passes in depth / hybrid mode)
  float data[5] = {k, Cfg().convergence, HudFactor(), 0.0f, ScreenFixFactor()};
  VkMemoryBarrier pre{VK_STRUCTURE_TYPE_MEMORY_BARRIER};  // after earlier reads and the previous update
  pre.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  pre.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  dd->d.CmdPipelineBarrier(cb, ALL, XFER, 0, 1, &pre, 0, nullptr, 0, nullptr);
  if (dd->depthMode) {
    dd->d.CmdUpdateBuffer(cb, dd->paramBuf, 0, 12, data);
    dd->d.CmdUpdateBuffer(cb, dd->paramBuf, 16, 4, &data[4]);
  } else {
    dd->d.CmdUpdateBuffer(cb, dd->paramBuf, 0, sizeof(data), data);
  }
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
  dd->d.CmdPipelineBarrier(cb, XFER, ALL, 0, 1, &mb, 0, nullptr, 0, nullptr);
}
void FullBarrier(DeviceData* dd, VkCommandBuffer cb) {
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                           nullptr, 0, nullptr);
}

// ------------------------------------------------------------------------------------
// Pairs (slots): written by the game thread, read by the presenter thread
// ------------------------------------------------------------------------------------
// caller holds sc->pm. Frees the pairs whose presenter copies have finished.
void ReleaseDoneLocked(DeviceData* dd, Swapchain* sc) {
  for (auto& e : sc->ring)
    if (e.pending && dd->d.GetFenceStatus(dd->device, e.fence) == VK_SUCCESS) {
      if (e.slot >= 0) --sc->busy[e.slot];
      e.pending = false;
    }
}

// A pair the game can write: not on screen, not waiting to be shown, not being read.
int AcquireWriteSlot(DeviceData* dd, Swapchain* sc) {
  std::unique_lock<std::mutex> l(sc->pm);
  const auto start = Clock::now();
  for (;;) {
    ReleaseDoneLocked(dd, sc);
    for (int i = 0; i < kSlots; ++i)
      if (i != sc->cur && i != sc->ready && i != sc->writing && sc->busy[i] == 0) return i;
    if (Clock::now() - start > std::chrono::milliseconds(250)) {
      Log(0, "pair slots stuck (cur %d ready %d writing %d busy %d/%d/%d), forcing", sc->cur, sc->ready, sc->writing,
          sc->busy[0], sc->busy[1], sc->busy[2]);
      for (int i = 0; i < kSlots; ++i)
        if (i != sc->cur && i != sc->ready) return i;
      return 0;
    }
    sc->pcv.wait_for(l, std::chrono::microseconds(500));
  }
}

void Publish(Swapchain* sc, int slot) {
  std::lock_guard<std::mutex> l(sc->pm);
  sc->ready = slot;  // an older pair that was never shown is simply dropped
  if (sc->writing == slot) sc->writing = -1;
  ++sc->pairsPublished;
}

// game sync: the game waits until the display has taken its previous pair (like vsync)
void WaitForDisplay(Swapchain* sc) {
  if (!Cfg().gameSync) return;
  std::unique_lock<std::mutex> l(sc->pm);
  sc->pcv.wait_for(l, std::chrono::milliseconds(40), [&] { return sc->ready < 0 || sc->stop.load(); });
}

// the game's copy of `vi` into one eye of a pair (slot images stay in GENERAL layout)
void WriteEye(DeviceData* dd, Swapchain* sc, VkCommandBuffer cb, VkImage gameImg, int slot, int eye) {
  const VkPipelineStageFlags XFER = VK_PIPELINE_STAGE_TRANSFER_BIT;
  VkImage dst = sc->slots[slot].img[eye];
  VkOffset2D so = sc->srcOff, dof = sc->dstOff;
  VkExtent2D se = sc->srcExt, de = sc->dstExt;
  if (const int in = Cfg().inputLayout) {
    // stereo video: the application's image already holds both eyes (side by side or top/bottom);
    // each half is stretched back to the whole display (the player keeps the eye's aspect)
    int e = Cfg().stereo ? (Cfg().swapEyes ? eye ^ 1 : eye) : 0;
    if (in == 1) {
      se = {sc->vext.width / 2, sc->vext.height};
      so = {int32_t(e * se.width), 0};
    } else {
      se = {sc->vext.width, sc->vext.height / 2};
      so = {0, int32_t(e * se.height)};
    }
    dof = {0, 0};
    de = sc->rext;
  }
  // previous writes of this image on this queue
  Barrier(dd, cb, dst, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, XFER, VK_ACCESS_TRANSFER_WRITE_BIT, XFER,
          VK_ACCESS_TRANSFER_WRITE_BIT);
  const bool fullDst = dof.x == 0 && dof.y == 0 && de.width == sc->rext.width && de.height == sc->rext.height;
  if (!fullDst) {
    VkClearColorValue black{};
    VkImageSubresourceRange whole{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    dd->d.CmdClearColorImage(cb, dst, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &whole);
    Barrier(dd, cb, dst, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, XFER, VK_ACCESS_TRANSFER_WRITE_BIT, XFER,
            VK_ACCESS_TRANSFER_WRITE_BIT);
  }
  CopyOrBlit(dd, sc, cb, gameImg, so, se, dst, VK_IMAGE_LAYOUT_GENERAL, dof.x, dof.y, de);
  // make the writes available (the presenter reads them from another queue after the fence)
  Barrier(dd, cb, dst, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, XFER, VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT);
}

// after the game's last submit for a pair: a fence that tells the presenter it is complete
VkResult SignalWritten(DeviceData* dd, VkQueue q, Swapchain* sc, int slot) {
  VkFence f = sc->slots[slot].written;
  dd->d.WaitForFences(dd->device, 1, &f, VK_TRUE, 2000000000ull);  // a dropped pair may still be in flight
  dd->d.ResetFences(dd->device, 1, &f);
  VkSubmitInfo e{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
  return dd->d.QueueSubmit(q, 1, &e, f);
}

// ------------------------------------------------------------------------------------
// Presenter thread: left, right, left, right... one image per display refresh
// ------------------------------------------------------------------------------------
void PresenterMain(DeviceData* dd, Swapchain* sc) {
  pthread_setname_np(pthread_self(), "3dvision-presnt");
  sched_param sp{};
  sp.sched_priority = 10;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);  // only if allowed; harmless otherwise

  const VkDevice dev = dd->device;
  const VkQueue q = PresQueue(dd);
  QueueInfo* qi = GetQueueInfo(dd, q);
  const uint32_t nImages = uint32_t(sc->realImages.size());
  const bool usePresentWait = dd->presentWait;
  const VkPipelineStageFlags XFER = VK_PIPELINE_STAGE_TRANSFER_BIT;
  uint64_t k = 0;  // presents done (present id of present k is k + 1)
  int eye = 0;     // eye of the next present: 0 left, 1 right
  int eyeOfId[64] = {};
  int errorsInRow = 0;
  auto statStart = Clock::now();
  uint64_t statK = 0, statShown = 0;

  SyncReset();
  while (!sc->stop.load()) {
    PresEntry& e = sc->ring[k % kRing];
    if (e.used) dd->d.WaitForFences(dev, 1, &e.fence, VK_TRUE, 1000000000ull);

    uint32_t idx = 0;
    VkResult ar = dd->d.AcquireNextImageKHR(dev, sc->handle, 100000000ull, e.acq, VK_NULL_HANDLE, &idx);
    if (ar == VK_TIMEOUT || ar == VK_NOT_READY) continue;
    if (ar < 0) {
      if (errorsInRow++ == 0) Log(1, "presenter: acquire failed (%d), waiting for the game to recreate the swapchain", ar);
      sc->presError = ar;
      SyncReset();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      continue;
    }
    errorsInRow = 0;
    if (!usePresentWait) {
      // without present wait: with FIFO and N images, the acquire returns when the present
      // made N-1 presents ago reaches the screen
      int shownEye = (eye ^ int((nImages - 1) & 1)) & 1;
      if (k >= nImages) SyncObserve(NowSeconds(), shownEye);
    }

    int slot;
    {
      std::lock_guard<std::mutex> l(sc->pm);
      ReleaseDoneLocked(dd, sc);
      if (e.used) dd->d.ResetFences(dev, 1, &e.fence);
      // a new pair only starts with the left eye, so both eyes of what is shown belong together
      if (eye == 0 && sc->ready >= 0 &&
          dd->d.GetFenceStatus(dev, sc->slots[sc->ready].written) == VK_SUCCESS) {
        sc->cur = sc->ready;
        sc->ready = -1;
        ++sc->pairsShown;
        sc->pcv.notify_all();
      }
      slot = sc->cur;
      if (slot >= 0) ++sc->busy[slot];
      e.slot = slot;
      e.pending = false;  // set after the submit: nobody may query the fence while it is submitted
    }

    // record: pair image (or test pattern) -> real swapchain image
    VkImage real = sc->realImages[idx];
    VkCommandBuffer cb = e.cmd;
    dd->d.ResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    dd->d.BeginCommandBuffer(cb, &bi);
    Barrier(dd, cb, real, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, XFER, 0, XFER,
            VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageSubresourceRange whole{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (Cfg().testPattern || slot < 0) {
      // test pattern: left eye white, right eye black (each eye should see only its colour)
      VkClearColorValue c{};
      float v = (Cfg().testPattern && eye == 0) ? 1.0f : 0.0f;
      c.float32[0] = c.float32[1] = c.float32[2] = v;
      c.float32[3] = 1.0f;
      dd->d.CmdClearColorImage(cb, real, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &whole);
    } else {
      VkImage src = sc->slots[slot].img[eye];
      Barrier(dd, cb, src, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, XFER, 0, XFER,
              VK_ACCESS_TRANSFER_READ_BIT);
      VkImageCopy cpy{};
      cpy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      cpy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      cpy.extent = {sc->rext.width, sc->rext.height, 1};
      dd->d.CmdCopyImage(cb, src, VK_IMAGE_LAYOUT_GENERAL, real, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cpy);
    }
    Barrier(dd, cb, real, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, XFER,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
    dd->d.EndCommandBuffer(cb);

    VkPipelineStageFlags waitStage = XFER;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &e.acq;
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &e.done;
    const uint64_t id = k + 1;
    VkPresentIdKHR pid{VK_STRUCTURE_TYPE_PRESENT_ID_KHR};
    pid.swapchainCount = 1;
    pid.pPresentIds = &id;
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, usePresentWait ? &pid : nullptr};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &e.done;
    pi.swapchainCount = 1;
    pi.pSwapchains = &sc->handle;
    pi.pImageIndices = &idx;
    VkResult sr, pr;
    {
      std::lock_guard<std::mutex> l(qi->m);
      sr = dd->d.QueueSubmit(q, 1, &si, e.fence);
      pr = sr == VK_SUCCESS ? dd->d.QueuePresentKHR(q, &pi) : sr;
    }
    e.used = sr == VK_SUCCESS;
    {
      std::lock_guard<std::mutex> l(sc->pm);
      if (sr == VK_SUCCESS) e.pending = true;      // busy is released when the fence signals
      else if (e.slot >= 0) --sc->busy[e.slot];   // never submitted: undo the busy count
    }
    if (pr < 0) {
      Log(1, "presenter: present failed (%d)", pr);
      sc->presError = pr;
      SyncReset();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    eyeOfId[id % 64] = eye;
    ++k;
    ++sc->refreshes;
    eye ^= 1;

    // exact timing: wait until the previous present reaches the screen (keeps 1-2 queued)
    if (usePresentWait && id >= 2) {
      VkResult wr = dd->d.WaitForPresentKHR(dev, sc->handle, id - 1, 100000000ull);
      if (wr == VK_SUCCESS) SyncObserve(NowSeconds(), eyeOfId[(id - 1) % 64]);
      else if (wr < 0) sc->presError = wr;
    }

    auto now = Clock::now();
    double el = std::chrono::duration<double>(now - statStart).count();
    if (el >= 5.0) {
      uint64_t shown;
      {
        std::lock_guard<std::mutex> l(sc->pm);
        shown = sc->pairsShown;
      }
      Log(1, "display %.1f Hz (%.1f per eye) | game pairs shown %.1f/s", (k - statK) / el, (k - statK) / el / 2,
          (shown - statShown) / el);
      statStart = now;
      statK = k;
      statShown = shown;
    }
  }
}

void StopPresenter(DeviceData* dd, Swapchain* sc) {
  if (!sc->presThread.joinable()) return;
  sc->stop = true;
  sc->pcv.notify_all();
  sc->presThread.join();
  VkQueue q = PresQueue(dd);
  if (q) {
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
    dd->d.QueueWaitIdle(q);
  }
  SyncReset();
}

void DestroySwapchainResources(DeviceData* dd, Swapchain* sc) {
  VkDevice dev = dd->device;
  for (auto& v : sc->v)
    if (v.fence) dd->d.WaitForFences(dev, 1, &v.fence, VK_TRUE, 2000000000ull);
  for (auto& s : sc->slots)
    if (s.written) dd->d.WaitForFences(dev, 1, &s.written, VK_TRUE, 2000000000ull);
  for (auto& v : sc->v) {
    if (v.cmd) dd->d.FreeCommandBuffers(dev, sc->pool, 1, &v.cmd);
    if (v.cmd2) dd->d.FreeCommandBuffers(dev, sc->pool, 1, &v.cmd2);
    if (v.fence) dd->d.DestroyFence(dev, v.fence, nullptr);
    if (v.image) {
      std::lock_guard<std::mutex> l(dd->depthLock);
      dd->presentedImgs.erase(v.image);
    }
    if (v.image) dd->d.DestroyImage(dev, v.image, nullptr);
    if (v.mem) dd->d.FreeMemory(dev, v.mem, nullptr);
  }
  for (auto& s : sc->slots) {
    for (int e = 0; e < 2; ++e) {
      if (s.img[e]) dd->d.DestroyImage(dev, s.img[e], nullptr);
      if (s.mem[e]) dd->d.FreeMemory(dev, s.mem[e], nullptr);
    }
    if (s.written) dd->d.DestroyFence(dev, s.written, nullptr);
  }
  for (auto& e : sc->ring) {
    if (e.used && e.fence) dd->d.WaitForFences(dev, 1, &e.fence, VK_TRUE, 2000000000ull);
    if (e.cmd) dd->d.FreeCommandBuffers(dev, sc->presPool, 1, &e.cmd);
    if (e.fence) dd->d.DestroyFence(dev, e.fence, nullptr);
    if (e.acq) dd->d.DestroySemaphore(dev, e.acq, nullptr);
    if (e.done) dd->d.DestroySemaphore(dev, e.done, nullptr);
  }
  if (sc->presPool) dd->d.DestroyCommandPool(dev, sc->presPool, nullptr);
  if (sc->pool) dd->d.DestroyCommandPool(dev, sc->pool, nullptr);
  for (int e = 0; e < 2; ++e) {
    if (sc->synth[e]) dd->d.DestroyImage(dev, sc->synth[e], nullptr);
    if (sc->synthMem[e]) dd->d.FreeMemory(dev, sc->synthMem[e], nullptr);
  }
  if (sc->work) dd->d.DestroyBuffer(dev, sc->work, nullptr);
  if (sc->workMem) dd->d.FreeMemory(dev, sc->workMem, nullptr);
}

bool AllocCmd(DeviceData* dd, VkCommandPool pool, VkCommandBuffer* cb) {
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = pool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  if (!pool || dd->d.AllocateCommandBuffers(dd->device, &cai, cb) != VK_SUCCESS) return false;
  if (dd->setLoaderData) dd->setLoaderData(dd->device, *cb);
  else *(void**)*cb = *(void**)dd->device;
  return true;
}

VKAPI_ATTR VkResult VKAPI_CALL hook_CreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR* ci,
                                                       const VkAllocationCallbacks* a, VkSwapchainKHR* pSwapchain) {
  DeviceData* dd = Dev(device);
  InstanceData* inst = dd->inst;
  if (Cfg().passthrough) return dd->d.CreateSwapchainKHR(device, ci, a, pSwapchain);
  // the old swapchain's presenter must stop using it before it is retired
  if (ci->oldSwapchain)
    if (Swapchain* old = FindSwapchain(dd, ci->oldSwapchain)) StopPresenter(dd, old);

  // optional: own full-screen window on the 3D monitor (NV3D_OWN_WINDOW=1)
  const bool wantOwn = Cfg().ownWindow == 1;
  if (wantOwn && dd->outState == 0) {
    dd->outState = -1;
    if (inst->createXlibSurface && dd->gfxFamily != UINT32_MAX) {
      dd->out = CreateOutputWindow();
      if (dd->out) {
        XlibSurfaceCI xci{VkStructureType(1000004000) /*XLIB_SURFACE_CREATE_INFO_KHR*/, nullptr, 0,
                          OutputDisplay(dd->out), OutputXid(dd->out)};
        VkBool32 sup = VK_FALSE;
        if (((PFN_CreateXlibSurface)inst->createXlibSurface)(inst->instance, &xci, nullptr, &dd->outSurface) ==
                VK_SUCCESS &&
            inst->d.GetPhysicalDeviceSurfaceSupportKHR(dd->phys, dd->gfxFamily, dd->outSurface, &sup) == VK_SUCCESS &&
            sup) {
          dd->outState = 1;
        } else {
          Log(0, "own output window: cannot present to it, using the game window");
          if (dd->outSurface) inst->d.DestroySurfaceKHR(inst->instance, dd->outSurface, nullptr);
          dd->outSurface = VK_NULL_HANDLE;
          DestroyOutputWindow(dd->out);
          dd->out = nullptr;
        }
      }
    }
  }
  const bool useOwn = wantOwn && dd->outState == 1;
  if (dd->out) SetOutputWindowVisible(dd->out, useOwn);
  const VkSurfaceKHR realSurface = useOwn ? dd->outSurface : ci->surface;
  VkSurfaceCapabilitiesKHR caps{};
  inst->d.GetPhysicalDeviceSurfaceCapabilitiesKHR(dd->phys, realSurface, &caps);
  VkBool32 presentOk = VK_FALSE;
  if (dd->gfxFamily != UINT32_MAX)
    inst->d.GetPhysicalDeviceSurfaceSupportKHR(dd->phys, dd->gfxFamily, realSurface, &presentOk);
  if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) || dd->gfxFamily == UINT32_MAX || !presentOk ||
      !dd->gfxQueue) {
    Log(0, "surface cannot be used by the presenter (transfer dst %d, graphics queue %d, present support %d): "
        "3D disabled for this swapchain", !!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT),
        dd->gfxQueue != VK_NULL_HANDLE, presentOk);
    return dd->d.CreateSwapchainKHR(device, ci, a, pSwapchain);
  }

  VkExtent2D rext = caps.currentExtent;
  if (rext.width == 0xFFFFFFFFu) {
    rext = ci->imageExtent;
    rext.width = std::clamp(rext.width, caps.minImageExtent.width, caps.maxImageExtent.width);
    rext.height = std::clamp(rext.height, caps.minImageExtent.height, caps.maxImageExtent.height);
  }

  VkSwapchainCreateInfoKHR rci = *ci;
  rci.surface = realSurface;
  rci.imageExtent = rext;
  // the format the game asked for must exist on our surface; otherwise pick one and blit
  VkFormat realFormat = ci->imageFormat;
  VkColorSpaceKHR realSpace = ci->imageColorSpace;
  {
    uint32_t nf = 0;
    inst->d.GetPhysicalDeviceSurfaceFormatsKHR(dd->phys, realSurface, &nf, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(nf);
    inst->d.GetPhysicalDeviceSurfaceFormatsKHR(dd->phys, realSurface, &nf, fmts.data());
    bool found = false;
    for (auto& f : fmts) found |= f.format == realFormat && f.colorSpace == realSpace;
    if (!found && nf) {
      realFormat = fmts[0].format;
      realSpace = fmts[0].colorSpace;
      for (auto& f : fmts)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM) { realFormat = f.format; realSpace = f.colorSpace; }
    }
  }
  rci.imageFormat = realFormat;
  rci.imageColorSpace = realSpace;
  rci.flags &= ~VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR;  // we only copy into our images
  rci.pNext = nullptr;
  // the real images are only used by our presenter queue (graphics family)
  rci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  rci.queueFamilyIndexCount = 0;
  rci.pQueueFamilyIndices = nullptr;
  rci.imageArrayLayers = 1;
  if (ci->oldSwapchain) {  // oldSwapchain must belong to the same surface
    Swapchain* old = FindSwapchain(dd, ci->oldSwapchain);
    if (old && old->onOwn != useOwn) rci.oldSwapchain = VK_NULL_HANDLE;
  }
  rci.compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                           ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR : ci->compositeAlpha;
  rci.preTransform = caps.currentTransform;
  rci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  // vsync is what makes frame-sequential 3D work: always FIFO, one image per refresh
  rci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  rci.minImageCount = std::max<uint32_t>(uint32_t(Cfg().presentImages), caps.minImageCount);
  if (caps.maxImageCount) rci.minImageCount = std::min(rci.minImageCount, caps.maxImageCount);

  VkResult r = dd->d.CreateSwapchainKHR(device, &rci, a, pSwapchain);
  if (r != VK_SUCCESS) return r;

  auto* sc = new Swapchain();
  sc->handle = *pSwapchain;
  sc->format = ci->imageFormat;
  sc->realFormat = realFormat;
  sc->formatDiffers = realFormat != ci->imageFormat;
  sc->onOwn = useOwn;
  sc->vext = ci->imageExtent;
  if (dd->depthMode) {
    EnsureDepthBuffer(dd, sc->vext);
    // HUD separation: the copy of the scene without HUD (the game's back buffer, 4 bytes per pixel)
    VkDeviceSize snapNeed = VkDeviceSize(sc->vext.width) * sc->vext.height * 4;
    VkBuffer nb = VK_NULL_HANDLE;
    VkDeviceMemory nm = VK_NULL_HANDLE;
    VkDeviceAddress na = 0;
    bool grow = dd->snapSize < snapNeed * kSnapSlots;
    if (grow && !CreateDevBuf(dd, snapNeed * kSnapSlots,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &nb, &nm, &na))
      grow = false;
    std::lock_guard<std::mutex> l(dd->depthLock);
    dd->gameExt = sc->vext;
    dd->scFormat = sc->format;
    if (grow) {
      if (dd->snapBuf) dd->retiredBufs.push_back({dd->snapBuf, dd->snapMem});
      dd->snapBuf = nb;
      dd->snapMem = nm;
      dd->snapAddr = na;
      dd->snapSize = snapNeed * kSnapSlots;
    }
    dd->snapSlotBytes = dd->snapSize >= snapNeed * kSnapSlots ? snapNeed : 0;
    for (auto& f : dd->slotFrame) f = 0;
    dd->bbImg = dd->bbCand = dd->lastColorImg = VK_NULL_HANDLE;  // learnt again for this swapchain
    dd->bbUsable = false;
    dd->bbVotes = 0;
  }
  sc->rext = rext;
  sc->srcOff = {0, 0};
  sc->srcExt = sc->vext;
  sc->dstOff = {0, 0};
  sc->dstExt = rext;
  {
    double eyeAspect = double(rext.width) / double(rext.height);
    double gameAspect = double(sc->vext.width) / double(sc->vext.height);
    int crop = Cfg().crop;
    bool wide = gameAspect > eyeAspect * 1.25;
    sc->detectOn = crop == 2 || (crop < 0 && wide);
    bool doCrop = crop == 1 || (crop < 0 && wide);
    sc->cx0 = 0; sc->cy0 = 0; sc->cx1 = int(sc->vext.width); sc->cy1 = int(sc->vext.height);
    if (doCrop) {
      uint32_t w = std::min<uint32_t>(sc->vext.width, uint32_t(sc->vext.height * eyeAspect + 0.5));
      sc->cx0 = int((sc->vext.width - w) / 2);
      sc->cx1 = sc->cx0 + int(w);
    }
    ComputeRects(sc);
  }
  uint32_t n = 0;
  dd->d.GetSwapchainImagesKHR(device, sc->handle, &n, nullptr);
  sc->realImages.resize(n);
  dd->d.GetSwapchainImagesKHR(device, sc->handle, &n, sc->realImages.data());

  VkFormatProperties fp{};
  inst->d.GetPhysicalDeviceFormatProperties(dd->phys, sc->format, &fp);
  VkFormatProperties fpr{};
  inst->d.GetPhysicalDeviceFormatProperties(dd->phys, realFormat, &fpr);
  sc->canBlit = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) &&
                (fpr.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT);

  bool ok = true;
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = dd->gfxFamily;
  sc->poolFamily = dd->gfxFamily;
  ok &= dd->d.CreateCommandPool(device, &pci, nullptr, &sc->pool) == VK_SUCCESS;
  ok &= dd->d.CreateCommandPool(device, &pci, nullptr, &sc->presPool) == VK_SUCCESS;

  // virtual images handed to the game
  VkImageFormatListCreateInfo fmtList{VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO};
  bool haveFmtList = false;
  for (auto* p = (const VkBaseInStructure*)ci->pNext; p; p = p->pNext)
    if (p->sType == VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO) {
      fmtList = *(const VkImageFormatListCreateInfo*)p;
      fmtList.pNext = nullptr;
      haveFmtList = true;
    }
  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  if (ci->flags & VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR) {
    ici.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
    if (haveFmtList) ici.pNext = &fmtList;
  }
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = ci->imageFormat;
  ici.extent = {ci->imageExtent.width, ci->imageExtent.height, 1};
  ici.mipLevels = 1;
  ici.arrayLayers = std::max(1u, ci->imageArrayLayers);
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = ci->imageUsage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  ici.sharingMode = ci->imageSharingMode;
  ici.queueFamilyIndexCount = ci->queueFamilyIndexCount;
  ici.pQueueFamilyIndices = ci->pQueueFamilyIndices;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  uint32_t count = std::max(ci->minImageCount, 3u);
  sc->v.resize(count);
  for (auto& v : sc->v) {
    ok &= CreateImageMem(dd, ici, &v.image, &v.mem);
    if (dd->depthMode && v.image) {
      std::lock_guard<std::mutex> l(dd->depthLock);
      dd->presentedImgs.insert(v.image);
    }
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    ok &= dd->d.CreateFence(device, &fci, nullptr, &v.fence) == VK_SUCCESS;
    ok &= AllocCmd(dd, sc->pool, &v.cmd);
    ok &= AllocCmd(dd, sc->pool, &v.cmd2);
  }
  // stereo pairs
  VkImageCreateInfo sci = ici;
  sci.format = realFormat;
  sci.pNext = nullptr;
  sci.flags = 0;
  sci.extent = {rext.width, rext.height, 1};
  sci.arrayLayers = 1;
  sci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  sci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  sci.queueFamilyIndexCount = 0;
  sci.pQueueFamilyIndices = nullptr;
  for (auto& s : sc->slots) {
    for (int e = 0; e < 2; ++e) ok &= CreateImageMem(dd, sci, &s.img[e], &s.mem[e]);
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    ok &= dd->d.CreateFence(device, &fci, nullptr, &s.written) == VK_SUCCESS;
  }
  // presenter ring
  for (auto& e : sc->ring) {
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    ok &= dd->d.CreateFence(device, &fci, nullptr, &e.fence) == VK_SUCCESS;
    VkSemaphoreCreateInfo semci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    ok &= dd->d.CreateSemaphore(device, &semci, nullptr, &e.acq) == VK_SUCCESS;
    ok &= dd->d.CreateSemaphore(device, &semci, nullptr, &e.done) == VK_SUCCESS;
    ok &= AllocCmd(dd, sc->presPool, &e.cmd);
  }
  // pairs start black, in GENERAL layout for good
  if (ok) {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    ok &= AllocCmd(dd, sc->pool, &cb);
    if (ok) {
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      dd->d.BeginCommandBuffer(cb, &bi);
      VkClearColorValue black{};
      VkImageSubresourceRange whole{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      for (auto& s : sc->slots)
        for (int e = 0; e < 2; ++e) {
          Barrier(dd, cb, s.img[e], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
          dd->d.CmdClearColorImage(cb, s.img[e], VK_IMAGE_LAYOUT_GENERAL, &black, 1, &whole);
        }
      VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
      dd->d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                               nullptr, 0, nullptr);
      dd->d.EndCommandBuffer(cb);
      VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si.commandBufferCount = 1;
      si.pCommandBuffers = &cb;
      std::lock_guard<std::mutex> l(GetQueueInfo(dd, dd->gfxQueue)->m);
      ok &= dd->d.QueueSubmit(dd->gfxQueue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS;
      dd->d.QueueWaitIdle(dd->gfxQueue);
      dd->d.FreeCommandBuffers(device, sc->pool, 1, &cb);
    }
  }

  if (!ok) {
    Log(0, "failed to create virtual swapchain resources");
    DestroySwapchainResources(dd, sc);
    delete sc;
    dd->d.DestroySwapchainKHR(device, *pSwapchain, a);
    *pSwapchain = VK_NULL_HANDLE;
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  }
  {
    std::lock_guard<std::mutex> l(dd->swapLock);
    dd->swapchains[sc->handle] = sc;
  }
  ++dd->swapCount;
  Log(1, "swapchain (%s): game %ux%u -> display %ux%u frame-sequential, %u virtual / %u real images (FIFO), "
      "format %d->%d, blit %d",
      useOwn ? "own window" : "game window", sc->vext.width, sc->vext.height, rext.width, rext.height, count, n,
      sc->format, realFormat, sc->canBlit);
  if (Cfg().emitter) SyncStart();
  sc->presThread = std::thread(PresenterMain, dd, sc);
  if (dd->recordResume) {
    dd->recordResume = false;
    sc->rec = CreateRecorder(dd, sc);
  }
  return VK_SUCCESS;
}

Swapchain* FindSwapchain(DeviceData* dd, VkSwapchainKHR h) {
  std::lock_guard<std::mutex> l(dd->swapLock);
  auto it = dd->swapchains.find(h);
  return it == dd->swapchains.end() ? nullptr : it->second;
}

VKAPI_ATTR void VKAPI_CALL hook_DestroySwapchainKHR(VkDevice device, VkSwapchainKHR h, const VkAllocationCallbacks* a) {
  DeviceData* dd = Dev(device);
  Swapchain* sc = nullptr;
  {
    std::lock_guard<std::mutex> l(dd->swapLock);
    auto it = dd->swapchains.find(h);
    if (it != dd->swapchains.end()) {
      sc = it->second;
      dd->swapchains.erase(it);
    }
  }
  bool wasRecording = false;
  if (sc) {
    --dd->swapCount;
    if (sc->rec) {  // keep recording on the next swapchain (resolution change, alt-tab...)
      wasRecording = true;
      DestroyRecorder(sc->rec);
      sc->rec = nullptr;
    }
    StopPresenter(dd, sc);
  }
  if (wasRecording) {  // continue (in a new file) on the swapchain that replaced this one
    Swapchain* other = nullptr;
    {
      std::lock_guard<std::mutex> l(dd->swapLock);
      if (!dd->swapchains.empty()) other = dd->swapchains.begin()->second;
    }
    if (other && !other->rec) other->rec = CreateRecorder(dd, other);
    else dd->recordResume = true;
  }
  if (sc && dd->replay) {
    std::lock_guard<std::mutex> rl(dd->recLock);
    AbortReplayLocked(dd, "swapchain destroyed");
  }
  if (sc && dd->gfxQueue) {
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, dd->gfxQueue)->m);
    dd->d.QueueWaitIdle(dd->gfxQueue);
  }
  dd->d.DestroySwapchainKHR(device, h, a);
  if (sc) {
    DestroySwapchainResources(dd, sc);
    delete sc;
  }
}

VKAPI_ATTR VkResult VKAPI_CALL hook_GetSwapchainImagesKHR(VkDevice device, VkSwapchainKHR h, uint32_t* count,
                                                          VkImage* images) {
  DeviceData* dd = Dev(device);
  Swapchain* sc = FindSwapchain(dd, h);
  if (!sc) return dd->d.GetSwapchainImagesKHR(device, h, count, images);
  uint32_t n = uint32_t(sc->v.size());
  if (!images) {
    *count = n;
    return VK_SUCCESS;
  }
  uint32_t c = std::min(*count, n);
  for (uint32_t i = 0; i < c; ++i) images[i] = sc->v[i].image;
  *count = c;
  return c < n ? VK_INCOMPLETE : VK_SUCCESS;
}

VkResult AcquireVirtual(DeviceData* dd, Swapchain* sc, uint64_t timeout, VkSemaphore sem, VkFence fence,
                        uint32_t* idx) {
  int pe = sc->presError.load();
  if (pe == VK_ERROR_OUT_OF_DATE_KHR || pe == VK_ERROR_SURFACE_LOST_KHR) return VkResult(pe);
  uint32_t i = sc->nextAcquire;
  dd->stage = "game acquire: waiting for the previous copy of this image";
  VkResult w = dd->d.WaitForFences(dd->device, 1, &sc->v[i].fence, VK_TRUE, timeout);
  dd->stage = "game acquire: done";
  if (w == VK_TIMEOUT) return timeout == 0 ? VK_NOT_READY : VK_TIMEOUT;
  if (w != VK_SUCCESS) return w;
  sc->nextAcquire = (i + 1) % uint32_t(sc->v.size());
  if (sem != VK_NULL_HANDLE || fence != VK_NULL_HANDLE) {
    VkQueue q = dd->gfxQueue;
    if (!q) return VK_ERROR_DEVICE_LOST;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.signalSemaphoreCount = sem ? 1 : 0;
    si.pSignalSemaphores = &sem;
    dd->stage = "game acquire: signalling (waiting for queue lock)";
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
    VkResult r = dd->d.QueueSubmit(q, 1, &si, fence);
    dd->stage = "game is rendering";
    if (r != VK_SUCCESS) return r;
  }
  *idx = i;
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL hook_AcquireNextImageKHR(VkDevice device, VkSwapchainKHR h, uint64_t timeout,
                                                        VkSemaphore sem, VkFence fence, uint32_t* idx) {
  DeviceData* dd = Dev(device);
  Swapchain* sc = FindSwapchain(dd, h);
  if (!sc) return dd->d.AcquireNextImageKHR(device, h, timeout, sem, fence, idx);
  return AcquireVirtual(dd, sc, timeout, sem, fence, idx);
}
VKAPI_ATTR VkResult VKAPI_CALL hook_AcquireNextImage2KHR(VkDevice device, const VkAcquireNextImageInfoKHR* info,
                                                         uint32_t* idx) {
  DeviceData* dd = Dev(device);
  Swapchain* sc = FindSwapchain(dd, info->swapchain);
  if (!sc) return dd->d.AcquireNextImage2KHR(device, info, idx);
  return AcquireVirtual(dd, sc, info->timeout, info->semaphore, info->fence, idx);
}

// ------------------------------------------------------------------------------------
// Game side: the game's frame becomes one eye (alternate) or both eyes (replay) of a pair
// ------------------------------------------------------------------------------------
// Ctrl+Shift+F11: game image + both eyes of the pair as PPM
struct Capture {
  HostBuf game, eye[2];
  bool on = false;
};
bool BeginCapture(DeviceData* dd, Swapchain* sc, Capture& c, uint64_t frame) {
  if (!TakeDumpRequest(frame)) return false;
  if (!Is32bpp(sc->format) || !Is32bpp(sc->realFormat)) {
    Log(0, "capture: unsupported format %d/%d", sc->format, sc->realFormat);
    return false;
  }
  c.on = CreateHostBuf(dd, VkDeviceSize(sc->vext.width) * sc->vext.height * BytesPerPixel(sc->format), c.game) &&
         CreateHostBuf(dd, VkDeviceSize(sc->rext.width) * sc->rext.height * BytesPerPixel(sc->realFormat), c.eye[0]) &&
         CreateHostBuf(dd, VkDeviceSize(sc->rext.width) * sc->rext.height * BytesPerPixel(sc->realFormat), c.eye[1]);
  return c.on;
}
void RecordCapture(DeviceData* dd, Swapchain* sc, VkCommandBuffer cb, Capture& c, int slot) {
  const VkPipelineStageFlags XFER = VK_PIPELINE_STAGE_TRANSFER_BIT;
  for (int e = 0; e < 2; ++e) {
    Barrier(dd, cb, sc->slots[slot].img[e], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, XFER,
            VK_ACCESS_TRANSFER_WRITE_BIT, XFER, VK_ACCESS_TRANSFER_READ_BIT);
    CopyToBuf(dd, cb, sc->slots[slot].img[e], VK_IMAGE_LAYOUT_GENERAL, sc->rext, c.eye[e]);
  }
}
void FinishCapture(DeviceData* dd, Swapchain* sc, Capture& c, VkFence wait, uint64_t frame) {
  if (!c.on) return;
  dd->d.WaitForFences(dd->device, 1, &wait, VK_TRUE, UINT64_MAX);
  std::string base = DumpDir() + "/" + GameName();
  // the earlier captures are kept: <game>_1_* is the previous one ... <game>_5_* the oldest
  static const char* kParts[] = {"_game.ppm", "_left.ppm", "_right.ppm", "_info.txt"};
  for (int i = 4; i >= 0; --i)
    for (const char* part : kParts) {
      const std::string from = i ? base + "_" + std::to_string(i) + part : base + part;
      rename(from.c_str(), (base + "_" + std::to_string(i + 1) + part).c_str());
    }
  bool ok = WritePPM(dd, c.game, sc->format, sc->vext, base + "_game.ppm") &&
            WritePPM(dd, c.eye[0], sc->realFormat, sc->rext, base + "_left.ppm") &&
            WritePPM(dd, c.eye[1], sc->realFormat, sc->rext, base + "_right.ppm");
  if (FILE* f = fopen((base + "_info.txt").c_str(), "w")) {
    fprintf(f, "game_image=%ux%u\ndisplay=%ux%u\nsrc=%d,%d %ux%u\ndst=%d,%d %ux%u\nformat=%d\nframe=%llu\n%s\n",
            sc->vext.width, sc->vext.height, sc->rext.width, sc->rext.height, sc->srcOff.x, sc->srcOff.y,
            sc->srcExt.width, sc->srcExt.height, sc->dstOff.x, sc->dstOff.y, sc->dstExt.width, sc->dstExt.height,
            sc->format, (unsigned long long)frame, SyncStatusLine().c_str());
    fclose(f);
  }
  Log(0, ok ? "capture saved: %s_{game,left,right}.ppm" : "capture FAILED: %s", base.c_str());
  FreeHostBuf(dd, c.game);
  FreeHostBuf(dd, c.eye[0]);
  FreeHostBuf(dd, c.eye[1]);
}

void LogStats(DeviceData* dd, Swapchain* sc, bool replayStats) {
  auto now = Clock::now();
  ++sc->statFrames;
  double el = std::chrono::duration<double>(now - sc->statStart).count();
  if (el < 5.0) return;
  uint64_t published;
  {
    std::lock_guard<std::mutex> l(sc->pm);
    published = sc->pairsPublished;
  }
  double pairs = published >= sc->statPublished ? (published - sc->statPublished) / el : 0.0;
  sc->statPublished = published;
  if (dd->screenFix) {  // only when something changed
    uint32_t a = dd->screenFixCount.load(), b = dd->depthOnlyCount.load();
    if (a != dd->lastFixLog[0] || b != dd->lastFixLog[1]) {
      dd->lastFixLog[0] = a;
      dd->lastFixLog[1] = b;
      Log(1, "fix_shadows: %u shaders corrected, %u shadow-map pipelines unshifted (Ctrl+Shift+Pause: %s)", a, b,
          Cfg().fixShadowsOn ? "on" : "off");
    }
  }
  if (replayStats) {
    uint32_t rf = dd->replayFrames.exchange(0), mf = dd->monoFrames.exchange(0);
    // the game needs GPU results in the middle of its frames: replay cannot work, use alternate
    // only judge real gameplay: menus, intros and loading screens (few shaders so far) often
    // wait for the GPU every frame and would switch good replay games to alternate for good
    if (Cfg().autoSwitch && mf > rf && rf + mf >= 30 && dd->patchedCount.load() >= 60) {
      if (++dd->badReplayWindows >= 2) {
        Log(0, "this game reads back from the GPU every frame (%u of %u frames 2D): switching to alternate "
            "mode (3D at half the game fps). Use mode = alternate in its profile to skip this.", mf, rf + mf);
        AbortReplayLocked(dd, "switching to alternate");
        dd->replay = false;
        dd->recMono = false;
        dd->frame = 0;
      }
    } else {
      dd->badReplayWindows = 0;
    }
    Log(1, "game %.1f fps -> %.1f stereo pairs/s [replay: %u frames 3D, %u frames 2D] | shaders %u | "
        "sep %.4f conv %.3f%s",
        sc->statFrames / el, pairs, rf, mf, dd->patchedCount.load(), Cfg().separation, Cfg().convergence,
        Cfg().stereo ? "" : " [2D]");
  } else if (dd->depthMode) {
    Log(1, "game %.1f fps -> %.1f stereo pairs/s [%s: depth copies %u, projection %s (%u/%u samples)] | "
        "shaders patched %u | sep %.4f conv %.3f%s",
        sc->statFrames / el, pairs, dd->hybrid ? "hybrid" : "depth", dd->depthCopies.exchange(0),
        dd->proj[dd->lastCopyGroup & 3].ok ? "ok" : "not found yet", dd->proj[dd->lastCopyGroup & 3].inliers,
        dd->proj[dd->lastCopyGroup & 3].samples, dd->patchedCount.load(), Cfg().separation, Cfg().convergence,
        Cfg().stereo ? "" : " [2D]");
  } else {
    double fps = sc->statFrames / el;
    if (!dd->hintedDepth && fps > 45 && fps < 66 && !Cfg().inputLayout && dd->presentCount.load() > 1200) {
      dd->hintedDepth = true;
      Log(0, "this game runs at ~60 fps in alternate mode: only %.0f stereo pairs/s. mode = hybrid in its profile "
          "gives one pair per game frame (one real eye, the other rebuilt from the depth buffer)", fps / 2);
    }
    Log(1, "game %.1f fps -> %.1f stereo pairs/s [alternate%s, repeated presents skipped %u] | shaders patched %u, "
        "skipped %u, depth-only pipelines unshifted %u | sep %.4f conv %.3f%s",
        sc->statFrames / el, pairs, Cfg().skipRepeats ? " skip_repeats" : "", dd->repeatFrames.exchange(0),
        dd->patchedCount.load(), dd->skippedCount.load(), dd->depthOnlyCount.load(), Cfg().separation,
        Cfg().convergence, Cfg().stereo ? "" : " [2D]");
  }
  sc->statFrames = 0;
  sc->statStart = now;
}

// ------------------------------------------------------------------------------------
// Recording (Ctrl+Shift+F12): every stereo pair is scaled into a full-SBS image on the GPU,
// read back and written as raw BGRA frames, at a constant frame rate, into a FIFO read by
// 3dvision-recorder (ffmpeg + NVENC on the host, outside the game's container).
// ------------------------------------------------------------------------------------
std::string RecordDir() {
  std::string base;
  if (const char* x = getenv("XDG_CACHE_HOME"); x && *x) base = x;
  else if (const char* h = getenv("HOME"); h && *h) base = std::string(h) + "/.cache";
  else base = "/tmp";
  return base + "/3dvision";
}

struct RecBuf {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  void* map = nullptr;
  bool coherent = true;
  VkFence fence = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  double t = 0;
  bool busy = false;
};
constexpr int kRecBufs = 4;

struct Recorder {
  DeviceData* dd = nullptr;
  uint32_t w = 0, h = 0;  // per eye
  VkFormat fmt = VK_FORMAT_B8G8R8A8_UNORM;
  VkImage img = VK_NULL_HANDLE;
  VkDeviceMemory imgMem = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  RecBuf b[kRecBufs];
  VkDeviceSize frameBytes = 0;
  std::thread th;
  std::mutex m;
  std::condition_variable cv;
  std::vector<int> queue;  // buffers waiting to be written, in order
  bool stop = false;
  double lastCapture = -1;
  std::atomic<uint64_t> captured{0}, dropped{0}, written{0};
  std::atomic<bool> failed{false};  // no recorder: stop capturing
};

void RecorderThread(Recorder* r) {
  pthread_setname_np(pthread_self(), "3dvision-record");
  sigset_t ss;  // a dead reader must not kill the game with SIGPIPE
  sigemptyset(&ss);
  sigaddset(&ss, SIGPIPE);
  pthread_sigmask(SIG_BLOCK, &ss, nullptr);
  DeviceData* dd = r->dd;
  // ask the host recorder to start ffmpeg, then connect to its FIFO
  std::string dir = RecordDir();
  {
    std::string tmp = dir + "/record.start.tmp";
    if (FILE* f = fopen(tmp.c_str(), "w")) {
      fprintf(f, "%s %ux%u %.3f\n", GameName().c_str(), r->w * 2, r->h, Cfg().recordFps);
      fclose(f);
      rename(tmp.c_str(), (dir + "/record.start").c_str());
    }
  }
  int fd = -1;
  bool spawned = false;
  for (int i = 0; i < 160 && fd < 0 && !r->stop; ++i) {  // up to 8 s
    fd = open((dir + "/record.fifo").c_str(), O_WRONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd >= 0) break;
    if (i == 10 && !spawned) {
      // no recorder: the game was not started with 3dvision-run. Start it ourselves (works for
      // native games; inside the Steam runtime container the host's ffmpeg may be missing)
      spawned = true;
      const char* home = getenv("HOME");
      std::string exe = std::string(home ? home : "") + "/.local/bin/3dvision-recorder";
      if (access(exe.c_str(), X_OK) == 0) {
        std::string pid = std::to_string(getpid());
        if (fork() == 0) {
          setsid();
          int nul = open("/dev/null", O_RDWR);
          if (nul >= 0) { dup2(nul, 0); dup2(nul, 1); dup2(nul, 2); }
          execl(exe.c_str(), exe.c_str(), "--parent", pid.c_str(), (char*)nullptr);
          _exit(127);
        }
        Log(1, "recording: started %s", exe.c_str());
        // it must see the start request again (it deletes stale ones when it starts)
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::string tmp = dir + "/record.start.tmp";
        if (FILE* f = fopen(tmp.c_str(), "w")) {
          fprintf(f, "%s %ux%u %.3f\n", GameName().c_str(), r->w * 2, r->h, Cfg().recordFps);
          fclose(f);
          rename(tmp.c_str(), (dir + "/record.start").c_str());
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (fd < 0) {
    r->failed = true;
    Log(0, "recording: 3dvision-recorder is not running and could not be started (start the game with "
           "3dvision-run; see ~/.cache/3dvision/recorder.log). Not recording.");
  } else {
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) & ~O_NONBLOCK);
    fcntl(fd, F_SETPIPE_SZ, 1 << 20);
    Log(0, "recording: started (%ux%u full SBS, %.0f fps)", r->w * 2, r->h, Cfg().recordFps);
  }
  const double period = 1.0 / Cfg().recordFps;
  double t0 = -1;
  int64_t lastIndex = -1;
  bool ok = fd >= 0;
  for (;;) {
    int bi;
    {
      std::unique_lock<std::mutex> l(r->m);
      r->cv.wait(l, [&] { return r->stop || !r->queue.empty(); });
      if (r->queue.empty()) break;  // stop requested and everything written
      bi = r->queue.front();
      r->queue.erase(r->queue.begin());
    }
    RecBuf& rb = r->b[bi];
    dd->d.WaitForFences(dd->device, 1, &rb.fence, VK_TRUE, 2000000000ull);
    if (ok) {
      if (!rb.coherent) {
        VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        mr.memory = rb.mem;
        mr.size = VK_WHOLE_SIZE;
        dd->d.InvalidateMappedMemoryRanges(dd->device, 1, &mr);
      }
      // constant frame rate: repeat this frame to fill gaps, drop it if it is early
      if (t0 < 0) t0 = rb.t;
      int64_t idx = int64_t((rb.t - t0) / period + 0.5);
      int64_t n = lastIndex < 0 ? 1 : idx - lastIndex;
      if (n > int64_t(Cfg().recordFps) * 2) n = 1;  // long pause (menu, loading): don't fill it
      for (int64_t k = 0; k < n && ok; ++k) {
        const char* p = (const char*)rb.map;
        size_t left = size_t(r->frameBytes);
        while (left && ok) {
          ssize_t wr = write(fd, p, left);
          if (wr < 0) {
            if (errno == EINTR) continue;
            Log(0, "recording: the recorder stopped (%s)", strerror(errno));
            ok = false;
            r->failed = true;
            break;
          }
          p += wr;
          left -= size_t(wr);
        }
        if (ok) ++r->written;
      }
      if (n > 0) lastIndex = lastIndex < 0 ? 0 : lastIndex + n;
    }
    std::lock_guard<std::mutex> l(r->m);
    rb.busy = false;
  }
  if (fd >= 0) close(fd);  // EOF: ffmpeg finishes the file
  Log(0, "recording: stopped (%llu frames written, %llu game pairs captured, %llu dropped)",
      (unsigned long long)r->written.load(), (unsigned long long)r->captured.load(),
      (unsigned long long)r->dropped.load());
}

void DestroyRecorder(Recorder* r) {
  if (!r) return;
  DeviceData* dd = r->dd;
  {
    std::lock_guard<std::mutex> l(r->m);
    r->stop = true;
  }
  r->cv.notify_all();
  if (r->th.joinable()) r->th.join();
  for (auto& rb : r->b) {
    if (rb.fence) dd->d.WaitForFences(dd->device, 1, &rb.fence, VK_TRUE, 2000000000ull);
    if (rb.map) dd->d.UnmapMemory(dd->device, rb.mem);
    if (rb.buf) dd->d.DestroyBuffer(dd->device, rb.buf, nullptr);
    if (rb.mem) dd->d.FreeMemory(dd->device, rb.mem, nullptr);
    if (rb.fence) dd->d.DestroyFence(dd->device, rb.fence, nullptr);
    if (rb.cmd) dd->d.FreeCommandBuffers(dd->device, r->pool, 1, &rb.cmd);
  }
  if (r->pool) dd->d.DestroyCommandPool(dd->device, r->pool, nullptr);
  if (r->img) dd->d.DestroyImage(dd->device, r->img, nullptr);
  if (r->imgMem) dd->d.FreeMemory(dd->device, r->imgMem, nullptr);
  delete r;
}

Recorder* CreateRecorder(DeviceData* dd, Swapchain* sc) {
  auto* r = new Recorder();
  r->dd = dd;
  r->w = Cfg().recordWidth ? uint32_t(Cfg().recordWidth) & ~1u : sc->rext.width;
  r->h = Cfg().recordHeight ? uint32_t(Cfg().recordHeight) & ~1u : sc->rext.height;
  // keep the encoded values: sRGB swapchains are copied into an sRGB image (no conversion)
  const bool srgb = sc->realFormat == VK_FORMAT_B8G8R8A8_SRGB || sc->realFormat == VK_FORMAT_R8G8B8A8_SRGB;
  r->fmt = srgb ? VK_FORMAT_B8G8R8A8_SRGB : VK_FORMAT_B8G8R8A8_UNORM;
  VkFormatProperties src{}, dst{};
  dd->inst->d.GetPhysicalDeviceFormatProperties(dd->phys, sc->realFormat, &src);
  dd->inst->d.GetPhysicalDeviceFormatProperties(dd->phys, r->fmt, &dst);
  if (!(src.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) ||
      !(dst.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
    Log(0, "recording: format %d cannot be scaled on this GPU", sc->realFormat);
    delete r;
    return nullptr;
  }
  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = r->fmt;
  ici.extent = {r->w * 2, r->h, 1};
  ici.mipLevels = 1;
  ici.arrayLayers = 1;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  bool ok = CreateImageMem(dd, ici, &r->img, &r->imgMem);
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = dd->gfxFamily;
  ok &= dd->d.CreateCommandPool(dd->device, &pci, nullptr, &r->pool) == VK_SUCCESS;
  r->frameBytes = VkDeviceSize(r->w) * 2 * r->h * 4;
  for (auto& rb : r->b) {
    if (!ok) break;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = r->frameBytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    ok &= dd->d.CreateBuffer(dd->device, &bci, nullptr, &rb.buf) == VK_SUCCESS;
    if (!ok) break;
    VkMemoryRequirements req;
    dd->d.GetBufferMemoryRequirements(dd->device, rb.buf, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    // cached memory: the CPU reads it (uncached reads are very slow)
    ai.memoryTypeIndex = FindMemType(dd, req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX ||
        !(dd->memProps.memoryTypes[ai.memoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
      ai.memoryTypeIndex = FindMemType(dd, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    ok &= dd->d.AllocateMemory(dd->device, &ai, nullptr, &rb.mem) == VK_SUCCESS;
    if (!ok) break;
    rb.coherent = (dd->memProps.memoryTypes[ai.memoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    dd->d.BindBufferMemory(dd->device, rb.buf, rb.mem, 0);
    ok &= dd->d.MapMemory(dd->device, rb.mem, 0, VK_WHOLE_SIZE, 0, &rb.map) == VK_SUCCESS;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    ok &= dd->d.CreateFence(dd->device, &fci, nullptr, &rb.fence) == VK_SUCCESS;
    ok &= AllocCmd(dd, r->pool, &rb.cmd);
  }
  mkdir(RecordDir().c_str(), 0755);
  if (!ok) {
    Log(0, "recording: cannot allocate the recording buffers");
    DestroyRecorder(r);
    return nullptr;
  }
  r->th = std::thread(RecorderThread, r);
  return r;
}

void ToggleRecording(DeviceData* dd, Swapchain* sc) {
  if (sc->rec) {
    DestroyRecorder(sc->rec);
    sc->rec = nullptr;
  } else {
    sc->rec = CreateRecorder(dd, sc);
  }
}

// After the game's submits for a pair, on the same queue: pair -> SBS image -> host buffer
void CapturePair(DeviceData* dd, Swapchain* sc, VkQueue q, int slot) {
  Recorder* r = sc->rec;
  if (!r || r->failed.load()) return;
  double now = NowSeconds();
  const double period = 1.0 / Cfg().recordFps;
  if (r->lastCapture >= 0 && now - r->lastCapture < period * 0.75) return;  // game faster than the video
  int bi = -1;
  {
    std::lock_guard<std::mutex> l(r->m);
    for (int i = 0; i < kRecBufs; ++i)
      if (!r->b[i].busy) { bi = i; break; }
    if (bi < 0) {  // the encoder is behind: drop this pair (the recorder repeats the previous one)
      ++r->dropped;
      return;
    }
    r->b[bi].busy = true;
  }
  r->lastCapture = now;
  RecBuf& rb = r->b[bi];
  dd->d.WaitForFences(dd->device, 1, &rb.fence, VK_TRUE, UINT64_MAX);
  dd->d.ResetFences(dd->device, 1, &rb.fence);
  VkCommandBuffer cb = rb.cmd;
  dd->d.ResetCommandBuffer(cb, 0);
  VkCommandBufferBeginInfo bi2{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi2.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  dd->d.BeginCommandBuffer(cb, &bi2);
  const VkPipelineStageFlags XFER = VK_PIPELINE_STAGE_TRANSFER_BIT;
  Barrier(dd, cb, r->img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, XFER, 0, XFER,
          VK_ACCESS_TRANSFER_WRITE_BIT);
  for (int e = 0; e < 2; ++e) {
    VkImage src = sc->slots[slot].img[e];
    Barrier(dd, cb, src, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, XFER, VK_ACCESS_TRANSFER_WRITE_BIT, XFER,
            VK_ACCESS_TRANSFER_READ_BIT);
    VkImageBlit b{};
    b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.srcOffsets[1] = {int32_t(sc->rext.width), int32_t(sc->rext.height), 1};
    b.dstOffsets[0] = {int32_t(e * r->w), 0, 0};
    b.dstOffsets[1] = {int32_t((e + 1) * r->w), int32_t(r->h), 1};
    dd->d.CmdBlitImage(cb, src, VK_IMAGE_LAYOUT_GENERAL, r->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b,
                       VK_FILTER_LINEAR);
  }
  Barrier(dd, cb, r->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, XFER,
          VK_ACCESS_TRANSFER_WRITE_BIT, XFER, VK_ACCESS_TRANSFER_READ_BIT);
  VkBufferImageCopy c{};
  c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  c.imageExtent = {r->w * 2, r->h, 1};
  dd->d.CmdCopyImageToBuffer(cb, r->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buf, 1, &c);
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  dd->d.CmdPipelineBarrier(cb, XFER, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
  dd->d.EndCommandBuffer(cb);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  VkResult res;
  {
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
    res = dd->d.QueueSubmit(q, 1, &si, rb.fence);
  }
  std::lock_guard<std::mutex> l(r->m);
  if (res != VK_SUCCESS) {
    rb.busy = false;
    return;
  }
  rb.t = now;
  ++r->captured;
  r->queue.push_back(bi);
  r->cv.notify_one();
}

// Alternate mode: even frames = left eye, odd frames = right eye of the same pair.
VkResult PresentAlternate(DeviceData* dd, VkQueue queue, Swapchain* sc, uint32_t idx, const VkSemaphore* waits,
                          uint32_t waitCount) {
  if (idx >= sc->v.size()) return VK_ERROR_OUT_OF_DATE_KHR;
  VImage& vi = sc->v[idx];
  const uint64_t frame = dd->frame;
  const bool stereoInput = Cfg().inputLayout != 0;  // every frame is a whole pair (SBS / top-bottom video)
  if (Cfg().skipRepeats && !stereoInput) {
    // a present with (almost) no GPU work since the previous one shows the same image again:
    // keep the eye and the pair as they are, otherwise the eyes get out of step with the frames
    uint64_t w = dd->work.exchange(0);
    bool repeat = dd->workEma > 0 && double(w) < dd->workEma * 0.25;
    if (!repeat) dd->workEma = dd->workEma > 0 ? dd->workEma * 0.9 + double(w) * 0.1 : double(w);
    if (repeat) {
      ++dd->repeatFrames;
      std::vector<VkPipelineStageFlags> st(waitCount, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
      VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};  // consume the game's semaphores
      si.waitSemaphoreCount = waitCount;
      si.pWaitSemaphores = waits;
      si.pWaitDstStageMask = st.data();
      std::lock_guard<std::mutex> l(GetQueueInfo(dd, queue)->m);
      return waitCount ? dd->d.QueueSubmit(queue, 1, &si, VK_NULL_HANDLE) : VK_SUCCESS;
    }
  }
  const bool right = !stereoInput && (frame & 1) != 0;
  bool both = false;  // first frame after a reset: the pair has no left eye yet
  int slot;
  if (stereoInput) {
    slot = -1;
  } else if (!right) {
    slot = AcquireWriteSlot(dd, sc);
    std::lock_guard<std::mutex> l(sc->pm);
    sc->writing = slot;
  } else {
    std::lock_guard<std::mutex> l(sc->pm);
    slot = sc->writing;
  }
  if (slot < 0) {
    slot = AcquireWriteSlot(dd, sc);
    both = true;
  }
  if (Cfg().zoom != sc->lastZoom) ComputeRects(sc);
  Capture cap;
  if (right || stereoInput) BeginCapture(dd, sc, cap, frame);
  if (sc->detectOn && !Is32bpp(sc->format)) {
    Log(0, "black-bar detection unavailable for swapchain format %d", sc->format);
    sc->detectOn = false;
  }
  HostBuf det;
  bool detect = !right && sc->detectOn && frame >= sc->nextDetect;
  if (detect) {
    sc->nextDetect = frame + (sc->detections < 20 ? 30 : 240);
    detect = CreateHostBuf(dd, VkDeviceSize(sc->vext.width) * sc->vext.height * BytesPerPixel(sc->format), det);
  }

  dd->stage = "present(alternate): waiting for previous copy";
  dd->d.WaitForFences(dd->device, 1, &vi.fence, VK_TRUE, UINT64_MAX);
  dd->d.ResetFences(dd->device, 1, &vi.fence);
  dd->stage = "present(alternate): recording";
  VkCommandBuffer cb = vi.cmd;
  dd->d.ResetCommandBuffer(cb, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  dd->d.BeginCommandBuffer(cb, &bi);
  const VkPipelineStageFlags ALL = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, XFER = VK_PIPELINE_STAGE_TRANSFER_BIT;
  Barrier(dd, cb, vi.image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ALL,
          VK_ACCESS_MEMORY_WRITE_BIT, XFER, VK_ACCESS_TRANSFER_READ_BIT);
  if (both) {
    WriteEye(dd, sc, cb, vi.image, slot, 0);
    WriteEye(dd, sc, cb, vi.image, slot, 1);
  } else {
    WriteEye(dd, sc, cb, vi.image, slot, right ? 1 : 0);
  }
  if (detect) CopyToBuf(dd, cb, vi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc->vext, det);
  if (cap.on) {
    CopyToBuf(dd, cb, vi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc->vext, cap.game);
    RecordCapture(dd, sc, cb, cap, slot);
  }
  Barrier(dd, cb, vi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, XFER, 0,
          VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
  // stereo parameters for the NEXT frame, ordered on the queue
  if (dd->stereoShaders) UpdateParams(dd, cb, EyeK(frame + 1));
  dd->d.EndCommandBuffer(cb);

  std::vector<VkPipelineStageFlags> wstages(waitCount, ALL);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.waitSemaphoreCount = waitCount;
  si.pWaitSemaphores = waits;
  si.pWaitDstStageMask = wstages.data();
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  QueueInfo* qi = GetQueueInfo(dd, queue);
  VkQueue subQueue = (qi->family == sc->poolFamily || !dd->gfxQueue) ? queue : dd->gfxQueue;
  {
    dd->stage = "present(alternate): submitting copy";
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, subQueue)->m);
    VkResult sr = dd->d.QueueSubmit(subQueue, 1, &si, vi.fence);
    if (sr != VK_SUCCESS) return sr;
  }
  if (right || both) {
    SignalWritten(dd, subQueue, sc, slot);
    Publish(sc, slot);
    CapturePair(dd, sc, subQueue, slot);
  }
  FinishCapture(dd, sc, cap, vi.fence, frame);
  if (detect) {
    dd->d.WaitForFences(dd->device, 1, &vi.fence, VK_TRUE, UINT64_MAX);
    DetectBars(dd, sc, det);
    FreeHostBuf(dd, det);
  }
  dd->frame = frame + 1;
  LogStats(dd, sc, false);
  return VK_SUCCESS;
}

// Depth mode: every game frame (rendered mono) becomes a whole pair, both eyes rebuilt from
// the image and the scene's depth buffer by two compute passes.
bool EnsureSynth(DeviceData* dd, Swapchain* sc) {
  if (sc->depthBroken) return false;
  if (sc->work) return true;
  const uint32_t W = sc->vext.width, H = sc->vext.height;
  if (BytesPerPixel(sc->format) != 4 || W > 4096) {
    Log(0, "depth mode: unsupported game image (%ux%u, format %d): the eyes will have no depth", W, H, sc->format);
    sc->depthBroken = true;
    return false;
  }
  VkDeviceSize px = VkDeviceSize(W) * H * 4;
  bool ok = CreateDevBuf(dd, px * 7,  // colour | keys x2 | eyes x2 | last real eyes x2 (hybrid)
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         &sc->work, &sc->workMem, &sc->workAddr) && sc->workAddr;
  for (int e = 0; ok && e < 2; ++e) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = sc->format;
    ici.extent = {W, H, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ok = CreateImageMem(dd, ici, &sc->synth[e], &sc->synthMem[e]);
  }
  if (!ok) {
    Log(0, "depth mode: cannot allocate the eye buffers: the eyes will have no depth");
    sc->depthBroken = true;
    return false;
  }
  return true;
}

void BufBarrier(DeviceData* dd, VkCommandBuffer cb, VkPipelineStageFlags src, VkAccessFlags sa,
                VkPipelineStageFlags dst, VkAccessFlags da) {
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = sa;
  mb.dstAccessMask = da;
  dd->d.CmdPipelineBarrier(cb, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

VkResult PresentDepth(DeviceData* dd, VkQueue queue, Swapchain* sc, uint32_t idx, const VkSemaphore* waits,
                      uint32_t waitCount) {
  if (idx >= sc->v.size()) return VK_ERROR_OUT_OF_DATE_KHR;
  VImage& vi = sc->v[idx];
  const uint64_t frame = dd->frame;
  int slot = AcquireWriteSlot(dd, sc);
  {
    std::lock_guard<std::mutex> l(sc->pm);
    sc->writing = slot;
  }
  if (Cfg().zoom != sc->lastZoom) ComputeRects(sc);
  const bool synth = EnsureSynth(dd, sc);
  if (frame >= dd->lastFitFrame + 3 || !dd->proj[dd->lastCopyGroup & 3].ok) {
    FitProjection(dd);
    dd->lastFitFrame = frame;
  }
  if (frame >= dd->lastClassFrame + 15) {
    UpdateMainDepthClass(dd);
    dd->lastClassFrame = frame;
  }
  Capture cap;
  if (dd->traceFrames.load() > 0 && --dd->traceFrames == 0) Log(1, "---- end of the pass trace");
  if (BeginCapture(dd, sc, cap, frame)) {
    dd->traceFrames = 2;
    Log(1, "---- pass trace of the next frame (Ctrl+Shift+F11):");
  }
  if (dd->traceFrames.load() > 0 && dd->paramMap) {
    volatile uint32_t* st = (volatile uint32_t*)((char*)dd->paramMap + kStatsOffset);
    Log(1, "  HUD separation, last frame: %u samples, equal to copy 0/1/2: %u %u %u, copy equal to an earlier "
           "frame: %u %u %u", st[11], st[8], st[9], st[10], st[12], st[13], st[14]);
  }
  const uint64_t sfa = dd->stageFrameAt.load();
  if (sfa == UINT64_MAX || frame > sfa + 10) {  // (unless the passes tell where frames begin)
    dd->calibGroup = 0;  // the next frame's scene stages count from 0
    std::lock_guard<std::mutex> l(dd->depthLock);
    for (auto& im : dd->stageImg) im = VK_NULL_HANDLE;
  }
  const uint32_t copied = dd->copiedStages.exchange(0);

  dd->stage = "present(depth): waiting for previous copy";
  dd->d.WaitForFences(dd->device, 1, &vi.fence, VK_TRUE, UINT64_MAX);
  dd->d.ResetFences(dd->device, 1, &vi.fence);
  dd->stage = "present(depth): recording";
  VkCommandBuffer cb = vi.cmd;
  dd->d.ResetCommandBuffer(cb, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  dd->d.BeginCommandBuffer(cb, &bi);
  const VkPipelineStageFlags ALL = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, XFER = VK_PIPELINE_STAGE_TRANSFER_BIT,
                             CS = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  Barrier(dd, cb, vi.image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ALL,
          VK_ACCESS_MEMORY_WRITE_BIT, XFER, VK_ACCESS_TRANSFER_READ_BIT);
  // the scene's depth must come from THIS frame: menus and loading screens draw no 3D scene, and
  // the depth of the last one would warp them
  const uint64_t lastDepth = dd->lastDepthFrame.load();
  const bool fresh = lastDepth != UINT64_MAX && lastDepth + 1 >= frame;
  // hybrid: this frame was rendered for one eye (as in alternate); the other eye is rebuilt from it
  const int realEye = int(frame & 1), otherEye = 1 - realEye;
  if (synth) {
    const uint32_t W = sc->vext.width, H = sc->vext.height;
    const VkDeviceSize px = VkDeviceSize(W) * H * 4;
    // previous frame's readers of the work buffer are done (same queue, fence-ordered), but make
    // the dependency explicit
    BufBarrier(dd, cb, XFER | CS, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT, XFER | CS,
               VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    VkBufferImageCopy r{};
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {W, H, 1};
    dd->d.CmdCopyImageToBuffer(cb, vi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc->work, 1, &r);
    dd->d.CmdFillBuffer(cb, sc->work, px, px * 2, 0);
    SynthPC pc{};
    pc.color = sc->workAddr;
    {
      std::lock_guard<std::mutex> l(dd->depthLock);
      pc.depth = dd->depthAddr;
    }
    pc.keys = sc->workAddr + px;
    pc.outc = sc->workAddr + px * 3;
    pc.meta = dd->metaAddr;
    pc.prev = sc->workAddr + px * 5;
    pc.stats = dd->paramAddr + kStatsOffset;
    {
      std::lock_guard<std::mutex> l(dd->depthLock);
      // the setting changed (profile, Ctrl+Shift+Scroll Lock): learn again
      const int hudCfg = (Cfg().hudSeparation ? 1 : 0) | (Cfg().hudSplit << 1);
      if (hudCfg != dd->lastHudCfg) {
        dd->lastHudCfg = hudCfg;
        dd->snapBad = false;
        dd->loggedSnap = false;
        dd->autoSplit = 1;
        dd->snapTried = dd->snapMatched = 0;
      }
      if (dd->snapBad && frame >= dd->snapBadUntil) dd->snapBad = false;  // try again now and then
      pc.snap = dd->snapAddr;
      // the copies of the last few frames: the shader uses the one that matches this image
      for (uint32_t k = 0; k < kSnapSlots; ++k) {
        const uint64_t f = dd->slotFrame[k].load();
        if (f && f + 6 >= frame + 1) pc.snapMask |= 1u << k;
      }
      pc.snapStride = uint32_t(dd->snapSlotBytes / 4);
      const bool use = Cfg().hudSeparation && dd->bbUsable && !dd->snapBad && pc.snapMask &&
                       dd->snapSlotBytes == VkDeviceSize(W) * H * 4;
      pc.snapFlags = (use ? 1u : 0u) | (dd->bbSwapRB ? 2u : 0u) | (dd->bb10 ? 4u : 0u);
      if (use && !dd->loggedSnap) {
        dd->loggedSnap = true;
        Log(1, "HUD separation: on (scene copied after back buffer draw %u, before the HUD; the HUD stays on the "
               "screen plane)", SplitDraw(dd));
      }
    }
    pc.W = W;
    pc.H = H;
    // the stages copied in this frame that have a known projection: the shader combines them
    uint32_t stages = copied, metaTail[12] = {};
    VkDeviceSize strideBytes;
    {
      std::lock_guard<std::mutex> l(dd->depthLock);
      strideBytes = dd->depthCap;
    }
    for (uint32_t g = 0; g < 4; ++g) {
      const DeviceData::Proj& pj = dd->proj[g];
      if (!pj.ok) stages &= ~(1u << g);
      memcpy(&metaTail[g], &pj.A, 4);
      memcpy(&metaTail[4 + g], &pj.B, 4);
    }
    metaTail[8] = stages;
    metaTail[9] = uint32_t(strideBytes / 4);
    // orientation: compare how many depth edges also are colour edges, as is and upside down
    if (dd->paramMap && frame >= dd->lastFlipCheck + 30) {
      dd->lastFlipCheck = frame;
      volatile uint32_t* st = (volatile uint32_t*)((char*)dd->paramMap + kStatsOffset);
      uint32_t eA = st[0], hA = st[1], eB = st[2], hB = st[3], sTot = st[4], sHud = st[5], matched = st[16], still = st[17];
      st[0] = st[1] = st[2] = st[3] = st[4] = st[5] = st[16] = st[17] = 0;
      static const bool snapDbg = getenv("NV3D_SNAPDBG") != nullptr;
      if (snapDbg)
        Log(1, "snapdbg still %u tried %u matched %u+%u n %u cur %u %u %u prev %u %u %u", still, dd->snapTried,
            dd->snapMatched, matched, st[11], st[8], st[9], st[10], st[12], st[13], st[14]);
      {
        std::lock_guard<std::mutex> l(dd->depthLock);
        dd->snapMatched += matched;
        dd->snapTried -= std::min(dd->snapTried, still);  // still images do not tell
        // HUD separation sanity: the "HUD" (what changed after the copy) must be a small part of the
        // image, and some copy must match the frames; else the copy was taken at the wrong point
        // (automatic: try after the next draw) or this game does not draw "scene, then HUD"
        const bool tooMuchHud = sTot >= 200 && sHud * 10 > sTot * 4;
        const bool noMatch = dd->snapTried >= 120 && dd->snapMatched * 5 < dd->snapTried;
        if (dd->snapTried >= 120 && !tooMuchHud && !noMatch) {
          if (dd->snapMatched * 10 < dd->snapTried * 9 || dd->traceFrames.load() > 0)
            Log(1, "HUD separation: a copy matched %u of %u frames (split after back buffer draw %u; %u copies taken)",
                dd->snapMatched, dd->snapTried, SplitDraw(dd), dd->snapTaken.exchange(0));
          dd->snapTried = dd->snapMatched = 0;
          dd->snapTaken = 0;
        }
        if ((tooMuchHud || noMatch) && !dd->snapBad) {
          const char* why = tooMuchHud ? "too much of the image changed after the copy" : "no copy matches the frames";
          if (Cfg().hudSplit == 0 && dd->autoSplit < 6) {
            ++dd->autoSplit;
            Log(1, "HUD separation: %s; now trying the copy after back buffer draw %u", why, dd->autoSplit);
          } else {
            Log(1, "HUD separation: off for now (%s: not a scene + HUD back buffer, or another split point is needed: "
                   "Ctrl+Shift+Scroll Lock)", why);
            dd->snapBad = true;
            dd->snapBadUntil = frame + 1800;
            dd->autoSplit = 1;
          }
          dd->loggedSnap = false;
          dd->snapTried = dd->snapMatched = 0;
        }
      }
      if (eA >= 200 && eB >= 200) {
        double rA = double(hA) / eA, rB = double(hB) / eB;
        int vote = rB > rA * 1.3 + 0.02 ? 1 : rA > rB * 1.3 + 0.02 ? 0 : -1;
        if (vote < 0 || uint32_t(vote) == dd->depthFlip) {
          dd->flipVotes = 0;
        } else if (++dd->flipVotes >= 3) {
          dd->depthFlip = uint32_t(vote);
          dd->flipVotes = 0;
          Log(1, "depth mode: the depth buffer is %s (edges matching the image: %.0f%% as is, %.0f%% flipped)",
              vote ? "upside down: flipped" : "upright", rA * 100, rB * 100);
        }
      }
    }
    metaTail[10] = dd->depthFlip;
    metaTail[11] = Cfg().depthNearest ? 1u : 0u;
    dd->d.CmdUpdateBuffer(cb, dd->metaBuf, 64, sizeof(metaTail), metaTail);
    bool on = stages != 0 && pc.depth && dd->depthCopies.load() > 0 && fresh;
    pc.a = 0.0f;
    pc.b = on ? 1.0f : 0.0f;  // 0: no depth this frame, both eyes = the game image
    pc.kL = EyeK(0);
    pc.kR = EyeK(1);
    if (dd->hybrid) {  // shifts relative to the eye the image was rendered for
      float kReal = EyeK(frame);
      pc.kL -= kReal;
      pc.kR -= kReal;
      pc.hyb = (sc->prevValid[otherEye] ? 1u : 0u) | (sc->prevValid[realEye] ? 2u : 0u) | (uint32_t(realEye) << 4);
    }
    pc.conv = Cfg().convergence;
    // (hybrid: from one eye to the other; objects very close to the camera need big shifts, and a
    // clamped shift would not match the real eye: they would tremble)
    pc.maxShift = float(W) * (dd->hybrid ? 0.5f : 0.1f);
    static const bool show = getenv("NV3D_DEPTH_SHOW") && getenv("NV3D_DEPTH_SHOW")[0] == '1';
    if (show) pc.maxShift = -1.0f;  // debug: show the captured depth
    const bool compute = !dd->hybrid || on || show;
    if (compute && (pc.snapFlags & 1u) && dd->paramBuf) {
      dd->d.CmdFillBuffer(cb, dd->paramBuf, kStatsOffset + 32, 32, 0);  // this frame's match counters
      std::lock_guard<std::mutex> l(dd->depthLock);
      ++dd->snapTried;
    } else {
      pc.snapFlags &= ~1u;
    }
    if (compute) {
      BufBarrier(dd, cb, XFER | ALL, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT, CS,
                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
      const uint32_t gx = (W + 15) / 16, gy = (H + 15) / 16;
      if (pc.snapFlags & 1u) {  // which copy belongs to this frame (one sample per 16x16 pixels)
        dd->d.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dd->synthPipe[2]);
        dd->d.CmdPushConstants(cb, dd->synthLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        dd->d.CmdDispatch(cb, (gx + 15) / 16, (gy + 15) / 16, 1);
        BufBarrier(dd, cb, CS, VK_ACCESS_SHADER_WRITE_BIT, CS, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
      }
      for (int pass = 0; pass < 2; ++pass) {
        dd->d.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, dd->synthPipe[pass]);
        dd->d.CmdPushConstants(cb, dd->synthLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        dd->d.CmdDispatch(cb, gx, gy, 1);
        BufBarrier(dd, cb, CS, VK_ACCESS_SHADER_WRITE_BIT, pass == 0 ? CS : XFER,
                   pass == 0 ? (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT) : VK_ACCESS_TRANSFER_READ_BIT);
      }
    } else {
      BufBarrier(dd, cb, XFER, VK_ACCESS_TRANSFER_WRITE_BIT, XFER,
                 VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    }
    // eye e from buffer offset 'off' through the synth image
    auto writeFromBuf = [&](int e, VkDeviceSize off) {
      Barrier(dd, cb, sc->synth[e], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, XFER,
              VK_ACCESS_TRANSFER_READ_BIT, XFER, VK_ACCESS_TRANSFER_WRITE_BIT);
      VkBufferImageCopy o{};
      o.bufferOffset = off;
      o.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      o.imageExtent = {W, H, 1};
      dd->d.CmdCopyBufferToImage(cb, sc->work, sc->synth[e], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &o);
      Barrier(dd, cb, sc->synth[e], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, XFER,
              VK_ACCESS_TRANSFER_WRITE_BIT, XFER, VK_ACCESS_TRANSFER_READ_BIT);
      WriteEye(dd, sc, cb, sc->synth[e], slot, e);
    };
    if (!dd->hybrid) {
      for (int e = 0; e < 2; ++e) writeFromBuf(e, px * 3 + px * VkDeviceSize(e));
    } else {
      // the real eye is the game image itself
      WriteEye(dd, sc, cb, vi.image, slot, realEye);
      if (on || show)
        writeFromBuf(otherEye, px * 3 + px * VkDeviceSize(otherEye));    // rebuilt from this frame's depth
      else if (sc->prevValid[otherEye])
        writeFromBuf(otherEye, px * 5 + px * VkDeviceSize(otherEye));    // no 3D scene: plain alternate
      else
        WriteEye(dd, sc, cb, vi.image, slot, otherEye);
      // keep this real image for the next frames (holes of this eye, menus)
      BufBarrier(dd, cb, XFER | CS, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, XFER,
                 VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
      VkBufferCopy keep{0, px * 5 + px * VkDeviceSize(realEye), px};
      dd->d.CmdCopyBuffer(cb, sc->work, sc->work, 1, &keep);
      sc->prevValid[realEye] = true;
    }
  } else {
    WriteEye(dd, sc, cb, vi.image, slot, 0);
    WriteEye(dd, sc, cb, vi.image, slot, 1);
  }
  if (cap.on) {
    CopyToBuf(dd, cb, vi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc->vext, cap.game);
    RecordCapture(dd, sc, cb, cap, slot);
  }
  Barrier(dd, cb, vi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, XFER, 0,
          VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
  // hybrid: the next frame is rendered for the other eye (stereo parameters ordered on the queue)
  if (dd->hybrid) {
    // flat_menus: screens without a 3D scene (title, menus, loading) are drawn without stereo shift
    // (games that draw their 2D menus with a perspective camera: they would sit far behind the screen)
    float kNext = EyeK(frame + 1);
    const uint64_t ld = dd->lastDepthFrame.load();
    if (Cfg().flatMenus && (ld == UINT64_MAX || ld + 10 < frame)) kNext = 0.0f;
    UpdateParams(dd, cb, kNext);
  }
  dd->d.EndCommandBuffer(cb);

  std::vector<VkPipelineStageFlags> wstages(waitCount, ALL);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.waitSemaphoreCount = waitCount;
  si.pWaitSemaphores = waits;
  si.pWaitDstStageMask = wstages.data();
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  QueueInfo* qi = GetQueueInfo(dd, queue);
  VkQueue subQueue = (qi->family == sc->poolFamily || !dd->gfxQueue) ? queue : dd->gfxQueue;
  {
    dd->stage = "present(depth): submitting";
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, subQueue)->m);
    VkResult sr = dd->d.QueueSubmit(subQueue, 1, &si, vi.fence);
    if (sr != VK_SUCCESS) return sr;
  }
  SignalWritten(dd, subQueue, sc, slot);
  Publish(sc, slot);
  CapturePair(dd, sc, subQueue, slot);
  FinishCapture(dd, sc, cap, vi.fence, frame);
  dd->frame = frame + 1;
  LogStats(dd, sc, false);
  return VK_SUCCESS;
}

// Replay mode: the game rendered this frame for the left eye. Copy it to the pair's left eye,
// switch the shaders to the right eye, submit the same command buffers again, copy the result
// to the right eye and publish the pair. Caller holds dd->recLock.
VkResult PresentReplay(DeviceData* dd, VkQueue queue, Swapchain* sc, uint32_t idx, const VkSemaphore* waits,
                       uint32_t waitCount) {
  QueueInfo* qi = GetQueueInfo(dd, queue);
  std::vector<VkCommandBuffer> cbs;
  cbs.swap(dd->recCbs);
  bool doReplay = !dd->recMono && !cbs.empty() && Cfg().stereo && queue == dd->gfxQueue;
  dd->recMono = false;
  const VkPipelineStageFlags ALL = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, XFER = VK_PIPELINE_STAGE_TRANSFER_BIT;
  std::vector<VkPipelineStageFlags> allStages(waitCount + 1, ALL);
  if (idx >= sc->v.size()) {
    std::lock_guard<std::mutex> l(qi->m);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = waitCount;
    si.pWaitSemaphores = waits;
    si.pWaitDstStageMask = allStages.data();
    dd->d.QueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    ReleaseHeldLocked(dd, queue);
    return VK_ERROR_OUT_OF_DATE_KHR;
  }
  VImage& vi = sc->v[idx];
  int slot = AcquireWriteSlot(dd, sc);
  dd->stage = "present(replay): waiting for previous copy";
  dd->d.WaitForFences(dd->device, 1, &vi.fence, VK_TRUE, UINT64_MAX);
  dd->d.ResetFences(dd->device, 1, &vi.fence);
  dd->stage = "present(replay): recording";
  if (Cfg().zoom != sc->lastZoom) ComputeRects(sc);
  Capture cap;
  BeginCapture(dd, sc, cap, dd->frame);
  HostBuf det;
  if (sc->detectOn && !Is32bpp(sc->format)) sc->detectOn = false;
  bool detect = sc->detectOn && dd->frame >= sc->nextDetect;
  if (detect) {
    sc->nextDetect = dd->frame + (sc->detections < 20 ? 30 : 240);
    detect = CreateHostBuf(dd, VkDeviceSize(sc->vext.width) * sc->vext.height * BytesPerPixel(sc->format), det);
  }
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  // A: left eye (+ right eye too if there is no replay this frame)
  VkCommandBuffer a = vi.cmd;
  dd->d.ResetCommandBuffer(a, 0);
  dd->d.BeginCommandBuffer(a, &bi);
  Barrier(dd, a, vi.image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ALL,
          VK_ACCESS_MEMORY_WRITE_BIT, XFER, VK_ACCESS_TRANSFER_READ_BIT);
  WriteEye(dd, sc, a, vi.image, slot, 0);
  if (!doReplay) WriteEye(dd, sc, a, vi.image, slot, 1);
  if (detect) CopyToBuf(dd, a, vi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc->vext, det);
  if (cap.on) CopyToBuf(dd, a, vi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc->vext, cap.game);
  if (cap.on && !doReplay) RecordCapture(dd, sc, a, cap, slot);
  Barrier(dd, a, vi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, XFER, 0, ALL, 0);
  if (doReplay) {
    UpdateParams(dd, a, EyeK(1));  // right eye
    FullBarrier(dd, a);
  }
  dd->d.EndCommandBuffer(a);

  // B: right eye, shaders back to the left eye
  VkCommandBuffer b = vi.cmd2;
  if (doReplay) {
    dd->d.ResetCommandBuffer(b, 0);
    dd->d.BeginCommandBuffer(b, &bi);
    FullBarrier(dd, b);
    Barrier(dd, b, vi.image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ALL,
            VK_ACCESS_MEMORY_WRITE_BIT, XFER, VK_ACCESS_TRANSFER_READ_BIT);
    WriteEye(dd, sc, b, vi.image, slot, 1);
    if (cap.on) RecordCapture(dd, sc, b, cap, slot);
    Barrier(dd, b, vi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, XFER, 0, ALL, 0);
    UpdateParams(dd, b, EyeK(0));  // left eye for the next frame
    dd->d.EndCommandBuffer(b);
  }

  VkSubmitInfo si[3] = {{VK_STRUCTURE_TYPE_SUBMIT_INFO}, {VK_STRUCTURE_TYPE_SUBMIT_INFO}, {VK_STRUCTURE_TYPE_SUBMIT_INFO}};
  si[0].waitSemaphoreCount = waitCount;
  si[0].pWaitSemaphores = waits;
  si[0].pWaitDstStageMask = allStages.data();
  si[0].commandBufferCount = 1;
  si[0].pCommandBuffers = &a;
  uint32_t nsi = 1;
  if (doReplay) {
    si[1].commandBufferCount = uint32_t(cbs.size());
    si[1].pCommandBuffers = cbs.data();
    si[2].commandBufferCount = 1;
    si[2].pCommandBuffers = &b;
    nsi = 3;
  }
  VkFence written = sc->slots[slot].written;
  dd->d.WaitForFences(dd->device, 1, &written, VK_TRUE, 2000000000ull);
  dd->d.ResetFences(dd->device, 1, &written);
  {
    dd->stage = "present(replay): submitting replay";
    std::lock_guard<std::mutex> l(qi->m);
    VkResult sr = dd->d.QueueSubmit(queue, nsi, si, vi.fence);
    VkSubmitInfo e{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    if (sr == VK_SUCCESS) sr = dd->d.QueueSubmit(queue, 1, &e, written);
    ReleaseHeldLocked(dd, queue);
    if (sr != VK_SUCCESS) return sr;
  }
  Publish(sc, slot);
  CapturePair(dd, sc, queue, slot);
  if (doReplay) ++dd->replayFrames;
  FinishCapture(dd, sc, cap, vi.fence, dd->frame);
  if (detect) {
    dd->d.WaitForFences(dd->device, 1, &vi.fence, VK_TRUE, UINT64_MAX);
    DetectBars(dd, sc, det);
    FreeHostBuf(dd, det);
  }
  ++dd->frame;
  LogStats(dd, sc, true);
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL hook_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pi) {
  DeviceData* dd = Dev(queue);
  std::vector<Swapchain*> scs(pi->swapchainCount, nullptr);
  bool any = false;
  for (uint32_t i = 0; i < pi->swapchainCount; ++i) any |= (scs[i] = FindSwapchain(dd, pi->pSwapchains[i])) != nullptr;
  if (!any) {
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, queue)->m);
    return dd->d.QueuePresentKHR(queue, pi);
  }
  PollControls();
  if (TakeRecordToggle())
    for (Swapchain* sc : scs)
      if (sc) { ToggleRecording(dd, sc); break; }
  // game sync: frames that complete a pair wait until the display took the previous pair
  dd->stage = "present: waiting for the display (game sync)";
  for (Swapchain* sc : scs)
    if (sc && !Cfg().skipRepeats && (dd->replay || dd->depthMode || (dd->frame & 1) || Cfg().inputLayout))
      WaitForDisplay(sc);

  VkResult overall = VK_SUCCESS;
  for (uint32_t i = 0; i < pi->swapchainCount; ++i) {
    const VkSemaphore* waits = i == 0 ? pi->pWaitSemaphores : nullptr;
    uint32_t wc = i == 0 ? pi->waitSemaphoreCount : 0;
    VkResult r;
    if (scs[i] && dd->replay) {
      std::lock_guard<std::mutex> rl(dd->recLock);
      r = PresentReplay(dd, queue, scs[i], pi->pImageIndices[i], waits, wc);
    } else if (scs[i] && dd->depthMode && !Cfg().inputLayout) {
      r = PresentDepth(dd, queue, scs[i], pi->pImageIndices[i], waits, wc);
    } else if (scs[i]) {
      r = PresentAlternate(dd, queue, scs[i], pi->pImageIndices[i], waits, wc);
    } else {
      VkPresentInfoKHR one{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
      one.waitSemaphoreCount = wc;
      one.pWaitSemaphores = waits;
      one.swapchainCount = 1;
      one.pSwapchains = &pi->pSwapchains[i];
      one.pImageIndices = &pi->pImageIndices[i];
      std::lock_guard<std::mutex> l(GetQueueInfo(dd, queue)->m);
      r = dd->d.QueuePresentKHR(queue, &one);
    }
    // errors of the presenter thread (window resized, surface lost) go back to the game
    if (scs[i] && r >= 0) {
      int pe = scs[i]->presError.load();
      if (pe == VK_ERROR_OUT_OF_DATE_KHR || pe == VK_ERROR_SURFACE_LOST_KHR || pe == VK_ERROR_DEVICE_LOST) r = VkResult(pe);
    }
    if (pi->pResults) pi->pResults[i] = r;
    overall = Worse(overall, r);
  }
  ++dd->presentCount;
  if (dd->warmupPending) {
    auto now = Clock::now();
    if (dd->firstPresent == Clock::time_point{}) dd->firstPresent = now;
    if ((dd->frame & 1) == 0 && dd->presentCount.load() >= 120 &&
        now - dd->firstPresent > std::chrono::duration<double>(Cfg().replayWarmup)) {
      std::lock_guard<std::mutex> rl(dd->recLock);
      dd->warmupPending = false;
      dd->recCbs.clear();
      dd->recMono = false;
      dd->frame = 0;
      dd->replaySince = now;
      dd->replay = true;
      Log(1, "warm-up over (%.0f s in alternate): replay mode, both eyes every game frame", Cfg().replayWarmup);
    }
  }
  dd->stage = "game is rendering (between presents)";
  if (dd->out && dd->presentCount % 120 == 1) RaiseOutputWindow(dd->out);
  // optional extra fps cap (game frames per second)
  float lim = Cfg().fpsLimit;
  if (lim > 1.0f) {
    auto now = Clock::now();
    auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / lim));
    if (dd->lastPresent != Clock::time_point{} && now < dd->lastPresent + period) {
      std::this_thread::sleep_until(dd->lastPresent + period);
      dd->lastPresent += period;
    } else {
      dd->lastPresent = now;
    }
  }
  return overall;
}

// ------------------------------------------------------------------------------------
// Proc addr
// ------------------------------------------------------------------------------------
#define DEVICE_HOOKS(X)                                                                          \
  X(DestroyDevice) X(GetDeviceQueue) X(GetDeviceQueue2) X(QueueSubmit) X(QueueSubmit2)          \
  X(QueueSubmit2KHR) X(QueueWaitIdle) X(QueueBindSparse) X(DeviceWaitIdle) X(CreateShaderModule) \
  X(DestroyShaderModule) X(CreateGraphicsPipelines) X(CreateSwapchainKHR) X(DestroySwapchainKHR) \
  X(GetSwapchainImagesKHR) X(AcquireNextImageKHR) X(AcquireNextImage2KHR) X(QueuePresentKHR)
#define REPLAY_HOOKS(X) \
  X(WaitForFences) X(WaitSemaphores) X(WaitSemaphoresKHR) X(CreateSemaphore) X(DestroySemaphore) X(BeginCommandBuffer)
#define DEPTH_HOOKS(X)                                                                            \
  X(CreateImage) X(DestroyImage) X(CreateImageView) X(DestroyImageView) X(CmdBeginRendering)     \
  X(CmdBeginRenderingKHR) X(CmdEndRendering) X(CmdEndRenderingKHR) X(CmdBeginRenderPass)         \
  X(CmdBeginRenderPass2) X(CmdDraw) X(CmdDrawIndexed) X(CmdDrawIndirect) X(CmdDrawIndexedIndirect) \
  X(CmdDrawIndirectCount) X(CmdDrawIndexedIndirectCount) X(CmdSetViewport) X(CmdSetViewportWithCount)   \
  X(CmdSetViewportWithCountEXT) X(BeginCommandBuffer) X(EndCommandBuffer) X(CmdPipelineBarrier)       \
  X(CmdPipelineBarrier2) X(CmdPipelineBarrier2KHR) X(CmdCopyImage) X(CmdBlitImage) X(CmdCopyImage2)  \
  X(CmdCopyImage2KHR) X(CmdBlitImage2) X(CmdBlitImage2KHR) X(CmdResolveImage) X(CmdResolveImage2)    \
  X(CmdResolveImage2KHR) X(CmdExecuteCommands)

PFN_vkVoidFunction DeviceHook(const char* name) {
#define CHECK(n) if (!strcmp(name, "vk" #n)) return (PFN_vkVoidFunction)&hook_##n;
  if (Cfg().passthrough) return nullptr;  // launchers / helpers: no hooks at all
  if (Cfg().mode == 1) {                 // replay mode needs to see waits and command buffer setup
    REPLAY_HOOKS(CHECK)
  }
  if (Cfg().mode == 2 || Cfg().mode == 3) {  // depth / hybrid: finds and copies the scene's depth buffer
    DEPTH_HOOKS(CHECK)
  }
  if (Cfg().dumpShaders || Cfg().fixShadows) {  // shader dump / screen-space fixes see the compute shaders
    CHECK(CreateComputePipelines)
  }
  DEVICE_HOOKS(CHECK)
#undef CHECK
  return nullptr;
}

}  // namespace

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL nv3d_GetDeviceProcAddr(VkDevice device, const char* name);

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL nv3d_GetInstanceProcAddr(VkInstance instance,
                                                                             const char* name) {
  if (!strcmp(name, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)&nv3d_GetInstanceProcAddr;
  if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)&nv3d_GetDeviceProcAddr;
  if (!strcmp(name, "vkCreateInstance")) return (PFN_vkVoidFunction)&hook_CreateInstance;
  if (!strcmp(name, "vkCreateDevice")) return (PFN_vkVoidFunction)&hook_CreateDevice;
  if (!instance) return nullptr;
  InstanceData* id = Inst(instance);
  if (!id) return nullptr;
  PFN_vkVoidFunction next = id->gipa(instance, name);
  if (!next) return nullptr;
#define CHECK(n) if (!strcmp(name, "vk" #n)) return (PFN_vkVoidFunction)&hook_##n;
  CHECK(DestroyInstance)
  CHECK(EnumeratePhysicalDevices)
  CHECK(EnumeratePhysicalDeviceGroups)
  CHECK(EnumeratePhysicalDeviceGroupsKHR)
  CHECK(EnumerateDeviceExtensionProperties)
#undef CHECK
  if (PFN_vkVoidFunction h = DeviceHook(name)) return h;
  return next;
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL nv3d_GetDeviceProcAddr(VkDevice device, const char* name) {
  if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)&nv3d_GetDeviceProcAddr;
  DeviceData* dd = Dev(device);
  if (!dd) return nullptr;
  PFN_vkVoidFunction next = dd->d.GetDeviceProcAddr(device, name);
  if (!next) return nullptr;
  if (PFN_vkVoidFunction h = DeviceHook(name)) return h;
  return next;
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL nv3d_NegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* p) {
  if (!p || p->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) return VK_ERROR_INITIALIZATION_FAILED;
  if (p->loaderLayerInterfaceVersion >= 2) {
    p->pfnGetInstanceProcAddr = nv3d_GetInstanceProcAddr;
    p->pfnGetDeviceProcAddr = nv3d_GetDeviceProcAddr;
    p->pfnGetPhysicalDeviceProcAddr = nullptr;
    p->loaderLayerInterfaceVersion = 2;
  }
  return VK_SUCCESS;
}
