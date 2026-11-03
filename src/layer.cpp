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
#include <dirent.h>
#include <pthread.h>
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
  X(ResetCommandBuffer) X(CmdPipelineBarrier) X(CmdCopyImage) X(CmdBlitImage)               \
  X(CmdUpdateBuffer) X(CmdClearColorImage) X(CmdCopyImageToBuffer) X(CreateFence) X(DestroyFence) X(WaitForFences)  \
  X(ResetFences) X(CreateSemaphore) X(DestroySemaphore) X(WaitSemaphores) X(WaitSemaphoresKHR)  \
  X(GetFenceStatus) X(WaitForPresentKHR)

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
  uint64_t pairsPublished = 0, pairsShown = 0;

  // ---- presenter thread
  VkCommandPool presPool = VK_NULL_HANDLE;
  PresEntry ring[kRing];
  std::thread presThread;
  std::atomic<bool> stop{false};
  std::atomic<int> presError{VK_SUCCESS};  // error to report to the game (out of date, lost...)
  std::atomic<uint64_t> refreshes{0};      // frames presented by the presenter
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

  OutputWindow* out = nullptr;     // own window on the glasses monitor
  VkSurfaceKHR outSurface = VK_NULL_HANDLE;
  int outState = 0;                // 0 undecided, 1 using own window, -1 using the game window

  bool stereoShaders = false;
  VkBuffer paramBuf = VK_NULL_HANDLE;
  VkDeviceMemory paramMem = VK_NULL_HANDLE;
  VkDeviceAddress paramAddr = 0;

  std::mutex shaderLock;
  std::unordered_map<VkShaderModule, ModuleEntry> modules;
  std::atomic<uint32_t> patchedCount{0}, skippedCount{0}, pipelineCount{0};

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
  std::atomic<uint64_t> presentCount{0};
  std::atomic<int> swapCount{0};
  std::atomic<uint64_t> lastAbortPresent{0};
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
bool IsHidden(const char* name) {
  for (const char* h : kAlwaysHidden)
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
          Log(0, "  held: %zu fences, %zu timeline semaphores, %zu command buffers -> releasing",
              dd->heldFences.size(), dd->heldTimeline.size(), dd->recCbs.size());
          AbortReplayLocked(dd, "watchdog");
          if (level == 2) {
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
// Device creation
// ------------------------------------------------------------------------------------
bool CreateParamBuffer(DeviceData* dd) {
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = 256;
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
    float init[4] = {EyeK(0), Cfg().convergence, EyeK(0) == 0 ? 0.0f : HudFactor(), 0};
    memcpy(map, init, sizeof(init));
    dd->d.UnmapMemory(dd->device, dd->paramMem);
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
  bool supported = false;
  auto gpf2 = inst->d.GetPhysicalDeviceFeatures2 ? inst->d.GetPhysicalDeviceFeatures2
                                                 : inst->d.GetPhysicalDeviceFeatures2KHR;
  if (gpf2) {
    VkPhysicalDeviceBufferDeviceAddressFeatures bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &bda};
    gpf2(phys, &f2);
    supported = bda.bufferDeviceAddress && f2.features.shaderInt64;
  }
  // Proton (DXVK / VKD3D-Proton) always uses Vulkan 1.3. For 1.0 apps BDA would need extra
  // instance extensions, so only Vulkan >= 1.1 is handled (NV3D_LEGACY=1 forces it).
  const char* legacy = getenv("NV3D_LEGACY");
  bool apiOk = api >= VK_API_VERSION_1_1 || (legacy && legacy[0] == '1');
  bool wantStereo = Cfg().patchShaders && supported && apiOk;

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
          haveI64 = true;
          break;
        default: break;
      }
    }
    if (!haveI64) {
      if (ci.pEnabledFeatures) featCopy = *ci.pEnabledFeatures;
      featCopy.shaderInt64 = VK_TRUE;
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
    dd->stereoShaders = CreateParamBuffer(dd);
    dd->replay = dd->stereoShaders && Cfg().mode == 1;
    if (!dd->stereoShaders) Log(0, "failed to create stereo parameter buffer");
  }
  Log(1, "device '%s' api %u.%u: shader stereo %s, mode %s, param @0x%llx", props.deviceName,
      VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api), dd->stereoShaders ? "ON" : "OFF",
      dd->replay ? "replay (both eyes every frame)" : "alternate (L/R frames)", (unsigned long long)dd->paramAddr);

  if (!Cfg().passthrough) dd->watchdog = std::thread(WatchdogMain, dd);
  std::lock_guard<std::mutex> l(g_lock);
  g_devByKey[Key(*pDevice)] = dd;
  return VK_SUCCESS;
}

struct Swapchain;
void StopPresenter(DeviceData* dd, Swapchain* sc);

VKAPI_ATTR void VKAPI_CALL hook_DestroyDevice(VkDevice device, const VkAllocationCallbacks* a) {
  DeviceData* dd = Dev(device);
  if (!dd) return;
  dd->stopWatchdog = true;
  if (dd->watchdog.joinable()) dd->watchdog.join();
  {  // swapchains the app did not destroy: stop their presenter threads
    std::lock_guard<std::mutex> l(dd->swapLock);
    for (auto& kv : dd->swapchains) StopPresenter(dd, kv.second);
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
  if (!dd->replay) return dd->d.BeginCommandBuffer(cb, bi);
  VkCommandBufferBeginInfo b = *bi;  // the frame is submitted twice
  b.flags &= ~VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  b.flags |= VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  return dd->d.BeginCommandBuffer(cb, &b);
}

// ------------------------------------------------------------------------------------
// Shader patching
// ------------------------------------------------------------------------------------
VKAPI_ATTR VkResult VKAPI_CALL hook_CreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* ci,
                                                       const VkAllocationCallbacks* a, VkShaderModule* m) {
  DeviceData* dd = Dev(device);
  VkResult r = dd->d.CreateShaderModule(device, ci, a, m);
  if (r == VK_SUCCESS && dd->stereoShaders && ci->codeSize % 4 == 0 &&
      HasPreRasterEntryPoint(ci->pCode, ci->codeSize / 4)) {
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
  PatchResult pr = PatchStereo(code, words, entry, model, dd->paramAddr, out, &log);
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
  if (!dd->stereoShaders) return dd->d.CreateGraphicsPipelines(device, cache, count, infos, a, out);

  std::vector<VkGraphicsPipelineCreateInfo> ci(infos, infos + count);
  std::vector<std::vector<VkPipelineShaderStageCreateInfo>> stages(count);
  struct Restore { VkShaderModuleCreateInfo* smci; const uint32_t* code; size_t size; };
  std::vector<Restore> restores;
  std::vector<std::unique_ptr<std::vector<uint32_t>>> inlineCode;

  for (uint32_t i = 0; i < count; ++i) {
    if (!ci[i].pStages || !ci[i].stageCount) continue;
    int best = -1, bestRank = -1;
    for (uint32_t s = 0; s < ci[i].stageCount; ++s) {
      int rank = -1;
      switch (ci[i].pStages[s].stage) {
        case VK_SHADER_STAGE_VERTEX_BIT: rank = 0; break;
        case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: rank = 1; break;
        case VK_SHADER_STAGE_GEOMETRY_BIT: rank = 2; break;
        default: break;
      }
      if (rank > bestRank) { bestRank = rank; best = int(s); }
    }
    if (best < 0) continue;
    ++dd->pipelineCount;
    const VkPipelineShaderStageCreateInfo& st = ci[i].pStages[best];
    uint32_t model = StageToModel(st.stage);
    if (st.module != VK_NULL_HANDLE) {
      VkShaderModule pm = GetPatchedModule(dd, st.module, model, st.pName);
      if (pm) {
        stages[i].assign(ci[i].pStages, ci[i].pStages + ci[i].stageCount);
        stages[i][best].module = pm;
        ci[i].pStages = stages[i].data();
      }
    } else {
      // VK_KHR_maintenance5: shader code chained inline
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
    }
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
  float data[3] = {k, Cfg().convergence, HudFactor()};
  VkMemoryBarrier pre{VK_STRUCTURE_TYPE_MEMORY_BARRIER};  // after earlier reads and the previous update
  pre.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  pre.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  dd->d.CmdPipelineBarrier(cb, ALL, XFER, 0, 1, &pre, 0, nullptr, 0, nullptr);
  dd->d.CmdUpdateBuffer(cb, dd->paramBuf, 0, sizeof(data), data);
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
  if (sc) {
    --dd->swapCount;
    StopPresenter(dd, sc);
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
  static uint64_t lastPublished = 0;
  double pairs = (published - lastPublished) / el;
  lastPublished = published;
  if (replayStats) {
    uint32_t rf = dd->replayFrames.exchange(0), mf = dd->monoFrames.exchange(0);
    // the game needs GPU results in the middle of its frames: replay cannot work, use alternate
    if (Cfg().autoSwitch && mf > rf && rf + mf >= 30) {
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
  } else {
    Log(1, "game %.1f fps -> %.1f stereo pairs/s [alternate] | shaders patched %u, skipped %u | sep %.4f conv %.3f%s",
        sc->statFrames / el, pairs, dd->patchedCount.load(), dd->skippedCount.load(), Cfg().separation,
        Cfg().convergence, Cfg().stereo ? "" : " [2D]");
  }
  sc->statFrames = 0;
  sc->statStart = now;
}

// Alternate mode: even frames = left eye, odd frames = right eye of the same pair.
VkResult PresentAlternate(DeviceData* dd, VkQueue queue, Swapchain* sc, uint32_t idx, const VkSemaphore* waits,
                          uint32_t waitCount) {
  if (idx >= sc->v.size()) return VK_ERROR_OUT_OF_DATE_KHR;
  VImage& vi = sc->v[idx];
  const uint64_t frame = dd->frame;
  const bool stereoInput = Cfg().inputLayout != 0;  // every frame is a whole pair (SBS / top-bottom video)
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
  // game sync: frames that complete a pair wait until the display took the previous pair
  dd->stage = "present: waiting for the display (game sync)";
  for (Swapchain* sc : scs)
    if (sc && (dd->replay || (dd->frame & 1) || Cfg().inputLayout)) WaitForDisplay(sc);

  VkResult overall = VK_SUCCESS;
  for (uint32_t i = 0; i < pi->swapchainCount; ++i) {
    const VkSemaphore* waits = i == 0 ? pi->pWaitSemaphores : nullptr;
    uint32_t wc = i == 0 ? pi->waitSemaphoreCount : 0;
    VkResult r;
    if (scs[i] && dd->replay) {
      std::lock_guard<std::mutex> rl(dd->recLock);
      r = PresentReplay(dd, queue, scs[i], pi->pImageIndices[i], waits, wc);
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

PFN_vkVoidFunction DeviceHook(const char* name) {
#define CHECK(n) if (!strcmp(name, "vk" #n)) return (PFN_vkVoidFunction)&hook_##n;
  if (Cfg().passthrough) return nullptr;  // launchers / helpers: no hooks at all
  if (Cfg().mode == 1) {                 // replay mode needs to see waits and command buffer setup
    REPLAY_HOOKS(CHECK)
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
