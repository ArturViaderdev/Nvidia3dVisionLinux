// ------------------------------------------------------------------------------------
// Live streaming (Twitch / YouTube, both at once) through gamestreamd, the daemon of the
// gamecapture project: the 3D picture goes out as one side-by-side frame (half SBS by default:
// 1920x1080 with each eye squeezed to 960x1080), which viewers can watch flat or in 3D.
//
// The layer talks to gamestreamd as gamecapture's own layer does: exportable VRAM buffers are
// handed over once (file descriptors over a UNIX socket) and, for every stereo pair streamed, the
// two eyes are blitted side by side into one of them on the GPU, in the same queue as the pair,
// after it. The daemon imports that memory in CUDA and encodes it with NVENC: no CPU copies, the
// game never waits (when the daemon is behind, pairs are skipped). Audio, the stream keys, the
// bitrates and the destinations are gamestreamd's (~/.config/gamestream/gamestream.conf).
//
// Start / stop: Ctrl+Shift+B in the game, or `gamestreamd toggle` from the desktop.
// Included by layer.cpp (it needs DeviceData, Swapchain, Barrier, Log...).
// ------------------------------------------------------------------------------------
// (system headers and gamestream_protocol.h: included at the top of layer.cpp)

enum { kStreamFree = 0, kStreamGpu = 1, kStreamDaemon = 2 };

// gamestreamd can ask for up to GS_MAX_STREAMS sizes at once (one per destination resolution,
// e.g. 4K for YouTube and 1080p for Twitch): every slot has one buffer per size.
struct StreamSlot {
  VkBuffer buf[GS_MAX_STREAMS] = {};
  VkDeviceMemory mem[GS_MAX_STREAMS] = {};
  VkCommandBuffer cb = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  std::atomic<int> state{kStreamFree};
  uint64_t ts = 0;
};

struct Streamer {
  DeviceData* dd = nullptr;
  std::thread io, waiter;
  std::atomic<bool> quit{false};

  std::mutex sockM;
  int sock = -1;
  std::atomic<bool> connected{false};

  std::atomic<bool> want{false};
  std::mutex reqM;
  uint32_t reqN = 1, reqW[GS_MAX_STREAMS] = {}, reqH[GS_MAX_STREAMS] = {};
  std::atomic<uint32_t> reqFps{60}, session{0};

  std::mutex resM;  // the resources below
  bool valid = false;
  uint32_t resSession = 0, family = UINT32_MAX;
  VkCommandPool pool = VK_NULL_HANDLE;
  uint32_t nstreams = 1, fps = 60, nslots = 4;
  uint32_t outW[GS_MAX_STREAMS] = {}, outH[GS_MAX_STREAMS] = {}, pitch[GS_MAX_STREAMS] = {};
  VkDeviceSize allocSize[GS_MAX_STREAMS] = {};
  StreamSlot slots[GS_MAX_SLOTS];
  std::atomic<uint32_t> gen{0};
  VkImage interm[GS_MAX_STREAMS] = {};  // the side-by-side frames, before the copy to the buffers
  VkDeviceMemory intermMem[GS_MAX_STREAMS] = {};
  VkFormat intermFmt = VK_FORMAT_UNDEFINED;
  uint64_t nextDue = 0, sent = 0, dropped = 0;

  std::mutex fqM;
  std::condition_variable fcv;
  std::deque<uint32_t> fq;
};

static uint64_t StreamNowNs() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

static bool StreamSend(Streamer* s, const void* data, size_t len, const int* fds = nullptr, int nfds = 0) {
  std::lock_guard<std::mutex> l(s->sockM);
  if (s->sock < 0) return false;
  iovec iov{const_cast<void*>(data), len};
  msghdr mh{};
  mh.msg_iov = &iov;
  mh.msg_iovlen = 1;
  char cbuf[CMSG_SPACE(sizeof(int) * GS_MAX_SLOTS * GS_MAX_STREAMS)];
  if (nfds > 0) {
    memset(cbuf, 0, sizeof cbuf);
    mh.msg_control = cbuf;
    mh.msg_controllen = CMSG_SPACE(sizeof(int) * nfds);
    cmsghdr* c = CMSG_FIRSTHDR(&mh);
    c->cmsg_level = SOL_SOCKET;
    c->cmsg_type = SCM_RIGHTS;
    c->cmsg_len = CMSG_LEN(sizeof(int) * nfds);
    memcpy(CMSG_DATA(c), fds, sizeof(int) * nfds);
  }
  return sendmsg(s->sock, &mh, MSG_NOSIGNAL) == ssize_t(len);
}

static bool StreamConnect(Streamer* s) {
  int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  if (fd < 0) return false;
  sockaddr_un a{};
  a.sun_family = AF_UNIX;
  size_t n = strlen(GS_SOCKET_NAME);
  memcpy(a.sun_path + 1, GS_SOCKET_NAME, n);  // abstract namespace
  if (connect(fd, (sockaddr*)&a, socklen_t(offsetof(sockaddr_un, sun_path) + 1 + n)) != 0) {
    close(fd);
    return false;
  }
  {
    std::lock_guard<std::mutex> l(s->sockM);
    s->sock = fd;
  }
  GsHello h{};
  h.type = GS_HELLO;
  h.version = GS_PROTO_VERSION;
  h.pid = getpid();
  h.role = GS_ROLE_GAME;
  snprintf(h.exe, sizeof h.exe, "%s (3D)", GameName().c_str());
  StreamSend(s, &h, sizeof h);
  s->connected = true;
  Log(1, "live streaming: connected to gamestreamd (Ctrl+Shift+B or 'gamestreamd toggle' starts / stops)");
  return true;
}

static void StreamDisconnect(Streamer* s) {
  {
    std::lock_guard<std::mutex> l(s->sockM);
    if (s->sock >= 0) close(s->sock);
    s->sock = -1;
  }
  s->connected = false;
  s->want = false;
  std::lock_guard<std::mutex> l(s->resM);
  for (auto& sl : s->slots) {
    int e = kStreamDaemon;
    sl.state.compare_exchange_strong(e, kStreamFree);
  }
}

static void StreamHandle(Streamer* s, const char* buf, ssize_t n) {
  if (n < 4) return;
  uint32_t type;
  memcpy(&type, buf, 4);
  if (type == GS_START && n >= ssize_t(sizeof(GsStart))) {
    GsStart st;
    memcpy(&st, buf, sizeof st);
    if (st.version != GS_PROTO_VERSION) {
      Log(0, "live streaming: gamestreamd speaks protocol %u, this layer %u: rebuild 3dvisionlinux with the "
             "gamestream_protocol.h of the installed gamecapture", st.version, GS_PROTO_VERSION);
      return;
    }
    std::string sizes;
    {
      std::lock_guard<std::mutex> l(s->reqM);
      s->reqN = std::min<uint32_t>(std::max<uint32_t>(st.nstreams, 1), GS_MAX_STREAMS);
      for (uint32_t k = 0; k < s->reqN; ++k) {
        s->reqW[k] = st.width[k];
        s->reqH[k] = st.height[k];
        char b[32];
        snprintf(b, sizeof b, "%s%ux%u", k ? " + " : "", st.width[k], st.height[k]);
        sizes += b;
      }
    }
    s->reqFps = st.fps ? st.fps : 60;
    ++s->session;
    s->want = true;
    Log(1, "live streaming: start (%s %s @ %u fps)", sizes.c_str(), Cfg().streamFull ? "full SBS" : "half SBS",
        s->reqFps.load());
  } else if (type == GS_STOP) {
    s->want = false;
    Log(1, "live streaming: stop");
  } else if (type == GS_RELEASE && n >= ssize_t(sizeof(GsRelease))) {
    GsRelease r;
    memcpy(&r, buf, sizeof r);
    std::lock_guard<std::mutex> l(s->resM);
    if (r.generation == s->gen.load() && r.slot < s->nslots) {
      int e = kStreamDaemon;
      s->slots[r.slot].state.compare_exchange_strong(e, kStreamFree);
    }
  }
}

static void StreamIoLoop(Streamer* s) {
  uint64_t lastTry = 0;
  while (!s->quit) {
    if (TakeStreamToggle()) {
      GsSimple t{GS_TOGGLE};
      if (!StreamSend(s, &t, sizeof t))
        Log(0, "live streaming: gamestreamd is not running (start it: systemctl --user start gamestreamd)");
    }
    int fd;
    {
      std::lock_guard<std::mutex> l(s->sockM);
      fd = s->sock;
    }
    if (fd < 0) {
      uint64_t now = StreamNowNs();
      if (now - lastTry > 2000000000ull) {
        lastTry = now;
        StreamConnect(s);
      }
      usleep(40000);
      continue;
    }
    pollfd p{fd, POLLIN, 0};
    if (::poll(&p, 1, 40) > 0) {
      if (p.revents & POLLIN) {
        char buf[GS_MAX_MSG];
        ssize_t n = recv(fd, buf, sizeof buf, 0);
        if (n <= 0) {
          StreamDisconnect(s);
          Log(1, "live streaming: gamestreamd closed the connection");
          continue;
        }
        StreamHandle(s, buf, n);
      } else if (p.revents & (POLLHUP | POLLERR)) {
        StreamDisconnect(s);
      }
    }
  }
}

// waits for each copy on its own thread (never the game's) and tells the daemon
static void StreamWaitLoop(Streamer* s) {
  DeviceData* dd = s->dd;
  for (;;) {
    uint32_t i;
    {
      std::unique_lock<std::mutex> l(s->fqM);
      s->fcv.wait(l, [&] { return s->quit.load() || !s->fq.empty(); });
      if (s->fq.empty()) return;
      i = s->fq.front();
      s->fq.pop_front();
    }
    StreamSlot& sl = s->slots[i];
    VkResult r;
    do r = dd->d.WaitForFences(dd->device, 1, &sl.fence, VK_TRUE, 500000000ull);
    while (r == VK_TIMEOUT && !s->quit);
    bool ok = false;
    if (r == VK_SUCCESS && s->want && !s->quit) {
      GsFrame f{GS_FRAME, s->gen.load(), i, 0, sl.ts};
      sl.state = kStreamDaemon;
      ok = StreamSend(s, &f, sizeof f);
      if (ok) ++s->sent;
    }
    if (!ok) sl.state = kStreamFree;
  }
}

static int StreamMemType(DeviceData* dd, uint32_t bits) {
  for (uint32_t i = 0; i < dd->memProps.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (dd->memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) return int(i);
  for (uint32_t i = 0; i < dd->memProps.memoryTypeCount; ++i)
    if (bits & (1u << i)) return int(i);
  return -1;
}

// caller holds resM; only when no copy is in flight
static void StreamDestroyLocked(Streamer* s) {
  DeviceData* dd = s->dd;
  for (auto& sl : s->slots) {
    for (uint32_t k = 0; k < GS_MAX_STREAMS; ++k) {
      if (sl.buf[k]) dd->d.DestroyBuffer(dd->device, sl.buf[k], nullptr);
      if (sl.mem[k]) dd->d.FreeMemory(dd->device, sl.mem[k], nullptr);  // the daemon keeps its own reference
      sl.buf[k] = VK_NULL_HANDLE;
      sl.mem[k] = VK_NULL_HANDLE;
    }
    if (sl.fence) dd->d.DestroyFence(dd->device, sl.fence, nullptr);
    if (sl.cb) dd->d.FreeCommandBuffers(dd->device, s->pool, 1, &sl.cb);
    sl.fence = VK_NULL_HANDLE;
    sl.cb = VK_NULL_HANDLE;
    sl.state = kStreamFree;
  }
  if (s->pool) dd->d.DestroyCommandPool(dd->device, s->pool, nullptr);
  s->pool = VK_NULL_HANDLE;
  for (uint32_t k = 0; k < GS_MAX_STREAMS; ++k) {
    if (s->interm[k]) dd->d.DestroyImage(dd->device, s->interm[k], nullptr);
    if (s->intermMem[k]) dd->d.FreeMemory(dd->device, s->intermMem[k], nullptr);
    s->interm[k] = VK_NULL_HANDLE;
    s->intermMem[k] = VK_NULL_HANDLE;
  }
  s->valid = false;
}

static bool StreamTryDestroy(Streamer* s) {
  for (uint32_t i = 0; i < s->nslots; ++i)
    if (s->slots[i].state.load() == kStreamGpu) return false;
  std::lock_guard<std::mutex> l(s->resM);
  if (s->valid && s->sent)
    Log(1, "live streaming: %llu pairs sent, %llu skipped (the encoder was behind)", (unsigned long long)s->sent,
        (unsigned long long)s->dropped);
  StreamDestroyLocked(s);
  s->sent = s->dropped = 0;
  return true;
}

static bool IsSrgbFormat(VkFormat f) {
  return f == VK_FORMAT_B8G8R8A8_SRGB || f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_A8B8G8R8_SRGB_PACK32;
}

// the buffers the daemon encodes from, sent to it once per session
static bool StreamCreate(Streamer* s, uint32_t fam, VkExtent2D eye, VkFormat eyeFmt) {
  DeviceData* dd = s->dd;
  std::lock_guard<std::mutex> l(s->resM);
  // each resolution of the daemon (gamestream.conf, 1920x1080 by default) is the whole
  // side-by-side frame (half SBS); full SBS: each eye has it; "auto" (0x0): each eye at the
  // monitor's size
  {
    std::lock_guard<std::mutex> lr(s->reqM);
    s->nstreams = s->reqN;
    for (uint32_t k = 0; k < s->nstreams; ++k) {
      uint32_t w = s->reqW[k], h = s->reqH[k];
      if (!w || !h) {
        w = Cfg().streamFull ? eye.width * 2 : eye.width;
        h = eye.height;
      } else if (Cfg().streamFull) {
        w *= 2;
      }
      s->outW[k] = w & ~3u;  // even halves, even size for NVENC
      s->outH[k] = h & ~1u;
      s->pitch[k] = (s->outW[k] * 4 + 255) & ~255u;
    }
  }
  s->fps = s->reqFps;
  s->nslots = 4;
  s->family = fam;
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = fam;
  if (dd->d.CreateCommandPool(dd->device, &pci, nullptr, &s->pool) != VK_SUCCESS) {
    StreamDestroyLocked(s);
    return false;
  }
  // the side-by-side pictures (sRGB like the eyes, so a blit does not change the colours)
  bool ok = true;
  for (uint32_t k = 0; k < s->nstreams && ok; ++k) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = s->intermFmt = IsSrgbFormat(eyeFmt) ? VK_FORMAT_B8G8R8A8_SRGB : VK_FORMAT_B8G8R8A8_UNORM;
    ici.extent = {s->outW[k], s->outH[k], 1};
    ici.mipLevels = ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ok = CreateImageMem(dd, ici, &s->interm[k], &s->intermMem[k]);
  }
  std::vector<int> fds;
  for (uint32_t i = 0; i < s->nslots && ok; ++i) {
    StreamSlot& sl = s->slots[i];
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = s->pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    ok = dd->d.AllocateCommandBuffers(dd->device, &ai, &sl.cb) == VK_SUCCESS;
    if (!ok) break;
    if (dd->setLoaderData) dd->setLoaderData(dd->device, sl.cb);
    else *(void**)sl.cb = *(void**)dd->device;
    // fds go in the order fd[slot * nstreams + stream]
    for (uint32_t k = 0; k < s->nstreams && ok; ++k) {
      VkExternalMemoryBufferCreateInfo ebi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
      ebi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
      VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &ebi};
      bci.size = VkDeviceSize(s->pitch[k]) * s->outH[k];
      bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      ok = dd->d.CreateBuffer(dd->device, &bci, nullptr, &sl.buf[k]) == VK_SUCCESS;
      if (!ok) break;
      VkMemoryRequirements mr;
      dd->d.GetBufferMemoryRequirements(dd->device, sl.buf[k], &mr);
      VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
      ded.buffer = sl.buf[k];
      VkExportMemoryAllocateInfo exp{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
                                     dd->streamDedicated ? (void*)&ded : nullptr};
      exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
      VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &exp};
      mai.allocationSize = mr.size;
      int mt = StreamMemType(dd, mr.memoryTypeBits);
      mai.memoryTypeIndex = uint32_t(mt);
      ok = mt >= 0 && dd->d.AllocateMemory(dd->device, &mai, nullptr, &sl.mem[k]) == VK_SUCCESS &&
           dd->d.BindBufferMemory(dd->device, sl.buf[k], sl.mem[k], 0) == VK_SUCCESS;
      if (!ok) break;
      s->allocSize[k] = mr.size;
      VkMemoryGetFdInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
      gi.memory = sl.mem[k];
      gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
      int fd = -1;
      ok = dd->getMemoryFd(dd->device, &gi, &fd) == VK_SUCCESS && fd >= 0;
      if (ok) fds.push_back(fd);
    }
    if (!ok) break;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    ok = dd->d.CreateFence(dd->device, &fci, nullptr, &sl.fence) == VK_SUCCESS;
    if (!ok) break;
    sl.state = kStreamFree;
  }
  if (ok) {
    GsTextures t{};
    t.type = GS_TEXTURES;
    t.version = GS_PROTO_VERSION;
    t.generation = ++s->gen;
    t.nslots = s->nslots;
    t.nstreams = s->nstreams;
    for (uint32_t k = 0; k < s->nstreams; ++k) {
      t.width[k] = s->outW[k];
      t.height[k] = s->outH[k];
      t.pitch[k] = s->pitch[k];
      t.alloc_size[k] = s->allocSize[k];
    }
    t.dedicated = dd->streamDedicated;
    t.has_uuid = dd->streamHasUuid;
    memcpy(t.device_uuid, dd->streamUuid, 16);
    t.pixfmt = 0;  // BGRA8
    t.stereo = Cfg().streamFull ? 2 : 1;  // la cámara de gamestreamd se pone en los dos ojos
    ok = StreamSend(s, &t, sizeof t, fds.data(), int(fds.size()));
    if (!ok) Log(0, "live streaming: could not hand the buffers to gamestreamd");
  }
  for (int fd : fds) close(fd);
  if (!ok) {
    StreamDestroyLocked(s);
    return false;
  }
  s->valid = true;
  s->resSession = s->session;
  s->nextDue = 0;
  for (uint32_t k = 0; k < s->nstreams; ++k)
    Log(1, "live streaming: %ux%u side by side (%s), each eye %ux%u, %u fps", s->outW[k], s->outH[k],
        Cfg().streamFull ? "full SBS" : "half SBS", s->outW[k] / 2, s->outH[k], s->fps);
  return true;
}

// A stereo pair is complete in sc->slots[slot] (both eyes, GENERAL layout), submitted to q:
// put it side by side into a free buffer for the encoder, right after it in the same queue.
void StreamPair(DeviceData* dd, Swapchain* sc, VkQueue q, int slot) {
  Streamer* s = dd->stream;
  if (!s || slot < 0) return;
  if (!s->want) {
    if (s->valid) StreamTryDestroy(s);
    return;
  }
  if (s->valid && s->resSession != s->session.load() && !StreamTryDestroy(s)) return;
  const uint32_t fam = GetQueueInfo(dd, q)->family;
  if (s->valid && fam != s->family && !StreamTryDestroy(s)) return;
  if (!s->valid && !StreamCreate(s, fam, sc->rext, sc->realFormat)) {
    s->want = false;
    return;
  }
  // only the stream's frame rate
  const uint64_t now = StreamNowNs(), interval = 1000000000ull / (s->fps ? s->fps : 60);
  if (s->nextDue && now + interval / 4 < s->nextDue) return;
  int free = -1;
  for (uint32_t i = 0; i < s->nslots; ++i)
    if (s->slots[i].state.load() == kStreamFree) {
      free = int(i);
      break;
    }
  if (free < 0) {  // the encoder is behind: skip this pair (the game never waits)
    ++s->dropped;
    return;
  }
  s->nextDue = std::max(s->nextDue + interval, now + interval / 2);
  StreamSlot& sl = s->slots[free];
  VkCommandBuffer cb = sl.cb;
  dd->d.ResetCommandBuffer(cb, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  dd->d.BeginCommandBuffer(cb, &bi);
  const VkPipelineStageFlags XFER = VK_PIPELINE_STAGE_TRANSFER_BIT, ALL = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  for (int e = 0; e < 2; ++e)
    Barrier(dd, cb, sc->slots[slot].img[e], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, ALL,
            VK_ACCESS_MEMORY_WRITE_BIT, XFER, VK_ACCESS_TRANSFER_READ_BIT);
  for (uint32_t k = 0; k < s->nstreams; ++k) {
    Barrier(dd, cb, s->interm[k], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, XFER, 0, XFER,
            VK_ACCESS_TRANSFER_WRITE_BIT);
    const int32_t half = int32_t(s->outW[k] / 2);
    for (int e = 0; e < 2; ++e) {
      VkImageBlit b{};
      b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      b.srcOffsets[1] = {int32_t(sc->rext.width), int32_t(sc->rext.height), 1};
      b.dstOffsets[0] = {e * half, 0, 0};
      b.dstOffsets[1] = {(e + 1) * half, int32_t(s->outH[k]), 1};
      dd->d.CmdBlitImage(cb, sc->slots[slot].img[e], VK_IMAGE_LAYOUT_GENERAL, s->interm[k],
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
    }
    Barrier(dd, cb, s->interm[k], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, XFER,
            VK_ACCESS_TRANSFER_WRITE_BIT, XFER, VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferImageCopy c{};
    c.bufferRowLength = s->pitch[k] / 4;
    c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.imageExtent = {s->outW[k], s->outH[k], 1};
    dd->d.CmdCopyImageToBuffer(cb, s->interm[k], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sl.buf[k], 1, &c);
  }
  dd->d.EndCommandBuffer(cb);
  dd->d.ResetFences(dd->device, 1, &sl.fence);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  VkResult r;
  {
    std::lock_guard<std::mutex> l(GetQueueInfo(dd, q)->m);
    r = dd->d.QueueSubmit(q, 1, &si, sl.fence);
  }
  if (r != VK_SUCCESS) return;
  sl.ts = now;
  sl.state = kStreamGpu;
  {
    std::lock_guard<std::mutex> l(s->fqM);
    s->fq.push_back(uint32_t(free));
  }
  s->fcv.notify_one();
}

void StartStreamer(DeviceData* dd) {
  if (dd->stream || !dd->streamOk) return;
  auto* s = new Streamer();
  s->dd = dd;
  dd->stream = s;
  s->io = std::thread(StreamIoLoop, s);
  s->waiter = std::thread(StreamWaitLoop, s);
}

void StopStreamer(DeviceData* dd) {
  Streamer* s = dd->stream;
  if (!s) return;
  dd->d.DeviceWaitIdle(dd->device);
  s->quit = true;
  s->fcv.notify_all();
  if (s->io.joinable()) s->io.join();
  if (s->waiter.joinable()) s->waiter.join();
  {
    std::lock_guard<std::mutex> l(s->resM);
    StreamDestroyLocked(s);
  }
  {
    std::lock_guard<std::mutex> l(s->sockM);
    if (s->sock >= 0) close(s->sock);
    s->sock = -1;
  }
  dd->stream = nullptr;
  delete s;
}
