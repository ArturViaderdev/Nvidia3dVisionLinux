// Minimal Vulkan 1.3 test app that mimics how DXVK talks to the driver:
// API 1.3, VkPhysicalDeviceFeatures2 + Vulkan12/13 feature chain, dynamic rendering,
// mutable-format swapchain, FIFO/IMMEDIATE present, semaphores + fences.
#define VK_USE_PLATFORM_XLIB_KHR
#include <X11/Xlib.h>
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "col.frag.h"
#include "hud.vert.h"
#include "scene.vert.h"
#include "scene3d.frag.h"
#include "scene3d.vert.h"
#define CK(x) do { VkResult r_ = (x); if (r_ < 0) { fprintf(stderr, "FAIL %s = %d (line %d)\n", #x, r_, __LINE__); exit(1); } } while (0)

int main(int argc, char** argv) {
  int frames = argc > 1 ? atoi(argv[1]) : 300;
  Display* dpy = XOpenDisplay(nullptr);
  int W = getenv("APP_W") ? atoi(getenv("APP_W")) : 1920, H = getenv("APP_H") ? atoi(getenv("APP_H")) : 1080;
  Window win = XCreateSimpleWindow(dpy, DefaultRootWindow(dpy), 0, 0, W, H, 0, 0, 0);
  XMapWindow(dpy, win);
  XMoveWindow(dpy, win, 0, 0);
  XFlush(dpy);

  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  ai.apiVersion = VK_API_VERSION_1_3;
  ai.pApplicationName = "stereo3d-test";
  const char* iext[] = {"VK_KHR_surface", "VK_KHR_xlib_surface"};
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &ai;
  ici.enabledExtensionCount = 2;
  ici.ppEnabledExtensionNames = iext;
  VkInstance inst;
  CK(vkCreateInstance(&ici, nullptr, &inst));
  uint32_t n = 1;
  VkPhysicalDevice pd;
  vkEnumeratePhysicalDevices(inst, &n, &pd);

  // check extensions like DXVK does
  uint32_t ne = 0;
  vkEnumerateDeviceExtensionProperties(pd, nullptr, &ne, nullptr);
  std::vector<VkExtensionProperties> exts(ne);
  vkEnumerateDeviceExtensionProperties(pd, nullptr, &ne, exts.data());
  bool mutableFmt = false, gpl = false;
  for (auto& e : exts) {
    if (!strcmp(e.extensionName, "VK_KHR_swapchain_mutable_format")) mutableFmt = true;
    if (!strcmp(e.extensionName, "VK_EXT_graphics_pipeline_library")) gpl = true;
  }
  printf("app: device exts=%u mutable_format=%d gpl_visible=%d\n", ne, mutableFmt, gpl);

  VkXlibSurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR};
  sci.dpy = dpy;
  sci.window = win;
  VkSurfaceKHR surf;
  CK(vkCreateXlibSurfaceKHR(inst, &sci, nullptr, &surf));

  float prio = 1;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = 0;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  f13.dynamicRendering = VK_TRUE;
  f13.synchronization2 = VK_TRUE;
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
  f12.timelineSemaphore = VK_TRUE;  // bufferDeviceAddress deliberately left FALSE
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
  std::vector<const char*> dext = {"VK_KHR_swapchain"};
  if (mutableFmt) dext.push_back("VK_KHR_swapchain_mutable_format");
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = uint32_t(dext.size());
  dci.ppEnabledExtensionNames = dext.data();
  VkDevice dev;
  CK(vkCreateDevice(pd, &dci, nullptr, &dev));
  VkQueue q;
  vkGetDeviceQueue(dev, 0, 0, &q);

  VkSurfaceCapabilitiesKHR caps;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surf, &caps);
  printf("app: surface caps currentExtent %ux%u\n", caps.currentExtent.width, caps.currentExtent.height);
  VkExtent2D ext = caps.currentExtent;
  VkFormat fmt = VK_FORMAT_B8G8R8A8_UNORM;
  VkFormat fmts[2] = {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB};
  VkImageFormatListCreateInfo fl{VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO};
  fl.viewFormatCount = 2;
  fl.pViewFormats = fmts;
  VkSwapchainCreateInfoKHR swci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  if (mutableFmt) {
    swci.pNext = &fl;
    swci.flags = VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR;
  }
  swci.surface = surf;
  swci.minImageCount = std::max(3u, caps.minImageCount);
  swci.imageFormat = fmt;
  swci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  swci.imageExtent = ext;
  swci.imageArrayLayers = 1;
  swci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  swci.preTransform = caps.currentTransform;
  swci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swci.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;  // like "vsync off" in a game
  uint32_t npm = 8;
  VkPresentModeKHR pms[8];
  vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surf, &npm, pms);
  bool hasImm = false;
  for (uint32_t i = 0; i < npm; ++i) hasImm |= pms[i] == VK_PRESENT_MODE_IMMEDIATE_KHR;
  if (!hasImm) swci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  swci.clipped = VK_TRUE;
  VkSwapchainKHR sw;
  CK(vkCreateSwapchainKHR(dev, &swci, nullptr, &sw));
  uint32_t ni = 0;
  vkGetSwapchainImagesKHR(dev, sw, &ni, nullptr);
  std::vector<VkImage> imgs(ni);
  vkGetSwapchainImagesKHR(dev, sw, &ni, imgs.data());
  std::vector<VkImageView> views(ni);
  for (uint32_t i = 0; i < ni; ++i) {
    VkImageViewCreateInfo v{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    v.image = imgs[i];
    v.viewType = VK_IMAGE_VIEW_TYPE_2D;
    v.format = mutableFmt ? VK_FORMAT_B8G8R8A8_SRGB : fmt;  // view with the other format (mutable)
    v.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    CK(vkCreateImageView(dev, &v, nullptr, &views[i]));
  }
  VkFormat viewFmt = mutableFmt ? VK_FORMAT_B8G8R8A8_SRGB : fmt;

  auto mod = [&](const uint32_t* c, size_t sz) {
    VkShaderModuleCreateInfo m{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    m.codeSize = sz;
    m.pCode = c;
    VkShaderModule s;
    CK(vkCreateShaderModule(dev, &m, nullptr, &s));
    return s;
  };
  VkShaderModule vsS = mod(spv_scene_vert, sizeof(spv_scene_vert));
  VkShaderModule vsH = mod(spv_hud_vert, sizeof(spv_hud_vert));
  VkShaderModule fs = mod(spv_col_frag, sizeof(spv_col_frag));
  // APP_DEPTH=1: perspective scene with a depth buffer (for the depth / hybrid modes)
  const bool depth = getenv("APP_DEPTH") && getenv("APP_DEPTH")[0] == '1';
  VkShaderModule vs3 = mod(spv_scene3d_vert, sizeof(spv_scene3d_vert));
  VkShaderModule fs3 = mod(spv_scene3d_frag, sizeof(spv_scene3d_frag));
  const VkFormat dfmt = VK_FORMAT_D32_SFLOAT;
  VkImage dimg = VK_NULL_HANDLE;
  VkImageView dview = VK_NULL_HANDLE;
  if (depth) {
    VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ic.imageType = VK_IMAGE_TYPE_2D; ic.format = dfmt; ic.extent = {ext.width, ext.height, 1};
    ic.mipLevels = 1; ic.arrayLayers = 1; ic.samples = VK_SAMPLE_COUNT_1_BIT;
    ic.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    CK(vkCreateImage(dev, &ic, nullptr, &dimg));
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(dev, dimg, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t mt = 0;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
      if (mr.memoryTypeBits & (1u << i)) { mt = i; break; }
    VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ma.allocationSize = mr.size; ma.memoryTypeIndex = mt;
    VkDeviceMemory dm;
    CK(vkAllocateMemory(dev, &ma, nullptr, &dm));
    vkBindImageMemory(dev, dimg, dm, 0);
    VkImageViewCreateInfo v{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    v.image = dimg; v.viewType = VK_IMAGE_VIEW_TYPE_2D; v.format = dfmt;
    v.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    CK(vkCreateImageView(dev, &v, nullptr, &dview));
  }
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  VkPipelineLayout pl;
  CK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));
  auto pipe = [&](VkShaderModule vs, VkShaderModule fs, bool ztest) {
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = vs; st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = fs; st[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.lineWidth = 1; rs.cullMode = VK_CULL_MODE_NONE;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;
    VkDynamicState ds[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = 2; dy.pDynamicStates = ds;
    VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ri.colorAttachmentCount = 1; ri.pColorAttachmentFormats = &viewFmt;
    if (depth) ri.depthAttachmentFormat = dfmt;
    VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dss.depthTestEnable = ztest; dss.depthWriteEnable = ztest; dss.depthCompareOp = VK_COMPARE_OP_LESS;
    VkGraphicsPipelineCreateInfo g{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri};
    g.stageCount = 2; g.pStages = st; g.pVertexInputState = &vi; g.pInputAssemblyState = &ia;
    g.pViewportState = &vp; g.pRasterizationState = &rs; g.pMultisampleState = &ms;
    g.pColorBlendState = &cb; g.pDynamicState = &dy; g.layout = pl;
    if (depth) g.pDepthStencilState = &dss;
    VkPipeline p;
    CK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &g, nullptr, &p));
    return p;
  };
  VkPipeline pScene = depth ? pipe(vs3, fs3, true) : pipe(vsS, fs, false), pHud = pipe(vsH, fs, false);

  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkCommandPool pool;
  vkCreateCommandPool(dev, &cpci, nullptr, &pool);
  const int F = 2;
  VkCommandBuffer cbs[F];
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = pool; cai.commandBufferCount = F;
  vkAllocateCommandBuffers(dev, &cai, cbs);
  VkSemaphore acq[F], done[8];
  VkFence fen[F];
  VkSemaphoreCreateInfo semci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  for (int i = 0; i < F; ++i) { vkCreateSemaphore(dev, &semci, nullptr, &acq[i]); vkCreateFence(dev, &fci, nullptr, &fen[i]); }
  for (uint32_t i = 0; i < ni; ++i) vkCreateSemaphore(dev, &semci, nullptr, &done[i]);

  for (int f = 0; f < frames; ++f) {
    int s = f % F;
    vkWaitForFences(dev, 1, &fen[s], VK_TRUE, UINT64_MAX);
    vkResetFences(dev, 1, &fen[s]);
    uint32_t idx;
    CK(vkAcquireNextImageKHR(dev, sw, UINT64_MAX, acq[s], VK_NULL_HANDLE, &idx));
    VkCommandBuffer cb = cbs[s];
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &bi);
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = imgs[idx]; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    if (depth) {
      VkImageMemoryBarrier db{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      db.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; db.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
      db.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
      db.srcQueueFamilyIndex = db.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      db.image = dimg; db.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, nullptr, 0, nullptr, 1, &db);
      da.imageView = dview; da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
      da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      da.clearValue.depthStencil = {1.0f, 0};
    }
    VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = views[idx]; ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ca.clearValue.color = {{0.1f, 0.1f, 0.1f, 1}};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, ext}; ri.layerCount = 1; ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca;
    if (depth) ri.pDepthAttachment = &da;
    // APP_STAGES=1: a first pass with the far stage (the wall, another camera, many vertices), then
    // the depth is cleared again for the near stage (the three quads)
    static const bool stages = depth && getenv("APP_STAGES") && getenv("APP_STAGES")[0] == '1';
    if (stages) {
      vkCmdBeginRendering(cb, &ri);
      VkViewport vp0{0, 0, float(ext.width), float(ext.height), 0, 1};
      VkRect2D sc0{{0, 0}, ext};
      vkCmdSetViewport(cb, 0, 1, &vp0); vkCmdSetScissor(cb, 0, 1, &sc0);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pScene);
      for (int d = 0; d < 8; ++d) vkCmdDraw(cb, 6, 128, 0, 65536 + d * 128);
      vkCmdEndRendering(cb);
      ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;  // keep the colour, clear the depth
    }
    vkCmdBeginRendering(cb, &ri);
    static const float vmin = getenv("APP_VP_MIN") ? atof(getenv("APP_VP_MIN")) : 0.0f;  // test: depth range
    VkViewport vp{0, 0, float(ext.width), float(ext.height), vmin, 1};
    VkRect2D sc{{0, 0}, ext};
    vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
    // APP_MENU_AT=N: from frame N on only the HUD is drawn (a menu without 3D scene)
    static const int menuAt = getenv("APP_MENU_AT") ? atoi(getenv("APP_MENU_AT")) : -1;
    const bool menu = menuAt >= 0 && f >= menuAt;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pScene);
    static const bool moving = getenv("APP_MOVE") && getenv("APP_MOVE")[0] == '1';
    for (int d = 0; d < (menu ? 0 : depth ? 8 : 1); ++d) {
      if (stages) vkCmdDraw(cb, 18, 32, 6, d * 32 + (moving ? f * 256 : 0));  // near stage: only the quads
      else vkCmdDraw(cb, depth ? 24 : 18, depth ? 32 : 1, 0, depth ? d * 32 + (moving ? f * 256 : 0) : 0);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pHud); vkCmdDraw(cb, 6, 1, 0, 0);
    vkCmdEndRendering(cb);
    b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; b.dstAccessMask = 0;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    vkEndCommandBuffer(cb);
    VkPipelineStageFlags ws = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1; si.pWaitSemaphores = &acq[s]; si.pWaitDstStageMask = &ws;
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = 1; si.pSignalSemaphores = &done[idx];
    CK(vkQueueSubmit(q, 1, &si, fen[s]));
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = &done[idx];
    pi.swapchainCount = 1; pi.pSwapchains = &sw; pi.pImageIndices = &idx;
    CK(vkQueuePresentKHR(q, &pi));
  }
  vkDeviceWaitIdle(dev);
  printf("app: done %d frames\n", frames);
  vkDestroySwapchainKHR(dev, sw, nullptr);
  vkDestroyDevice(dev, nullptr);
  vkDestroyInstance(inst, nullptr);
  return 0;
}
