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
#include "hud2.vert.h"
#include "panel.vert.h"
#include "comp.frag.h"
#include "comp.vert.h"
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
  // APP_UNITY=1: like Unity, the scene goes upside down into a 16:10 texture (colour + depth) that
  // is copied upright and centred into the 16:9 window, with black bars
  const bool unity = depth && getenv("APP_UNITY") && getenv("APP_UNITY")[0] == '1';
  const VkExtent2D sext = unity ? VkExtent2D{ext.height * 16 / 10, ext.height} : ext;
  VkImage rtImg = VK_NULL_HANDLE;
  VkImageView rtView = VK_NULL_HANDLE;
  VkImageView dview = VK_NULL_HANDLE;
  VkImage dimg2 = VK_NULL_HANDLE;   // APP_STAGES=3: a second depth image (a layer with its own camera)
  VkImageView dview2 = VK_NULL_HANDLE;
  if (depth) {
    VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ic.imageType = VK_IMAGE_TYPE_2D; ic.format = dfmt; ic.extent = {sext.width, sext.height, 1};
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
    {
      CK(vkCreateImage(dev, &ic, nullptr, &dimg2));
      VkDeviceMemory dm2;
      CK(vkAllocateMemory(dev, &ma, nullptr, &dm2));
      vkBindImageMemory(dev, dimg2, dm2, 0);
      v.image = dimg2;
      CK(vkCreateImageView(dev, &v, nullptr, &dview2));
    }
    if (unity) {
      VkImageCreateInfo rc = ic;
      rc.format = viewFmt;
      rc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
      CK(vkCreateImage(dev, &rc, nullptr, &rtImg));
      VkMemoryRequirements rr;
      vkGetImageMemoryRequirements(dev, rtImg, &rr);
      VkMemoryAllocateInfo ra{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      ra.allocationSize = rr.size;
      for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if (rr.memoryTypeBits & (1u << i)) { ra.memoryTypeIndex = i; break; }
      VkDeviceMemory rm;
      CK(vkAllocateMemory(dev, &ra, nullptr, &rm));
      vkBindImageMemory(dev, rtImg, rm, 0);
      VkImageViewCreateInfo rv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      rv.image = rtImg; rv.viewType = VK_IMAGE_VIEW_TYPE_2D; rv.format = viewFmt;
      rv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      CK(vkCreateImageView(dev, &rv, nullptr, &rtView));
    }
  }
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  VkPipelineLayout pl;
  CK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));
  VkPipelineLayout layoutFor = pl;  // (APP_DXVK: the composite pipeline samples the scene)
  VkSampleCountFlagBits samplesFor = VK_SAMPLE_COUNT_1_BIT;  // (APP_DXVK=2: the HUD target is 4x MSAA)
  VkFormat fmtFor = viewFmt;
  bool blendFor = false;  // (APP_PANEL: dst * 0.5 + src)  // (APP_BB10=1: the back buffer is 10-bit, as in Unreal games)
  auto pipe = [&](VkShaderModule vs, VkShaderModule fs, bool ztest, bool withDepth) {
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
    ms.rasterizationSamples = samplesFor;
    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = 0xF;
    if (blendFor) {
      cba.blendEnable = VK_TRUE;
      cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; cba.dstColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR;
      cba.colorBlendOp = VK_BLEND_OP_ADD;
      cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      cba.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;
    cb.blendConstants[0] = cb.blendConstants[1] = cb.blendConstants[2] = cb.blendConstants[3] = 0.35f;
    VkDynamicState ds[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = 2; dy.pDynamicStates = ds;
    VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ri.colorAttachmentCount = 1; ri.pColorAttachmentFormats = &fmtFor;
    if (withDepth) ri.depthAttachmentFormat = dfmt;
    VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dss.depthTestEnable = ztest; dss.depthWriteEnable = ztest; dss.depthCompareOp = VK_COMPARE_OP_LESS;
    VkGraphicsPipelineCreateInfo g{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri};
    g.stageCount = 2; g.pStages = st; g.pVertexInputState = &vi; g.pInputAssemblyState = &ia;
    g.pViewportState = &vp; g.pRasterizationState = &rs; g.pMultisampleState = &ms;
    g.pColorBlendState = &cb; g.pDynamicState = &dy; g.layout = layoutFor;
    if (withDepth) g.pDepthStencilState = &dss;
    VkPipeline p;
    CK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &g, nullptr, &p));
    return p;
  };
  VkPipeline pScene = depth ? pipe(vs3, fs3, true, true) : pipe(vsS, fs, false, false);
  VkPipeline pHud = pipe(vsH, fs, false, depth && !unity);  // (APP_UNITY: the HUD pass has no depth)
  VkPipeline pHud2 = pipe(mod(spv_hud2_vert, sizeof(spv_hud2_vert)), fs, false, depth && !unity);
  // APP_DXVK=1: like DXVK / VKD3D: scene -> texture; full-screen composite into the game's back
  // buffer, HUD drawn on top in the same pass; the back buffer copied to the swapchain image
  const int dxvkMode = depth && !unity && getenv("APP_DXVK") ? atoi(getenv("APP_DXVK")) : 0;
  const bool dxvk = dxvkMode > 0;
  VkImage msImg = VK_NULL_HANDLE;
  VkImageView msView = VK_NULL_HANDLE;
  VkPipeline pComp = VK_NULL_HANDLE, pHudBB = VK_NULL_HANDLE, pHud2BB = VK_NULL_HANDLE, pPanelBB = VK_NULL_HANDLE;
  VkImage sceneImg = VK_NULL_HANDLE, bbImg = VK_NULL_HANDLE;
  VkImageView sceneView = VK_NULL_HANDLE, bbView = VK_NULL_HANDLE;
  VkDescriptorSet dset = VK_NULL_HANDLE;
  VkPipelineLayout compLayout = VK_NULL_HANDLE;
  if (dxvk) {
    VkPhysicalDeviceMemoryProperties mp2;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp2);
    // APP_BB10=1: 10-bit back buffer; APP_BB_UNORM=1: 8-bit UNORM (no sRGB: blending in the stored values, like EDF 6)
    const bool bb10 = (getenv("APP_BB10") && getenv("APP_BB10")[0] == '1') ||
                      (getenv("APP_BB_UNORM") && getenv("APP_BB_UNORM")[0] == '1');
    const VkFormat bbFmt = getenv("APP_BB10") && getenv("APP_BB10")[0] == '1' ? VK_FORMAT_A2B10G10R10_UNORM_PACK32
                           : bb10 ? VK_FORMAT_B8G8R8A8_UNORM : viewFmt;
    auto mkImg = [&](VkImage* im, VkImageView* vw, VkImageUsageFlags usage, VkSampleCountFlagBits smp = VK_SAMPLE_COUNT_1_BIT,
                     bool ten = false) {
      VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      ic.flags = ten ? 0 : VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
      ic.imageType = VK_IMAGE_TYPE_2D; ic.format = ten ? bbFmt : VK_FORMAT_B8G8R8A8_UNORM; ic.extent = {ext.width, ext.height, 1};
      ic.mipLevels = 1; ic.arrayLayers = 1; ic.samples = smp; ic.usage = usage;
      CK(vkCreateImage(dev, &ic, nullptr, im));
      VkMemoryRequirements mr;
      vkGetImageMemoryRequirements(dev, *im, &mr);
      VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      ma.allocationSize = mr.size;
      for (uint32_t i = 0; i < mp2.memoryTypeCount; ++i)
        if (mr.memoryTypeBits & (1u << i)) { ma.memoryTypeIndex = i; break; }
      VkDeviceMemory m;
      CK(vkAllocateMemory(dev, &ma, nullptr, &m));
      vkBindImageMemory(dev, *im, m, 0);
      VkImageViewCreateInfo v{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      v.image = *im; v.viewType = VK_IMAGE_VIEW_TYPE_2D; v.format = ten ? bbFmt : viewFmt;
      v.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      CK(vkCreateImageView(dev, &v, nullptr, vw));
    };
    mkImg(&sceneImg, &sceneView, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    mkImg(&bbImg, &bbView, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SAMPLE_COUNT_1_BIT, bb10);
    if (dxvkMode >= 2) mkImg(&msImg, &msView, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SAMPLE_COUNT_4_BIT);
    VkDescriptorSetLayoutBinding b0{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT};
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = 1; dl.pBindings = &b0;
    VkDescriptorSetLayout dsl;
    CK(vkCreateDescriptorSetLayout(dev, &dl, nullptr, &dsl));
    VkPipelineLayoutCreateInfo cl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    cl.setLayoutCount = 1; cl.pSetLayouts = &dsl;
    CK(vkCreatePipelineLayout(dev, &cl, nullptr, &compLayout));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dp.maxSets = 1; dp.poolSizeCount = 1; dp.pPoolSizes = &ps;
    VkDescriptorPool pool0;
    CK(vkCreateDescriptorPool(dev, &dp, nullptr, &pool0));
    VkDescriptorSetAllocateInfo da0{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    da0.descriptorPool = pool0; da0.descriptorSetCount = 1; da0.pSetLayouts = &dsl;
    CK(vkAllocateDescriptorSets(dev, &da0, &dset));
    VkSamplerCreateInfo sci0{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    VkSampler smp;
    CK(vkCreateSampler(dev, &sci0, nullptr, &smp));
    VkDescriptorImageInfo ii{smp, sceneView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wd{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wd.dstSet = dset; wd.descriptorCount = 1; wd.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wd.pImageInfo = &ii;
    vkUpdateDescriptorSets(dev, 1, &wd, 0, nullptr);
    layoutFor = compLayout;
    if (dxvkMode >= 2) samplesFor = VK_SAMPLE_COUNT_4_BIT;
    if (bb10) fmtFor = bbFmt;
    pComp = pipe(mod(spv_comp_vert, sizeof(spv_comp_vert)), mod(spv_comp_frag, sizeof(spv_comp_frag)), false, false);
    layoutFor = pl;
    pHudBB = pipe(vsH, fs, false, false);
    pHud2BB = pipe(mod(spv_hud2_vert, sizeof(spv_hud2_vert)), fs, false, false);
    blendFor = true;
    pPanelBB = pipe(mod(spv_panel_vert, sizeof(spv_panel_vert)), fs, false, false);
    blendFor = false;
    samplesFor = VK_SAMPLE_COUNT_1_BIT;
    fmtFor = viewFmt;
  }

  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkCommandPool pool;
  vkCreateCommandPool(dev, &cpci, nullptr, &pool);
  const int F = 2;
  VkCommandBuffer cbs[F];
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = pool; cai.commandBufferCount = F;
  vkAllocateCommandBuffers(dev, &cai, cbs);
  // APP_DXVK=3: the back buffer pass draws from secondary command buffers (composite, then HUD)
  VkCommandBuffer secs[F][2];
  for (int i = 0; i < F; ++i) {
    VkCommandBufferAllocateInfo sai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    sai.commandPool = pool; sai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY; sai.commandBufferCount = 2;
    vkAllocateCommandBuffers(dev, &sai, secs[i]);
  }
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
    if (dxvk) {
      auto barrier = [&](VkImage im, VkImageLayout o, VkImageLayout n, VkImageAspectFlags asp) {
        VkImageMemoryBarrier x{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        x.oldLayout = o; x.newLayout = n;
        x.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; x.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        x.srcQueueFamilyIndex = x.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        x.image = im; x.subresourceRange = {asp, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &x);
      };
      VkViewport vpx{0, 0, float(ext.width), float(ext.height), 0, 1};
      VkRect2D scx{{0, 0}, ext};
      // 1. the scene into its texture (with the depth)
      barrier(sceneImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
      barrier(dimg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
      VkRenderingAttachmentInfo sa{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
      sa.imageView = sceneView; sa.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      sa.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; sa.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      sa.clearValue.color = {{0.1f, 0.1f, 0.1f, 1}};
      VkRenderingAttachmentInfo za{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
      za.imageView = dview; za.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
      za.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; za.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      za.clearValue.depthStencil = {1.0f, 0};
      VkRenderingInfo sri{VK_STRUCTURE_TYPE_RENDERING_INFO};
      sri.renderArea = {{0, 0}, ext}; sri.layerCount = 1; sri.colorAttachmentCount = 1; sri.pColorAttachments = &sa;
      sri.pDepthAttachment = &za;
      vkCmdBeginRendering(cb, &sri);
      vkCmdSetViewport(cb, 0, 1, &vpx); vkCmdSetScissor(cb, 0, 1, &scx);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pScene);
      static const bool mv = getenv("APP_MOVE") && getenv("APP_MOVE")[0] == '1';
      for (int d = 0; d < 8; ++d) vkCmdDraw(cb, 24, 32, 0, d * 32 + (mv ? f * 256 : 0));
      vkCmdEndRendering(cb);
      // 2. composite into the back buffer, HUD on top in the same pass
      barrier(sceneImg, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
      barrier(bbImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
      VkRenderingAttachmentInfo ba{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
      ba.imageView = bbView; ba.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      ba.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; ba.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      if (dxvkMode == 3) {  // EDF 6: 4x MSAA back buffer drawn by secondary command buffers, resolved by the presenter
        barrier(msImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        ba.imageView = msView;
        VkCommandBufferInheritanceRenderingInfo ir{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO};
        ir.colorAttachmentCount = 1; ir.pColorAttachmentFormats = &viewFmt;
        ir.rasterizationSamples = VK_SAMPLE_COUNT_4_BIT;
        VkCommandBufferInheritanceInfo ih{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
        ih.pNext = &ir;
        static const bool mv3 = getenv("APP_MOVE") && getenv("APP_MOVE")[0] == '1';
        for (int k = 0; k < 2; ++k) {
          VkCommandBuffer sc = secs[s][k];
          vkResetCommandBuffer(sc, 0);
          VkCommandBufferBeginInfo sbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
          sbi.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
          sbi.pInheritanceInfo = &ih;
          vkBeginCommandBuffer(sc, &sbi);
          vkCmdSetViewport(sc, 0, 1, &vpx); vkCmdSetScissor(sc, 0, 1, &scx);
          if (k == 0) {
            vkCmdBindPipeline(sc, VK_PIPELINE_BIND_POINT_GRAPHICS, pComp);
            vkCmdBindDescriptorSets(sc, VK_PIPELINE_BIND_POINT_GRAPHICS, compLayout, 0, 1, &dset, 0, nullptr);
            vkCmdDraw(sc, 3, 1, 0, 0);
          } else {
            vkCmdBindPipeline(sc, VK_PIPELINE_BIND_POINT_GRAPHICS, pHudBB); vkCmdDraw(sc, 6, 1, 0, 0);
            if (mv3) { vkCmdBindPipeline(sc, VK_PIPELINE_BIND_POINT_GRAPHICS, pHud2BB); vkCmdDraw(sc, 6, 1, 0, f); }
          }
          vkEndCommandBuffer(sc);
        }
        VkRenderingInfo bri3{VK_STRUCTURE_TYPE_RENDERING_INFO};
        bri3.flags = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT;
        bri3.renderArea = {{0, 0}, ext}; bri3.layerCount = 1; bri3.colorAttachmentCount = 1; bri3.pColorAttachments = &ba;
        vkCmdBeginRendering(cb, &bri3);
        vkCmdExecuteCommands(cb, 2, secs[s]);
        vkCmdEndRendering(cb);
        barrier(msImg, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        barrier(imgs[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        VkImageResolve rr{};
        rr.srcSubresource = rr.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        rr.extent = {ext.width, ext.height, 1};
        vkCmdResolveImage(cb, msImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, imgs[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rr);
        barrier(imgs[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_ASPECT_COLOR_BIT);
      } else {
      if (dxvkMode == 2) {  // draw into the 4x MSAA target, resolved into the back buffer at the end of the pass
        barrier(msImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        ba.imageView = msView;
        ba.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
        ba.resolveImageView = bbView;
        ba.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      }
      VkRenderingInfo bri{VK_STRUCTURE_TYPE_RENDERING_INFO};
      bri.renderArea = {{0, 0}, ext}; bri.layerCount = 1; bri.colorAttachmentCount = 1; bri.pColorAttachments = &ba;
      vkCmdBeginRendering(cb, &bri);
      vkCmdSetViewport(cb, 0, 1, &vpx); vkCmdSetScissor(cb, 0, 1, &scx);
      static const bool pre = getenv("APP_PRE") && getenv("APP_PRE")[0] == '1';
      if (pre) { vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pHudBB); vkCmdDraw(cb, 6, 1, 0, 0); }  // (draw 1 is not the scene)
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pComp);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, compLayout, 0, 1, &dset, 0, nullptr);
      vkCmdDraw(cb, 3, 1, 0, 0);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pHudBB); vkCmdDraw(cb, 6, 1, 0, 0);
      if (mv) { vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pHud2BB); vkCmdDraw(cb, 6, 1, 0, f); }
      static const bool panel = getenv("APP_PANEL") && getenv("APP_PANEL")[0] == '1';
      if (panel) { vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pPanelBB); vkCmdDraw(cb, 6, 1, 0, 0); }
      vkCmdEndRendering(cb);
      // 3. back buffer -> swapchain image
      barrier(bbImg, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
      barrier(imgs[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
      VkImageBlit bl{};
      bl.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      bl.srcOffsets[1] = {int(ext.width), int(ext.height), 1};
      bl.dstSubresource = bl.srcSubresource; bl.dstOffsets[1] = bl.srcOffsets[1];
      vkCmdBlitImage(cb, bbImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, imgs[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_NEAREST);
      barrier(imgs[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_ASPECT_COLOR_BIT);
      }  // dxvkMode != 3
    } else {
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
    if (unity) {
      VkImageMemoryBarrier rb = b;
      rb.image = rtImg;
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &rb);
      ca.imageView = rtView;
      ri.renderArea = {{0, 0}, sext};
    }
    // APP_STAGES=1: a first pass with the far stage (the wall, another camera, many vertices), then
    // the depth is cleared again for the near stage (the three quads)
    // APP_STAGES=2: the far stage is skipped every other 30 frames (the near camera's stage number changes)
    static const int stagesMode = depth && getenv("APP_STAGES") ? atoi(getenv("APP_STAGES")) : 0;
    static const bool stages = stagesMode > 0 && stagesMode < 3;
    // APP_STAGES=3 (like EDF 6): the main scene in depth image A (wall + red quad), a layer with
    // another camera in depth image B (green quad), then more of the main scene in A again (blue quad)
    if (stagesMode == 3) {
      static const bool mv3 = getenv("APP_MOVE") && getenv("APP_MOVE")[0] == '1';
      const uint32_t mi = mv3 ? f * 256 : 0;
      VkViewport vp0{0, 0, float(ext.width), float(ext.height), 0, 1};
      VkRect2D sc0{{0, 0}, ext};
      VkImageMemoryBarrier db2 = b;
      db2.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; db2.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
      db2.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
      db2.image = dimg2; db2.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, nullptr, 0, nullptr, 1, &db2);
      vkCmdBeginRendering(cb, &ri);  // A, cleared: wall + red quad
      vkCmdSetViewport(cb, 0, 1, &vp0); vkCmdSetScissor(cb, 0, 1, &sc0);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pScene);
      for (int d = 0; d < 8; ++d) vkCmdDraw(cb, 12, 32, 0, d * 32 + mi);
      vkCmdEndRendering(cb);
      ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
      VkRenderingAttachmentInfo da2 = da;
      da2.imageView = dview2;
      VkRenderingInfo ri2 = ri;
      ri2.pDepthAttachment = &da2;
      vkCmdBeginRendering(cb, &ri2);  // B, cleared: green quad with the far camera
      vkCmdSetViewport(cb, 0, 1, &vp0); vkCmdSetScissor(cb, 0, 1, &sc0);
      for (int d = 0; d < 8; ++d) vkCmdDraw(cb, 6, 32, 12, 65536 + d * 32 + mi);
      vkCmdEndRendering(cb);
      da.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;  // A again, kept: blue quad
    }
    if (stages && !(stagesMode == 2 && (f / 30) % 2 == 1)) {
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
    if (unity) {  // upside down (negative height), in the 16:10 texture
      vp = VkViewport{0, float(sext.height), float(sext.width), -float(sext.height), vmin, 1};
      sc = VkRect2D{{0, 0}, sext};
    }
    vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
    // APP_MENU_AT=N: from frame N on only the HUD is drawn (a menu without 3D scene)
    static const int menuAt = getenv("APP_MENU_AT") ? atoi(getenv("APP_MENU_AT")) : -1;
    const bool menu = menuAt >= 0 && f >= menuAt;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pScene);
    static const bool moving = getenv("APP_MOVE") && getenv("APP_MOVE")[0] == '1';
    for (int d = 0; d < (menu ? 0 : depth ? 8 : 1); ++d) {
      if (stagesMode == 3) vkCmdDraw(cb, 6, 32, 18, d * 32 + (moving ? f * 256 : 0));  // A again: the blue quad
      else if (stages) vkCmdDraw(cb, 18, 32, 6, d * 32 + (moving ? f * 256 : 0));  // near stage: only the quads
      else vkCmdDraw(cb, depth ? 24 : 18, depth ? 32 : 1, 0, depth ? d * 32 + (moving ? f * 256 : 0) : 0);
    }
    if (unity) {
      vkCmdEndRendering(cb);
      // texture -> window: upright, centred, black bars
      VkImageMemoryBarrier t = b;
      t.image = rtImg; t.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; t.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      t.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; t.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      VkImageMemoryBarrier w = b;
      w.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; w.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      w.srcAccessMask = 0; w.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      VkImageMemoryBarrier two[2] = {t, w};
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, two);
      VkClearColorValue black{{0, 0, 0, 1}};
      VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vkCmdClearColorImage(cb, imgs[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &rng);
      VkMemoryBarrier cm{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      cm.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; cm.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &cm, 0, nullptr, 0, nullptr);
      int x0 = int(ext.width - sext.width) / 2;
      VkImageBlit bl{};
      bl.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      bl.srcOffsets[0] = {0, 0, 0}; bl.srcOffsets[1] = {int(sext.width), int(sext.height), 1};
      bl.dstSubresource = bl.srcSubresource;
      bl.dstOffsets[0] = {x0, int(ext.height), 0}; bl.dstOffsets[1] = {x0 + int(sext.width), 0, 1};  // flipped
      vkCmdBlitImage(cb, rtImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, imgs[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_NEAREST);
      VkImageMemoryBarrier h = b;
      h.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; h.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      h.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; h.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &h);
      // the HUD on the window
      VkRenderingAttachmentInfo hc{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
      hc.imageView = views[idx]; hc.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      hc.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; hc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      VkRenderingInfo hi{VK_STRUCTURE_TYPE_RENDERING_INFO};
      hi.renderArea = {{0, 0}, ext}; hi.layerCount = 1; hi.colorAttachmentCount = 1; hi.pColorAttachments = &hc;
      vkCmdBeginRendering(cb, &hi);
      VkViewport hv{0, 0, float(ext.width), float(ext.height), 0, 1};
      VkRect2D hs{{0, 0}, ext};
      vkCmdSetViewport(cb, 0, 1, &hv); vkCmdSetScissor(cb, 0, 1, &hs);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pHud); vkCmdDraw(cb, 6, 1, 0, 0);
    if (moving) { vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pHud2); vkCmdDraw(cb, 6, 1, 0, f); }
    vkCmdEndRendering(cb);
    b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; b.dstAccessMask = 0;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    }  // !dxvk
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
