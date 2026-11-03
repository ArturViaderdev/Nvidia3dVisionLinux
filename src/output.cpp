// 3dvisionlinux - optional own output window on the 3D (120 Hz) monitor.
// libX11 / libXrandr / libXext are loaded with dlopen, so there is no link-time dependency.
#include "output.h"

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <ctime>
#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "config.h"

namespace stereo3d {

namespace {
// XRandR monitor info (stable ABI, declared here to avoid the extra headers)
struct RRMonitorInfo {
  Atom name;
  Bool primary;
  Bool automatic;
  int noutput;
  int x, y, width, height;
  int mwidth, mheight;
  unsigned long* outputs;
};

struct X {
  bool tried = false, ok = false;
  decltype(&XOpenDisplay) fOpenDisplay;
  decltype(&XCloseDisplay) fCloseDisplay;
  decltype(&XDefaultScreen) fDefaultScreen;
  decltype(&XRootWindow) fRootWindow;
  decltype(&XDisplayWidth) fDisplayWidth;
  decltype(&XDisplayHeight) fDisplayHeight;
  decltype(&XCreateWindow) fCreateWindow;
  decltype(&XMapRaised) fMapRaised;
  decltype(&XUnmapWindow) fUnmapWindow;
  decltype(&XRaiseWindow) fRaiseWindow;
  decltype(&XDestroyWindow) fDestroyWindow;
  decltype(&XStoreName) fStoreName;
  decltype(&XFlush) fFlush;
  decltype(&XSync) fSync;
  decltype(&XBlackPixel) fBlackPixel;
  decltype(&XDefaultVisual) fDefaultVisual;
  decltype(&XDefaultDepth) fDefaultDepth;
  decltype(&XInitThreads) fInitThreads;
  decltype(&XInternAtom) fInternAtom;
  decltype(&XChangeProperty) fChangeProperty;
  RRMonitorInfo* (*GetMonitors)(Display*, Window, Bool, int*) = nullptr;
  void (*FreeMonitors)(RRMonitorInfo*) = nullptr;
  void (*ShapeCombineRectangles)(Display*, Window, int, int, int, XRectangle*, int, int, int) = nullptr;
};
X g;

bool LoadX() {
  if (g.tried) return g.ok;
  g.tried = true;
  void* lib = dlopen("libX11.so.6", RTLD_NOW | RTLD_LOCAL);
  if (!lib) return false;
#define L(n) g.f##n = (decltype(g.f##n))dlsym(lib, "X" #n); if (!g.f##n) return false;
  L(OpenDisplay) L(CloseDisplay) L(DefaultScreen) L(RootWindow) L(DisplayWidth) L(DisplayHeight)
  L(CreateWindow) L(MapRaised) L(UnmapWindow) L(RaiseWindow) L(DestroyWindow) L(StoreName) L(Flush) L(Sync)
  L(BlackPixel) L(DefaultVisual) L(DefaultDepth) L(InitThreads) L(InternAtom) L(ChangeProperty)
#undef L
  if (void* rr = dlopen("libXrandr.so.2", RTLD_NOW | RTLD_LOCAL)) {
    g.GetMonitors = (decltype(g.GetMonitors))dlsym(rr, "XRRGetMonitors");
    g.FreeMonitors = (decltype(g.FreeMonitors))dlsym(rr, "XRRFreeMonitors");
  }
  if (void* xe = dlopen("libXext.so.6", RTLD_NOW | RTLD_LOCAL))
    g.ShapeCombineRectangles = (decltype(g.ShapeCombineRectangles))dlsym(xe, "XShapeCombineRectangles");
  g.ok = true;
  return true;
}

bool ParseRect(const char* s, int* r) {  // "x,y,w,h"
  if (!s) return false;
  for (int i = 0; i < 4; ++i) {
    char* end = nullptr;
    float v = strtof(s, &end);
    if (end == s) return false;
    r[i] = int(v);
    s = end;
    if (i < 3) {
      if (*s != ',') return false;
      ++s;
    }
  }
  return r[2] > 0 && r[3] > 0;
}
}  // namespace

struct OutputWindow {
  Display* dpy = nullptr;
  Window win = 0;
  int x = 0, y = 0, w = 0, h = 0;
  bool visible = true;
  std::mutex m;
};

OutputWindow* CreateOutputWindow() {
  if (!LoadX()) {
    Log(1, "output window: libX11 not available");
    return nullptr;
  }
  Display* dpy = g.fOpenDisplay(nullptr);
  if (!dpy) {
    Log(1, "output window: cannot open X display");
    return nullptr;
  }
  int scr = g.fDefaultScreen(dpy);
  Window root = g.fRootWindow(dpy, scr);
  int rect[4] = {0, 0, g.fDisplayWidth(dpy, scr), g.fDisplayHeight(dpy, scr)};
  bool fromEnv = ParseRect(getenv("NV3D_OUTPUT_RECT"), rect);
  if (!fromEnv && g.GetMonitors) {
    int n = 0;
    RRMonitorInfo* mons = g.GetMonitors(dpy, root, True, &n);
    int best = -1;  // the primary monitor, else the first one
    for (int i = 0; i < n; ++i)
      if (best < 0 || mons[i].primary) best = i;
    if (best >= 0) {
      rect[0] = mons[best].x; rect[1] = mons[best].y; rect[2] = mons[best].width; rect[3] = mons[best].height;
    }
    if (mons && g.FreeMonitors) g.FreeMonitors(mons);
  }

  XSetWindowAttributes a;
  memset(&a, 0, sizeof(a));
  a.override_redirect = True;
  a.background_pixel = g.fBlackPixel(dpy, scr);
  a.border_pixel = 0;
  a.event_mask = 0;
  Window win = g.fCreateWindow(dpy, root, rect[0], rect[1], unsigned(rect[2]), unsigned(rect[3]), 0,
                              g.fDefaultDepth(dpy, scr), InputOutput, g.fDefaultVisual(dpy, scr),
                              CWOverrideRedirect | CWBackPixel | CWBorderPixel | CWEventMask, &a);
  if (!win) {
    g.fCloseDisplay(dpy);
    return nullptr;
  }
  g.fStoreName(dpy, win, "3dvision");
  // let mouse/keyboard events go through to the game window underneath
  if (g.ShapeCombineRectangles) g.ShapeCombineRectangles(dpy, win, 2 /*ShapeInput*/, 0, 0, nullptr, 0, 0 /*ShapeSet*/, 0);
  g.fMapRaised(dpy, win);
  g.fSync(dpy, False);
  auto* o = new OutputWindow();
  o->dpy = dpy;
  o->win = win;
  o->x = rect[0]; o->y = rect[1]; o->w = rect[2]; o->h = rect[3];
  Log(1, "output window %dx%d at (%d,%d)%s", o->w, o->h, o->x, o->y, fromEnv ? " (NV3D_OUTPUT_RECT)" : "");
  return o;
}

void* OutputDisplay(OutputWindow* o) { return o->dpy; }
unsigned long OutputXid(OutputWindow* o) { return o->win; }
void OutputRect(OutputWindow* o, int* x, int* y, int* w, int* h) {
  *x = o->x; *y = o->y; *w = o->w; *h = o->h;
}

void RaiseOutputWindow(OutputWindow* o) {
  if (!o) return;
  std::lock_guard<std::mutex> l(o->m);
  if (!o->visible) return;
  g.fRaiseWindow(o->dpy, o->win);
  g.fFlush(o->dpy);
}

void SetOutputWindowVisible(OutputWindow* o, bool visible) {
  if (!o) return;
  std::lock_guard<std::mutex> l(o->m);
  if (visible == o->visible) return;
  o->visible = visible;
  if (visible) g.fMapRaised(o->dpy, o->win);
  else g.fUnmapWindow(o->dpy, o->win);
  g.fFlush(o->dpy);
}

void DestroyOutputWindow(OutputWindow* o) {
  if (!o) return;
  {
    std::lock_guard<std::mutex> l(o->m);
    g.fDestroyWindow(o->dpy, o->win);
    g.fSync(o->dpy, False);
    g.fCloseDisplay(o->dpy);
  }
  delete o;
}

}  // namespace stereo3d
