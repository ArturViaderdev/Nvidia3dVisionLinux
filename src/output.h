// 3dvisionlinux - optional own output window on the 3D monitor
#pragma once

namespace stereo3d {

struct OutputWindow;

// Creates a borderless, always-on-top, input-transparent X11 window covering the
// primary monitor (or NV3D_OUTPUT_RECT=x,y,w,h).
// Returns nullptr if X11 is not available.
OutputWindow* CreateOutputWindow();
void* OutputDisplay(OutputWindow* o);       // Display*
unsigned long OutputXid(OutputWindow* o);   // Window
void OutputRect(OutputWindow* o, int* x, int* y, int* w, int* h);
void RaiseOutputWindow(OutputWindow* o);
void SetOutputWindowVisible(OutputWindow* o, bool visible);
void DestroyOutputWindow(OutputWindow* o);

}  // namespace stereo3d
