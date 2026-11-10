#version 450
// APP_PANEL=1: a translucent dark green HUD panel (blended: dst * 0.35 + this) over the scene
layout(location = 0) out vec3 col;
void main() {
  const vec2 c[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
  // (w = 2: drawn with a projection, like the HUD of EDF 6: the stereo shift moves it unless told not to)
  gl_Position = vec4((vec2(0.2, -0.3) + c[gl_VertexIndex] * vec2(0.45, 0.35)) * 2.0, 0.0, 2.0);
  col = vec3(0.03, 0.04, 0.08);  // (tinted, like the subtitle panel of EDF 6)
}
