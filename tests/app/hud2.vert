#version 450
// APP_MOVE=1: a HUD marker (2D, w = 1) that moves across the screen; the frame comes in the instance
layout(location = 0) out vec3 col;
void main() {
  const vec2 c[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
  float t = float(gl_InstanceIndex);
  vec2 centre = vec2(-0.6 + 0.4 * sin(t * 0.05), 0.55);
  gl_Position = vec4(centre + c[gl_VertexIndex] * vec2(0.08, 0.05), 0.0, 1.0);
  col = vec3(1.0, 0.9, 0.2);
}
