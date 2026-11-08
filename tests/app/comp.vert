#version 450
// APP_DXVK=1: full-screen triangle (the game's composite of the finished scene into its back buffer)
layout(location = 0) out vec2 uv;
void main() {
  vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  uv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
