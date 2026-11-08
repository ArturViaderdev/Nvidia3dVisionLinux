#version 450
layout(location = 0) in vec3 col;
layout(location = 1) in vec2 uv;
layout(location = 0) out vec4 o;
void main() {
  vec2 f = floor(uv);
  float k = mod(f.x + f.y, 2.0) < 1.0 ? 1.0 : 0.55;
  o = vec4(col * k, 1);
}
