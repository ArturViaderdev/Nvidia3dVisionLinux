#version 450
// APP_DEPTH=1 test scene: a far checkered wall and three quads in front, real perspective depth
// (z_clip = A*w + B like a D3D projection, near 1, far 100)
layout(location = 0) out vec3 col;
layout(location = 1) out vec2 uv;
void main() {
  const vec2 c[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
  const float depth[4] = float[](30.0, 3.0, 6.0, 12.0);
  const float size[4] = float[](40.0, 0.5, 0.9, 1.6);
  const float xoff[4] = float[](0.0, -0.35, 0.05, 0.45);
  const vec3 cols[4] = vec3[](vec3(0.9, 0.9, 0.9), vec3(1, 0.2, 0.2), vec3(0.2, 1, 0.2), vec3(0.3, 0.4, 1));
  int q = gl_VertexIndex / 6, v = gl_VertexIndex % 6;
  float w = depth[q];
  vec2 p = c[v] * size[q];                      // view space (x right, y up), units at distance w
  p.x += float(gl_InstanceIndex & 65535) * 1.0e-5;      // instances: many vertices for the projection samples
  float t = float((gl_InstanceIndex & 65535) / 256);       // APP_MOVE=1: the frame number comes in the instance
  float move = q > 0 ? 0.25 * sin(t * 0.07 + float(q)) : 0.0;
  vec2 ndc = vec2(p.x / w * 1.0 + xoff[q] + move, -p.y / w * 1.7778);
  // APP_STAGES=1: instances >= 65536 belong to a "far stage" drawn with another camera
  // (near 20, far 2000), like games that draw far and near scenery separately
  bool farStage = gl_InstanceIndex >= 65536;
  float A = farStage ? 2000.0 / 1980.0 : 100.0 / 99.0, B = farStage ? -A * 20.0 : -100.0 / 99.0;
  gl_Position = vec4(ndc * w, A * w + B, w);
  col = cols[q];
  uv = c[v] * size[q] * 2.0;
}
