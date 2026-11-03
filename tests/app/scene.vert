#version 450
layout(location=0) out vec3 col;
// 3 quads (6 verts each) at view depth 2, 6, 20 ; perspective: w = depth
void main(){
  int q = gl_VertexIndex / 6, v = gl_VertexIndex % 6;
  vec2 c[6] = vec2[](vec2(-1,-1),vec2(1,-1),vec2(1,1),vec2(-1,-1),vec2(1,1),vec2(-1,1));
  float depth[3] = float[](2.0, 6.0, 20.0);
  float xoff[3]  = float[](-0.5, 0.0, 0.5);
  vec3 cols[3] = vec3[](vec3(1,0,0), vec3(0,1,0), vec3(0,0,1));
  float d = depth[q];
  vec2 p = c[v]*0.12*d*0.5 + vec2(xoff[q]*d, 0.0);   // keep similar on-screen size
  // clip = (x, y, z, w) with w = depth (perspective)
  gl_Position = vec4(p.x, p.y, d*0.5, d);
  col = cols[q];
}
