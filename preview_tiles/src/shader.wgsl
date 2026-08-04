struct Camera {
    m0: vec2<f32>,
    m1: vec2<f32>,
    t: vec2<f32>,
};
@group(0) @binding(0) var<uniform> cam: Camera;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

const PALETTE_LEN: u32 = 10u;
fn palette(i: u32) -> vec3<f32> {
    var colors = array<vec3<f32>, 10>(
        vec3<f32>(0.130, 0.623, 0.930), // 0 water
        vec3<f32>(0.196, 0.549, 0.251), // 1 forest
        vec3<f32>(0.823, 0.980, 0.831), // 2 grass
        vec3<f32>(0.750, 0.720, 0.680), // 3 building
        vec3<f32>(0.941, 0.933, 0.902), // 4 land
        vec3<f32>(0.460, 0.219, 0.124), // 5 road: major
        vec3<f32>(0.390, 0.439, 0.470), // 6 road: medium
        vec3<f32>(0.390, 0.439, 0.470), // 7 road: minor
        vec3<f32>(0.902, 0.451, 0.129), // 8 road: merged major network
        vec3<f32>(0.400, 0.400, 0.420), // 9 railway
    );
    return colors[min(i, PALETTE_LEN - 1u)];
}

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = palette(model.color_index);
    let m = mat2x2<f32>(cam.m0, cam.m1);
    out.clip_position = vec4<f32>(m * model.position + cam.t, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
