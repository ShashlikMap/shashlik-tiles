mod map;
mod vertex;

use std::{iter, sync::Arc};

use clap::Parser;
use tiles::view::{Camera, TILE_RENDER_PX};
use tiles::{LatLon, MERCATOR_EXTENT, Mercator};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

#[derive(Parser)]
pub struct Args {
    /// PMTile file path/url
    pmtiles: Option<String>,
}

/// Nominal stroke width in screen pixels (baked per scene; scales with zoom
/// within a tile band).
const STROKE_PX: f64 = 2.0;

/// Camera transform uploaded to the shader: `clip = mat2(m0, m1) * pos + t`,
/// where `pos` is a scene-relative vertex. `_pad` keeps the 16-byte alignment a
/// uniform buffer wants.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    m0: [f32; 2],
    m1: [f32; 2],
    t: [f32; 2],
    _pad: [f32; 2],
}

/// Build the scene→clip transform for `cam`. Vertices are `scene_coord - origin`;
/// this maps them to NDC (subtract camera centre, rotate by bearing, scale to
/// pixels, flip y, normalize to `[-1, 1]`).
fn camera_uniform(cam: &Camera, origin: [f64; 2], target_zoom: u8, extent: u16) -> CameraUniform {
    let e = extent as f64;
    let k = TILE_RENDER_PX * 2f64.powf(cam.zoom - target_zoom as f64) / e; // px per scene unit
    let (w, h) = (cam.viewport_px[0] as f64, cam.viewport_px[1] as f64);
    let (fx, fy) = cam.center.to_fractional_tile(target_zoom);
    let c = [fx * e, fy * e]; // camera centre in scene units
    let d = [origin[0] - c[0], origin[1] - c[1]];
    let (sin, cos) = cam.bearing.sin_cos();
    let (ax, ay) = (2.0 * k / w, 2.0 * k / h);
    CameraUniform {
        m0: [(ax * cos) as f32, (ay * sin) as f32],
        m1: [(ax * sin) as f32, (-ay * cos) as f32],
        t: [
            (ax * (cos * d[0] + sin * d[1])) as f32,
            (ay * (sin * d[0] - cos * d[1])) as f32,
        ],
        _pad: [0.0, 0.0],
    }
}

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    window: Arc<Window>,
    // Camera-driven scene + transform.
    world: map::World,
    camera: Camera,
    origin: [f64; 2],
    target_zoom: u8,
    extent: u16,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    // Input state.
    cursor: Option<(f64, f64)>,
    panning: bool,
}

impl State {
    async fn new(window: Arc<Window>, args: &Args) -> anyhow::Result<State> {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: true,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we'll have to disable some.
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off, // Trace path
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an Srgb surface texture. Using a different
        // one will result all the colors comming out darker. If you want to support non
        // Srgb surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[Some(&camera_bind_group_layout)],
                immediate_size: 0,
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(vertex::Vetex2d::layout())],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // 2D geometry: don't cull (the y-flip in the camera transform
                // reverses winding, and stroke ribbons face both ways anyway).
                cull_mode: None,
                // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                // or Features::POLYGON_MODE_POINT
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            // If the pipeline will be used with a multiview render pass, this
            // tells wgpu to render to just specific texture layers.
            multiview_mask: None,
            // Useful for optimizing shader compilation on Android
            cache: None,
        });

        let source = args
            .pmtiles
            .as_deref()
            .unwrap_or("https://d41xpk1mpwqmt.cloudfront.net/planet.pmtiles");
        let mut world = map::open_world(source).await;
        let camera = Camera::new(
            // Tokyo starting point
            LatLon::new(35.677045, 139.752748).to_mercator(),
            10.0,
            [size.width.max(1) as f32, size.height.max(1) as f32],
        );
        let scene = world.update(&camera).await.expect("initial scene");
        let target_zoom = scene.target_zoom;
        let extent = scene.extent;
        let (fx, fy) = camera.center.to_fractional_tile(target_zoom);
        let origin = [fx * extent as f64, fy * extent as f64];
        let k = TILE_RENDER_PX * 2f64.powf(camera.zoom - target_zoom as f64) / extent as f64;
        let data = map::tessellate(&scene, origin, (STROKE_PX / k).max(1.0) as f32);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&data.verteces),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&data.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = data.indices.len() as u32;

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::bytes_of(&camera_uniform(&camera, origin, target_zoom, extent)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            window,
            world,
            camera,
            origin,
            target_zoom,
            extent,
            camera_buffer,
            camera_bind_group,
            cursor: None,
            panning: false,
        })
    }

    /// Re-run the view for the current camera and refresh GPU state: on a scene
    /// change (new tiles) re-tessellate; always rewrite the transform uniform.
    fn apply_camera(&mut self) {
        let scene = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.world.update(&self.camera))
        });
        if let Some(scene) = scene {
            self.target_zoom = scene.target_zoom;
            self.extent = scene.extent;
            let (fx, fy) = self.camera.center.to_fractional_tile(self.target_zoom);
            self.origin = [fx * self.extent as f64, fy * self.extent as f64];
            let k = TILE_RENDER_PX * 2f64.powf(self.camera.zoom - self.target_zoom as f64)
                / self.extent as f64;
            let data = map::tessellate(&scene, self.origin, (STROKE_PX / k).max(1.0) as f32);

            if data.indices.is_empty() {
                return;
            }

            println!(
                "new scene {} {:?} {}",
                data.verteces.len(),
                self.camera.center,
                self.camera.zoom
            );
            self.vertex_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Vertex Buffer"),
                        contents: bytemuck::cast_slice(&data.verteces),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
            self.index_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Index Buffer"),
                    contents: bytemuck::cast_slice(&data.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
            self.num_indices = data.indices.len() as u32;
        }
        let u = camera_uniform(&self.camera, self.origin, self.target_zoom, self.extent);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&u));
        self.window.request_redraw();
    }

    /// Pan by a cursor delta (screen pixels): shift the camera centre so content
    /// follows the cursor.
    fn pan_pixels(&mut self, dx: f64, dy: f64) {
        let merc_per_px = MERCATOR_EXTENT / (TILE_RENDER_PX * 2f64.powf(self.camera.zoom));
        self.camera.center = Mercator::new(
            self.camera.center.x() - dx * merc_per_px,
            self.camera.center.y() + dy * merc_per_px, // screen y-down ↔ mercator y-up
        );
        self.apply_camera();
    }

    /// Zoom by `dz` levels (top-down, toward the centre).
    fn zoom_by(&mut self, dz: f64) {
        self.camera.zoom = (self.camera.zoom + dz).clamp(3.0, 20.0);
        self.apply_camera();
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
            // Keep the camera viewport in sync so the transform + visible-tile
            // computation match the framebuffer.
            self.camera.viewport_px = [width as f32, height as f32];
            self.apply_camera();
        }
    }

    fn update(&mut self) {}

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, key: KeyCode, pressed: bool) {
        match (key, pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            _ => {}
        }
    }

    fn render(&mut self) -> anyhow::Result<()> {
        self.window.request_redraw();

        // We can't render unless the surface is configured
        if !self.is_surface_configured {
            return Ok(());
        }

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                // Skip this frame
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                // You could recreate the devices and all resources
                // created with it here, but we'll just bail
                anyhow::bail!("Lost device");
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.965,
                            g: 0.965,
                            b: 0.965,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            if self.vertex_buffer.size() != 0 {
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
            }
        }

        self.queue.submit(iter::once(encoder.finish()));
        self.queue.present(output);

        Ok(())
    }
}

pub struct App {
    args: Args,
    state: Option<State>,
}

impl App {
    pub fn new(args: Args) -> Self {
        Self { args, state: None }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes().with_title("Tile preview");

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        self.state = Some(
            tokio::task::block_in_place(|| {
                let h = tokio::runtime::Handle::current();
                h.block_on(State::new(window, &self.args))
            })
            .unwrap(),
        );
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        self.state = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(e) => {
                        // Log the error and exit gracefully
                        println!("{e}");
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => {
                if button == MouseButton::Left {
                    state.panning = btn_state.is_pressed();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = (position.x, position.y);
                if state.panning
                    && let Some((lx, ly)) = state.cursor
                {
                    state.pan_pixels(pos.0 - lx, pos.1 - ly);
                }
                state.cursor = Some(pos);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dz = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64 * 0.2,
                    MouseScrollDelta::PixelDelta(p) => p.y * 0.005,
                };
                state.zoom_by(dz);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),
            _ => {}
        }
    }
}

pub fn run(args: Args) -> anyhow::Result<()> {
    let event_loop = EventLoop::with_user_event().build()?;
    {
        let mut app = App::new(args);
        event_loop.run_app(&mut app)?;
    }

    Ok(())
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    run(args).unwrap()
}
