use lyon_tessellation::{
    FillGeometryBuilder, FillVertex, GeometryBuilder, GeometryBuilderError, StrokeGeometryBuilder,
    StrokeVertex, VertexId,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Vetex2d {
    x: f32,
    y: f32,
    /// Index into the shader's static color palette (see `shader.wgsl`).
    color: u32,
}

impl Vetex2d {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vetex2d>() as wgpu::BufferAddress, // 12 bytes
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

#[derive(Default, Debug)]
pub struct Vertex2dBuffer {
    pub verteces: Vec<Vetex2d>,
    pub indices: Vec<u32>,
    /// Palette index stamped onto every vertex pushed while tessellating the
    /// current feature. Set this before each feature's tessellation.
    pub current_color: u32,
}

impl GeometryBuilder for Vertex2dBuffer {
    fn add_triangle(&mut self, a: VertexId, b: VertexId, c: VertexId) {
        self.indices.push(a.0);
        self.indices.push(b.0);
        self.indices.push(c.0);
    }
}

impl StrokeGeometryBuilder for Vertex2dBuffer {
    fn add_stroke_vertex(
        &mut self,
        vertex: StrokeVertex,
    ) -> Result<VertexId, GeometryBuilderError> {
        if self.verteces.len() == u32::MAX as usize {
            return Err(GeometryBuilderError::TooManyVertices);
        }

        let point = vertex.position();

        self.verteces.push(Vetex2d {
            x: point.x,
            y: point.y,
            color: self.current_color,
        });

        Ok(((self.verteces.len() - 1) as u32).into())
    }
}

impl FillGeometryBuilder for Vertex2dBuffer {
    fn add_fill_vertex(&mut self, vertex: FillVertex) -> Result<VertexId, GeometryBuilderError> {
        if self.verteces.len() == u32::MAX as usize {
            return Err(GeometryBuilderError::TooManyVertices);
        }

        let point = vertex.position();

        self.verteces.push(Vetex2d {
            x: point.x,
            y: point.y,
            color: self.current_color,
        });

        Ok(((self.verteces.len() - 1) as u32).into())
    }
}
