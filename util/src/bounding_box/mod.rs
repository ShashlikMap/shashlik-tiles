use geo::Coord;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BoundingBox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl From<(Option<f64>, Option<f64>, Option<f64>, Option<f64>)> for BoundingBox {
    fn from(value: (Option<f64>, Option<f64>, Option<f64>, Option<f64>)) -> Self {
        let (min_x, min_y, max_x, max_y) = value;

        Self {
            min_x: min_x.unwrap_or_default(),
            min_y: min_y.unwrap_or_default(),
            max_x: max_x.unwrap_or_default(),
            max_y: max_y.unwrap_or_default(),
        }
    }
}

impl From<Coord> for BoundingBox {
    fn from(value: Coord) -> Self {
        BoundingBox {
            min_x: value.x,
            min_y: value.y,
            max_x: value.x,
            max_y: value.y,
        }
    }
}

impl From<BoundingBox> for geo::Rect {
    fn from(value: BoundingBox) -> Self {
        Self::new(
            geo::coord! { x: value.min_x, y: value.min_y },
            geo::coord! { x: value.max_x, y: value.max_y },
        )
    }
}

#[inline]
fn minmax(
    acc: (Option<f64>, Option<f64>, Option<f64>, Option<f64>),
    coord: &Coord,
) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
    let (min_x, min_y, max_x, max_y) = acc;
    (
        min_x.map(|x| coord.x.min(x)).or(Some(coord.x)),
        min_y.map(|y| coord.y.min(y)).or(Some(coord.y)),
        max_x.map(|x| coord.x.max(x)).or(Some(coord.x)),
        max_y.map(|y| coord.x.max(y)).or(Some(coord.y)),
    )
}

impl BoundingBox {
    pub fn new<'a>(shape: impl IntoIterator<Item = &'a Coord>) -> Self {
        shape
            .into_iter()
            .fold((None, None, None, None), minmax)
            .into()
    }

    pub fn try_new<'a, E>(
        shape: impl IntoIterator<Item = Result<&'a Coord, E>>,
    ) -> Result<Self, E> {
        shape
            .into_iter()
            .try_fold((None, None, None, None), |acc, coord| {
                Ok(minmax(acc, coord?))
            })
            .map(BoundingBox::from)
    }

    pub fn coords(&self) -> impl Iterator<Item = Coord> {
        [
            Coord {
                x: self.min_x,
                y: self.min_y,
            },
            Coord {
                x: self.max_x,
                y: self.max_y,
            },
        ]
        .into_iter()
    }
}