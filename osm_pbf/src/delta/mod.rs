/// Packed delta value decoder
pub struct Delta<I> {
    acu: Option<i64>,
    iter: I,
}

impl<I> Delta<I> {
    pub fn new(iter: I) -> Self {
        Delta { acu: None, iter }
    }
}

impl<I: Iterator<Item = i64>> Iterator for Delta<I> {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|v| match &mut self.acu {
            Some(acu) => {
                *acu += v;

                *acu
            }
            None => {
                self.acu = Some(v);

                v
            }
        })
    }
}

pub trait IntoDelta: Sized {
    fn delta(self) -> Delta<Self>;
}

impl<I: Iterator<Item = i64>> IntoDelta for I {
    fn delta(self) -> Delta<Self> {
        Delta::new(self)
    }
}
