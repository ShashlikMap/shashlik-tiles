use bytemuck::{Pod, Zeroable};
use hashbrown::HashMap;

/// Tags with string IDs into local blob's string table
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq, Eq)]
pub struct Tag {
    pub key: u32,
    pub value: u32,
}

impl Tag {
    pub fn new(key_val: (u32, u32)) -> Self {
        let (key, value) = key_val;

        Tag { key, value }
    }
}

pub type Tags = Vec<Tag>;

/// OSM packed tag iterator
pub struct TagIterator<I> {
    iter: I,
    tags: Tags,
}

impl<I: Iterator<Item = i32>> Iterator for TagIterator<I> {
    type Item = Tags;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let key = match self.iter.next() {
                Some(key) => key as u32,
                _ => {
                    if self.tags.is_empty() {
                        return None;
                    } else {
                        return Some(std::mem::take(&mut self.tags));
                    }
                }
            };

            if key == 0 {
                return Some(std::mem::take(&mut self.tags));
            }

            let value = self.iter.next()? as u32;

            self.tags.push(Tag { key, value });
        }
    }
}

pub trait IntoTagIterator<I> {
    fn tags(self) -> TagIterator<I>;
}

impl<I: Iterator<Item = i32>> IntoTagIterator<I> for I {
    fn tags(self) -> TagIterator<I> {
        TagIterator {
            iter: self,
            tags: Default::default(),
        }
    }
}

pub struct TagFilter(pub Vec<Vec<(String, Vec<String>)>>);

impl TagFilter {
    pub fn get_sid_filter(&self, string_table: &HashMap<&[u8], u32>) -> SidTagFilter {
        SidTagFilter(
            self.0.iter()
                .filter_map(|f| {
                    let mut filter: Vec<(u32, Vec<u32>)> = Vec::with_capacity(f.len());

                    for (tag, vals) in f {
                        let sid_tag = match string_table.get(tag.as_bytes()) {
                            Some(tag) => *tag,
                            None => continue
                        };

                        let mut sid_vals: Vec<u32> = Vec::with_capacity(vals.len());

                        for val in vals {
                            if let Some(sid_val) = string_table.get(val.as_bytes()) {
                                sid_vals.push(*sid_val);
                            }
                        }

                        if !sid_vals.is_empty() || vals.is_empty() {
                            filter.push((sid_tag, sid_vals));
                        }
                    } 

                    if filter.is_empty() {
                        None
                    } else {
                        Some(filter)
                    }
                })
                .collect()
        )
    }
}

pub struct SidTagFilter(Vec<Vec<(u32, Vec<u32>)>>);

impl SidTagFilter {
    pub fn matches(&self, tags: &Tags) -> bool {
        for tag in tags {
            for filter in &self.0 {
                for (key, vals) in filter {
                    if &tag.key != key {
                        continue
                    }

                    if vals.is_empty() {
                        return true
                    }

                    if vals.contains(&tag.value) {
                        return true
                    }
                }
            }
        }

        false
    }
}