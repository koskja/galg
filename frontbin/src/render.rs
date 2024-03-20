use nalgebra::{Complex, Vector2};
use svg::node::element::{path::Data, Path};

fn lerp(start: f32, end: f32, t: f32) -> f32 {
    (1.0 - t) * start + t * end
}
fn even_sample(start: f32, end: f32, steps: usize) -> impl Iterator<Item = f32> + Clone {
    (0..=steps).map(move |i| lerp(start, end, i as f32 / steps as f32))
}

#[derive(Clone, Copy, Debug)]
pub struct Domain2D {
    lx: f32,
    ly: f32,
    hx: f32,
    hy: f32,
}
impl Domain2D {
    pub fn even_sample_x(&self, steps: usize) -> impl Iterator<Item = f32> + Clone {
        even_sample(self.lx, self.hx, steps)
    }
    pub fn even_sample_y(&self, steps: usize) -> impl Iterator<Item = f32> + Clone {
        even_sample(self.ly, self.hy, steps)
    }
    pub fn lattice_sample_xy(
        self,
        steps1: usize,
        steps2: usize,
    ) -> impl Iterator<Item = impl Iterator<Item = Vector2<f32>>> {
        self.even_sample_y(steps1)
            .map(move |y| self.even_sample_x(steps2).map(move |x| Vector2::new(x, y)))
    }
    pub fn lattice_sample_yx(
        self,
        steps1: usize,
        steps2: usize,
    ) -> impl Iterator<Item = impl Iterator<Item = Vector2<f32>>> {
        self.even_sample_x(steps1)
            .map(move |y| self.even_sample_y(steps2).map(move |x| Vector2::new(y, x)))
    }
}

pub fn transform_line(
    map: impl Fn(Vector2<f32>) -> Vector2<f32>,
    points: impl Iterator<Item = Vector2<f32>>,
) -> Path {
    let mut transformed = points.map(&map);
    let cringe = |vec: Vector2<f32>| (vec[0], vec[1]);
    let start = Data::new().move_to(cringe(transformed.next().unwrap()));
    let data = transformed.fold(start, |data, point| data.line_to(cringe(point)));
    Path::new()
        .set("fill", "none")
        .set("stroke", "lightgray")
        .set("stroke-width", 0.001)
        .set("d", data)
}

pub fn visualise_map(
    map: &impl Fn(Vector2<f32>) -> Vector2<f32>,
) -> impl Iterator<Item = Path> + '_ {
    let domain = Domain2D {
        lx: -10.0,
        ly: -10.0,
        hx: 10.0,
        hy: 10.0,
    };
    let steps1 = 100;
    let steps2 = 10;
    let lines_xy = domain
        .lattice_sample_xy(steps1, steps2)
        .map(move |line| transform_line(map, line));
    let lines_yx = domain
        .lattice_sample_yx(steps1, steps2)
        .map(move |line| transform_line(map, line));
    lines_xy.chain(lines_yx)
}

pub fn c_to_r2<T: Copy>(f: impl Fn(Complex<T>) -> Complex<T>) -> impl Fn(Vector2<T>) -> Vector2<T> {
    move |vec| {
        let c = f(Complex::new(vec[0], vec[1]));
        Vector2::new(c.re, c.im)
    }
}
