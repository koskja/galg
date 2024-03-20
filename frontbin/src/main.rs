#![allow(incomplete_features)]
#![allow(dead_code)]
#![feature(generic_const_exprs)]
#![feature(iter_intersperse)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(associated_const_equality)]

mod render;

use galg::{
    plusalg::PlusAlgebra, subset::{GradedSpace, Subbin}, variable::Expr, AnyCliff, CliffAlgebra, G2Var, G3Var, G4Var, M4Var, G2, G3, M1, M4
};
use nalgebra::{Matrix2, RealField, SMatrix, SVector, Vector1, Vector2, Vector3, Vector4};
use num_traits::{real::Real, One};
use rand::Rng;
use std::{f32::consts::PI, marker::PhantomData};
use svg::{node::element::Circle, Document};

fn create_matrix2<const N: usize, const M: usize, F: RealField + Copy>(
    f: impl Fn(SVector<F, N>) -> SVector<F, M>,
) -> SMatrix<F, M, N> {
    let mut columns = vec![];
    for i in 0..N {
        let mut ovec = SVector::zeros();
        let mut vec = SVector::zeros();
        vec[i] = F::one();
        let res = f(vec);
        for j in 0..M {
            ovec[j] = res[j];
        }
        columns.push(ovec);
    }
    SMatrix::from_columns(&columns)
}
fn create_matrix<const DIM: usize, F: RealField + Copy, A: CliffAlgebra<DIM, F>>(
    versor: A,
) -> SMatrix<F, DIM, DIM> {
    let mut columns = vec![];
    for i in 0..DIM {
        let mut ovec = SVector::zeros();
        let vec = A::new(One::one(), [i]);
        let res = versor.clone().sandwich(vec);
        for j in 0..DIM {
            ovec[j] = res.project([j]);
        }
        columns.push(ovec);
    }
    SMatrix::from_columns(&columns)
}

#[derive(Debug, Clone, Copy)]
pub struct MobiusMap<const N: usize, F, A> {
    pub a: A,
    pub b: A,
    pub c: A,
    pub d: A,
    pub f: PhantomData<F>,
}

impl<const DIM: usize, F: RealField + Copy, A: CliffAlgebra<DIM, F>> MobiusMap<DIM, F, A> {
    pub fn transform_celestial_sphere(
        &self,
        x: SVector<F, { DIM + 1 }>,
        e_inf: SVector<F, { DIM + 1 }>,
    ) -> SVector<F, { DIM + 1 }>
    where
        [(); DIM - 1]:,
        [(); DIM + 1 - 1]:,
    {
        let a_inf = x.dot(&e_inf);
        let transform = hyperplane_conversion::<{ DIM + 1 }, F, PlusAlgebra<DIM, 1, F, A>>(e_inf);
        let x: SVector<F, { DIM + 1 - 1 }> = transform.apply_inverse(x);
        let x: SVector<F, DIM> = SVector::from_column_slice(x.as_slice());
        let (a, a_inf) = self.tcs(x, a_inf);
        let a: SVector<F, { DIM + 1 - 1 }> = SVector::from_column_slice(a.as_slice());
        transform.apply(a) + e_inf * a_inf
    }
    pub fn tcs(&self, x: SVector<F, DIM>, a_inf: F) -> (SVector<F, DIM>, F) {
        let xp = stereo(x, a_inf);
        let xp = self.apply(xp);
        stereo2(xp)
    }
    pub fn lorentz(angle: F) -> Self {
        let half_angle = angle / (F::one() + F::one());
        Self::from_reflection(SVector::zeros(), F::one(), F::zero())
            * Self::from_reflection(SVector::zeros(), half_angle.cosh(), half_angle.sinh())
    }
    pub fn parabolic(a: SVector<F, DIM>, angle: F) -> Self {
        let half_angle = angle / (F::one() + F::one());
        Self::from_reflection(a, F::zero(), F::zero())
            * Self::from_reflection(a, half_angle, -half_angle)
    }
    pub fn from_reflection(a: SVector<F, DIM>, a_inf: F, a0: F) -> Self {
        assert_ne!(a0.powi(2), a.norm_squared() + a_inf.powi(2));
        let a = A::nvec(a.as_slice());
        Self {
            a: -a.clone(),
            b: A::nscalar(a_inf - a0),
            c: A::nscalar(a_inf + a0),
            d: a,
            f: PhantomData,
        }
    }
    pub fn apply(&self, v: SVector<F, DIM>) -> SVector<F, DIM> {
        let x = A::nvec(v.as_slice());
        ((self.a.clone() * x.clone() + self.b.clone())
            * (self.c.clone() * x.clone() + self.d.clone())
                .inverse()
                .unwrap())
        .pt_vec()
    }
}
impl<const N: usize, F: RealField + Copy, A: CliffAlgebra<N, F>> core::ops::Mul
    for MobiusMap<N, F, A>
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            a: self.a.clone() * rhs.a.clone() + self.b.clone() * rhs.c.clone(),
            b: self.a * rhs.b.clone() + self.b * rhs.d.clone(),
            c: self.c.clone() * rhs.a * self.d.clone() * rhs.c,
            d: self.c * rhs.b + self.d * rhs.d,
            f: PhantomData,
        }
    }
}
trait VecMap<F, const N: usize, const M: usize> {
    fn apply(&self, a: SVector<F, N>) -> SVector<F, M>;
    fn apply_inverse(&self, a: SVector<F, M>) -> SVector<F, N>;
    fn inverse(self) -> impl VecMap<F, M, N>;
}
pub struct UniversalInverse<const N: usize, const M: usize, F, A>(pub A, pub PhantomData<F>);
impl<const N: usize, const M: usize, F: RealField + Copy, A: VecMap<F, M, N>> VecMap<F, N, M>
    for UniversalInverse<N, M, F, A>
{
    fn apply(&self, a: SVector<F, N>) -> SVector<F, M> {
        self.0.apply_inverse(a)
    }

    fn apply_inverse(&self, a: SVector<F, M>) -> SVector<F, N> {
        self.0.apply(a)
    }

    fn inverse(self) -> impl VecMap<F, M, N> {
        self.0
    }
}
#[derive(Debug, Clone, Copy)]
pub struct MatrixMap<const N: usize, const M: usize, F: RealField + Copy>(
    SMatrix<F, N, M>,
    SMatrix<F, M, N>,
);
impl<const N: usize, const M: usize, F: RealField + Copy> VecMap<F, N, M> for MatrixMap<N, M, F> {
    fn inverse(self) -> MatrixMap<M, N, F> {
        MatrixMap(self.1, self.0)
    }

    fn apply(&self, a: SVector<F, N>) -> SVector<F, M> {
        self.1 * a
    }

    fn apply_inverse(&self, a: SVector<F, M>) -> SVector<F, N> {
        self.0 * a
    }
}

pub fn hyperplane_conversion<const N: usize, F: RealField + Copy, A: CliffAlgebra<N, F>>(
    normal: SVector<F, N>,
) -> MatrixMap<{ N - 1 }, N, F>
where
    [(); N - 1]:,
{
    let normal = normal.normalize();
    let rotor = A::nvec(normal.as_slice()) * A::new(F::one(), [0]);
    let unrotor = rotor.clone().inverse().unwrap();
    let to = move |x: SVector<F, { N - 1 }>| {
        let mut a = [F::zero(); N];
        a[1..].copy_from_slice(x.as_slice());
        let a = A::nvec(a.as_slice());
        rotor.clone().sandwich(a).pt_vec()
    };
    let from = move |x: SVector<F, N>| {
        let x = A::nvec(x.as_slice());
        let x = unrotor.clone().sandwich(x).pt_vec();
        let mut a = [F::zero(); N - 1];
        a[..].copy_from_slice(&x.as_slice()[1..]);
        a.into()
    };
    MatrixMap(create_matrix2(from), create_matrix2(to))
}
fn stereo<const N: usize, F: RealField + Copy>(x: SVector<F, N>, y: F) -> SVector<F, N> {
    x * (F::one() / (F::one() - y))
}
fn stereo2<const N: usize, F: RealField + Copy>(x: SVector<F, N>) -> (SVector<F, N>, F) {
    let one = F::one();
    let two = one + one;
    let x2 = x.norm_squared();
    let a = one / (x2 + one);
    (x * two * a, (x2 - one) * a)
}
pub struct CelPer<const N: usize, F: RealField + Copy>
where
    [(); N - 1]:,
{
    dir: SVector<F, N>,
    fov: F,
    dist: F,
    proj: MatrixMap<{ N - 1 }, N, F>,
}
impl<const N: usize, F: RealField + Copy> CelPer<N, F>
where
    [(); N - 1]:,
{
    pub fn new(dir: SVector<F, N>, dist: F) -> Self
    where
        [(); 1 << N]:,
    {
        Self {
            dir,
            fov: F::one(),
            dist,
            proj: hyperplane_conversion::<N, F, AnyCliff<N, F>>(dir),
        }
    }
    pub fn dist(&self, a: SVector<F, N>) -> F {
        self.dir.dot(&a)
    }
    pub fn visible(&self, a: SVector<F, N>) -> bool {
        let z = self.dir.dot(&a);
        z >= self.dist
    }
}
impl<const N: usize, F: RealField + Copy> VecMap<F, N, { N - 1 }> for CelPer<N, F> {
    fn apply(&self, a: SVector<F, N>) -> SVector<F, { N - 1 }> {
        let z = self.dir.dot(&a);
        let lambda = self.dist / z;
        self.proj.apply_inverse(a) * lambda
    }

    fn apply_inverse(&self, a: SVector<F, { N - 1 }>) -> SVector<F, N> {
        (self.proj.apply(a) + self.dir).normalize()
    }

    fn inverse(self) -> impl VecMap<F, { N - 1 }, N> {
        UniversalInverse(self, PhantomData)
    }
}

fn main() {
    let proj = CelPer::new(Vector3::new(Expr::nconst(1.), Expr::zero(), Expr::zero()), Expr::nconst(0.3));
    let l_phi = Expr::nvar("lp");
    let p_phi = Expr::nvar("pp");
    let l: MobiusMap<2, Expr, G2Var> = MobiusMap::lorentz(l_phi);
    let p: MobiusMap<2, Expr, G2Var> = MobiusMap::parabolic(Vector2::new(Expr::nconst(1.), Expr::zero()), p_phi);
    let x = Vector3::new(Expr::nvar("a"), Expr::nvar("b"), Expr::nvar("c"));
    println!("{}", x.norm_squared());
    let dir = Vector3::new(Expr::nconst(1.), Expr::zero(), Expr::zero());
    
    println!("{}", js_transformation("transform", |a: SVector<Expr, 5>| {
        let l: MobiusMap<2, Expr, G2Var> = MobiusMap::lorentz(a[3]);
        let p: MobiusMap<2, Expr, G2Var> = MobiusMap::parabolic(Vector2::new(Expr::nconst(1.), Expr::zero()), a[4]);
        let x = Vector3::new(a[0], a[1], a[2]);
        let x = p.transform_celestial_sphere(x, dir);
        let x = l.transform_celestial_sphere(x, dir);
        x
    }));
    println!("{}", js_transformation("depth", |a: SVector<Expr, 3>| {
        Vector1::new(proj.dist(a))
    }));
    println!("{}", js_transformation("project", |a: SVector<Expr, 3>| {
        proj.apply(a)
    }));
}

fn js_transformation<const N: usize, const M: usize>(name: &str, f: impl Fn(SVector<Expr, N>) -> SVector<Expr, M>) -> String {
    let vec = SVector::from_fn(|i, _| Expr::nvar(&format!("val{i}")));
    let res = f(vec);
    let mut o = format!("function {name}(val) {{ var res = []; \n");
    for i in 0..N {
        o += &format!("var val{i} = val[{i}]; \n")
    }
    for j in 0..M {
        o += &format!("res[{j}] = {}; \n", res[j])
    }
    o += "return res; \n}\n";
    o

}

pub fn save_points(points: impl Iterator<Item = Vector2<f32>>, name: &str) {
    let mut d = Document::new().set("viewBox", (0, 0, 4, 4));
    let circle = Circle::new()
        .set("cx", 0.)
        .set("cy", 0.)
        .set("r", 3000.)
        .set("fill", "black");
    d = d.add(circle);
    svg::save(name, &draw_points(d, points)).unwrap()
}
pub fn draw_points(mut d: Document, points: impl Iterator<Item = Vector2<f32>>) -> Document {
    for p in points {
        let circle = Circle::new()
            .set("cx", p.x)
            .set("cy", p.y)
            .set("r", 0.003)
            .set("fill", "white");
        d = d.add(circle);
    }
    d
}

fn space_main() {
    let bivec = M1::new(1., Subbin::bits(0b0011));
    println!("Accelerating through {}", bivec.square_norm());
    let n = 40;
    let max = 2. * PI;
    let doc = Document::new().set("viewBox", (0, 0, 4, 4));
    for i in 0..=n {
        let angle = max * i as f32 / n as f32;
        let rotor = bivec.plane_rotor(angle / 2.);
        let mat = create_matrix(rotor);
        println!("phi = {angle}; {}", mat); //rotor.sandwich(event))
        let doc = render::visualise_map(&|a| mat * a).fold(doc.clone(), Document::add);
        let transformed_event = mat * Vector2::new(1., 2.);

        let x = transformed_event[1];
        let y = transformed_event[0];

        let circle = Circle::new()
            .set("cx", x)
            .set("cy", y)
            .set("r", 0.05)
            .set("fill", "red");

        let doc = doc.add(circle);
        svg::save(format!("/home/koskja/svg/phi-{i}.svg"), &doc).unwrap();
    }

    // let rotor2 = bivec.plane_rotor(-PI / 6.);
    // println!("rotor1 {rotor:?} = exp({bivec:?}, PI / 6)");
    // println!("rotor2 {rotor2:?} = exp({bivec:?}, -PI / 6)");
    // println!("r1 {:?}", rotor * rotor.conjugation());
    // println!("r2 {:?}", rotor2 * rotor2.conjugation());
    // let mat = create_matrix(rotor);
    // println!("det |{}| = {}", mat, mat.determinant());

    //println!("{rotor:?} -> {:?}", rotor.clone().inverse());
    //let i = rotor.clone().inverse().unwrap();
    //println!("invself {rotor:?} * {i:?} = {:?}", rotor.clone() * i);
    //println!("rot({event:?}, {bivec:?}) = conjugate({event:?}, {rotor:?}) = {res:?}");
}
