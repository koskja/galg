use std::{
    f32::consts::PI,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

use nalgebra::{Complex, Matrix2, Vector3};

use crate::subset::{count_combinations, get_tuple_k, IndexSubset, Sublist};
fn test() {
    const N: usize = 30; // Change n as needed
    let k = 2; // Change k as needed
    let mut x = [0; 2];
    for (a, i) in Sublist::<N>::iter_grade(k).zip(0..count_combinations(N, k)) {
        a.to_elems(&mut x);
        println!("{:?}: {i} - {}", a, get_tuple_k(&x));
    }

    let x = Multivector::from(Vector3::new(1., 0., 0.));
    let y = Multivector::from(Vector3::new(0., 1., 0.));
    let z = Multivector::from(Vector3::new(0., 0., 1.));
    let w = x * y + z;
    println!("{w}");
    println!("{}", Multivector::rotor(x.pt_vec(), PI * 0.5).sandwich(w))
}
pub type Pauli2<F> = Matrix2<Complex<F>>;
/// This is an element of the full Clifford algebra of real 3-space. `Cl(3, 0)` can be embedded into GL<sub>2</sub>(ℂ) as the Pauli matrices.
/// Scalars are multiples of the identity matrix, vectors can be seen in the `From<Vector3>` implementation below.
/// The pseudoscalar is an imaginary multiple of the identity matrix. By the pseudoscalar's commutativity in R3, bivectors are vectors multiplied by `i`.
#[derive(Debug, Clone, Copy)]
pub struct Multivector(pub Matrix2<Complex<f32>>);
impl Multivector {
    pub const I: Self = Self(Pauli2::new(
        Complex::new(0., 1.),
        Complex::new(0., 0.),
        Complex::new(0., 0.),
        Complex::new(0., 1.),
    ));
    pub fn wedge(self, rhs: Self) -> Self {
        (self * rhs + rhs.reversion() * self) * 0.5
    }
    pub fn contraction(self, rhs: Self) -> Self {
        (self * rhs - rhs.reversion() * self) * 0.5
    }
    pub fn unary_map(self, map: impl FnOnce(Pauli2<f32>) -> Pauli2<f32>) -> Self {
        Self(map(self.0))
    }
    pub fn binary_map(
        self,
        other: Self,
        map: impl FnOnce(Pauli2<f32>, Pauli2<f32>) -> Pauli2<f32>,
    ) -> Self {
        Self(map(self.0, other.0))
    }
    pub fn reversion(mut self) -> Self {
        self.0.swap((0, 1), (1, 0));
        Self(self.0.conjugate())
    }
    pub fn involution(self) -> Self {
        let mut this = self.reversion();
        this.0.swap((0, 0), (1, 1));
        this.0[(0, 1)] *= -1.;
        this.0[(1, 0)] *= -1.;
        this
    }
    pub fn conjugation(self) -> Self {
        self.reversion().involution()
    }
    pub fn norm_squared(self) -> f32 {
        self.pt_scalar().powi(2)
            + self.pt_vec().norm_squared()
            + self.pt_bivec().norm_squared()
            + self.pt_pscalar().powi(2)
    }
    pub fn inverse(self) -> Self {
        self.reversion() / self.norm_squared()
    }
    pub fn sandwich(self, mhs: Self) -> Self {
        self * mhs * self.inverse()
    }
    pub fn pt_scalar(self) -> f32 {
        s1(self[(0, 0)].re, self[(1, 1)].re)
    }
    pub fn pt_pscalar(self) -> f32 {
        s1(self[(0, 0)].im, self[(1, 1)].im)
    }
    pub fn pt_vec(self) -> Vector3<f32> {
        Vector3::new(
            s1(self[(1, 0)].re, self[(0, 1)].re),
            s2(self[(1, 0)].im, self[(0, 1)].im),
            s2(self[(0, 0)].re, self[(1, 1)].re),
        )
    }
    pub fn pt_bivec(mut self) -> Vector3<f32> {
        self *= -Self::I;
        Vector3::new(
            s1(self[(1, 0)].re, self[(0, 1)].re),
            s2(self[(1, 0)].im, self[(0, 1)].im),
            s2(self[(0, 0)].re, self[(1, 1)].re),
        )
    }
    pub fn dual(self) -> Self {
        self * (-Self::I)
    }
    pub fn rotor(axis: Vector3<f32>, rad: f32) -> Self {
        let bi = Self::from(axis.normalize()).dual();
        let (s, c) = rad.div(2.).sin_cos();
        bi * s + c.into()
    }
}
impl core::fmt::Display for Multivector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{{:?} / ", self.pt_scalar())?;
        let v: [_; 3] = self.pt_vec().into();
        let b: [_; 3] = self.pt_bivec().into();
        write!(f, "{:?} / ", v)?;
        write!(f, "{:?} / ", b)?;
        write!(f, "{:?}}}", self.pt_pscalar())?;
        Ok(())
    }
}
impl Index<(usize, usize)> for Multivector {
    type Output = Complex<f32>;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index]
    }
}
impl IndexMut<(usize, usize)> for Multivector {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index]
    }
}
fn s1(a: f32, b: f32) -> f32 {
    (a + b) / 2.
}
fn s2(a: f32, b: f32) -> f32 {
    (a - b) / 2.
}

impl From<Complex<f32>> for Multivector {
    fn from(value: Complex<f32>) -> Self {
        Self(Pauli2::identity() * value)
    }
}

impl From<f32> for Multivector {
    fn from(value: f32) -> Self {
        Self::from(Complex::from(value))
    }
}

impl From<Vector3<f32>> for Multivector {
    fn from(value: Vector3<f32>) -> Self {
        let [a, b, c]: [f32; 3] = value.into();
        let (ze, re, im) = (
            Complex::new(0., 0.),
            Complex::new(1., 0.),
            Complex::new(0., 1.),
        );
        Self(Pauli2::new(ze, re, re, ze)) * a
            + Self(Pauli2::new(ze, im, -im, ze)) * b
            + Self(Pauli2::new(re, ze, ze, -re)) * c
    }
}

impl Add for Multivector {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.binary_map(rhs, Add::add)
    }
}
impl Sub for Multivector {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.binary_map(rhs, Sub::sub)
    }
}

impl Mul for Multivector {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.binary_map(rhs, Mul::mul)
    }
}
impl<G: Into<Complex<f32>>> Mul<G> for Multivector {
    type Output = Self;

    fn mul(self, rhs: G) -> Self::Output {
        Self(self.0 * rhs.into())
    }
}
impl<G: Into<Complex<f32>>> Div<G> for Multivector {
    type Output = Self;

    fn div(self, rhs: G) -> Self::Output {
        Self(self.0 / rhs.into())
    }
}
impl Neg for Multivector {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.unary_map(Neg::neg)
    }
}

impl AddAssign for Multivector {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for Multivector {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl MulAssign for Multivector {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl<G: Into<Complex<f32>>> MulAssign<G> for Multivector {
    fn mul_assign(&mut self, rhs: G) {
        self.0 *= rhs.into();
    }
}

impl<G: Into<Complex<f32>>> DivAssign<G> for Multivector {
    fn div_assign(&mut self, rhs: G) {
        self.0 /= rhs.into();
    }
}
