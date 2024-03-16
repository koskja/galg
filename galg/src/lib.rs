#![allow(dead_code)]
#![allow(incomplete_features)]
#![feature(iter_intersperse)]
#![feature(generic_const_exprs)]
#![feature(anonymous_lifetime_in_impl_trait)]
#![feature(associated_const_equality)]
#![feature(closure_lifetime_binder)]
#![feature(coroutines)]
#![feature(coroutine_trait)]
#![feature(const_mut_refs)]
#![feature(const_trait_impl)]
#![feature(const_slice_first_last)]
#![feature(const_option)]
#![feature(const_for)]
#![feature(effects)]
#![feature(strict_provenance)]
#![feature(slice_ptr_len)]
#![feature(slice_ptr_get)]
#![feature(vec_push_within_capacity)]

pub mod matrix;
pub mod plusalg;
pub mod subset;
pub mod test;
pub mod variable;

use std::{
    f32::consts::PI,
    fmt::Debug,
    ops::{Add, Div, Mul, Neg, Sub},
};

use nalgebra::{RealField, SVector};
use num_traits::Zero;
use plusalg::PlusAlgebra;
use subset::{GradedSpace, Subbin};
use variable::Expr;

use crate::subset::IndexSet;

pub type C = PlusAlgebra<0, -1, f32, f32>;
pub type M1 = PlusAlgebra<1, 1, f32, C>;
pub type M2 = PlusAlgebra<2, 1, f32, M1>;
pub type M3 = PlusAlgebra<3, 1, f32, M2>;
pub type M4 = PlusAlgebra<4, 1, f32, M3>;
pub type G0 = f32;
pub type G1 = PlusAlgebra<0, 1, f32, f32>;
pub type G2 = PlusAlgebra<1, 1, f32, G1>;
pub type G3 = PlusAlgebra<2, 1, f32, G2>;
pub type G4 = PlusAlgebra<3, 1, f32, G3>;
pub type G5 = PlusAlgebra<4, 1, f32, G4>;
pub type G6 = PlusAlgebra<5, 1, f32, G5>;
pub type G7 = PlusAlgebra<6, 1, f32, G6>;
pub type G8 = PlusAlgebra<7, 1, f32, G7>;

pub type GG0<F> = F;
pub type GG1<F> = PlusAlgebra<0, 1, F, F>;
pub type GG2<F> = PlusAlgebra<1, 1, F, GG1<F>>;
pub type GG3<F> = PlusAlgebra<2, 1, F, GG2<F>>;
pub type GG4<F> = PlusAlgebra<3, 1, F, GG3<F>>;
pub type GG5<F> = PlusAlgebra<4, 1, F, GG4<F>>;
pub type GG6<F> = PlusAlgebra<5, 1, F, GG5<F>>;
pub type GG7<F> = PlusAlgebra<6, 1, F, GG6<F>>;
pub type GG8<F> = PlusAlgebra<7, 1, F, GG7<F>>;

pub type CVar = PlusAlgebra<0, -1, Expr, Expr>;
pub type M1Var = PlusAlgebra<1, 1, Expr, CVar>;
pub type M2Var = PlusAlgebra<2, 1, Expr, M1Var>;
pub type M3Var = PlusAlgebra<3, 1, Expr, M2Var>;
pub type M4Var = PlusAlgebra<4, 1, Expr, M3Var>;
pub type G0Var = Expr;
pub type G1Var = PlusAlgebra<0, 1, Expr, Expr>;
pub type G2Var = PlusAlgebra<1, 1, Expr, G1Var>;
pub type G3Var = PlusAlgebra<2, 1, Expr, G2Var>;
pub type G4Var = PlusAlgebra<3, 1, Expr, G3Var>;
pub type G5Var = PlusAlgebra<4, 1, Expr, G4Var>;
pub type G6Var = PlusAlgebra<5, 1, Expr, G5Var>;
pub type G7Var = PlusAlgebra<6, 1, Expr, G6Var>;
pub type G8Var = PlusAlgebra<7, 1, Expr, G7Var>;

fn main() {
    let a = G3::nvec(&[1., 0., 0.]);
    let b = G3::nvec(&[3., 4., 5.]);
    println!("{:?}", a.rdual().plane_rotor(PI / 4.).sandwich(b));
}

pub fn rep<const K: usize>(a: [usize; K], step: usize, n: usize) -> impl Iterator<Item = usize> {
    (0..)
        .flat_map(move |i| a.map(|x| x + step * i))
        .take_while(move |&i| i <= n)
}
#[derive(Debug, Clone, Copy)]
pub struct AnyCliff<const DIM: usize, F>([F; 1 << DIM])
where
    [(); 1 << DIM]:;
impl<const DIM: usize, F: Copy + RealField> Add for AnyCliff<DIM, F>
where
    [(); 1 << DIM]:,
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        for (i, j) in (&mut self.0).into_iter().zip(rhs.0.into_iter()) {
            *i += j
        }
        self
    }
}
impl<const DIM: usize, F: Copy + RealField> Sub for AnyCliff<DIM, F>
where
    [(); 1 << DIM]:,
{
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        for (i, j) in (&mut self.0).into_iter().zip(rhs.0.into_iter()) {
            *i -= j
        }
        self
    }
}
impl<const DIM: usize, F: Copy + RealField> Mul<F> for AnyCliff<DIM, F>
where
    [(); 1 << DIM]:,
{
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self::Output {
        for i in &mut self.0 {
            *i *= rhs
        }
        self
    }
}
macro_rules! a {
    ((assign $s:ident, $e:ident, $i2:ident), $l:literal, $i:ident) => {
        unsafe { (*($s as *mut Self as *mut crate::$i<F>)).assign($e, Subbin::<$l>::from_iter($i2.iter_elems())); }
    };
    ((project $s:ident, $i2:ident), $l:literal, $i:ident) => {
        unsafe { (*($s as *const Self as *const crate::$i<F>)).project(Subbin::<$l>::from_iter($i2.iter_elems())) }
    };
    ((mul $s:ident, $r:ident), $l:literal, $i:ident) => {
        unsafe {
            let a = *(&$s as *const Self as *const crate::$i<F>);
            let b = *(&$r as *const Self as *const crate::$i<F>);
            let c = a * b;
            *(&c as *const crate::$i<F>as *const Self )
        }
    };
    (($($t_:tt)*) => $ty:tt => [$(($l:literal, $t:ident)),*]) => {
        match $($t_)* {
            $($l => a!($ty, $l, $t),)*
            _ => todo!()
        }
    }
}
impl<const DIM: usize, F: Copy + RealField> GradedSpace<DIM, F> for AnyCliff<DIM, F>
where
    [(); 1 << DIM]:,
{
    type Index = Subbin<DIM>;

    fn zero() -> Self {
        Self([F::zero(); 1 << DIM])
    }

    fn assign(&mut self, elem: F, i: impl IndexSet<DIM>) { 
        a!((DIM) => (assign self, elem, i) => [(0, GG0),(1, GG1),(2, GG2),(3, GG3),(4, GG4),(5, GG5),(6, GG6),(7, GG7),(8, GG8)]);
    }

    fn project(&self, i: impl IndexSet<DIM>) -> F {
        a!((DIM) => (project self, i) => [(0, GG0),(1, GG1),(2, GG2),(3, GG3),(4, GG4),(5, GG5),(6, GG6),(7, GG7),(8, GG8)])
    }
}
impl<const DIM: usize, F: Copy + RealField> Div<F> for AnyCliff<DIM, F>
where
    [(); 1 << DIM]:,
{
    type Output = Self;

    fn div(mut self, rhs: F) -> Self::Output {
        for i in &mut self.0 {
            *i *= rhs
        }
        self
    }
}
impl<const DIM: usize, F: Copy + RealField> Neg for AnyCliff<DIM, F>
where
    [(); 1 << DIM]:,
{
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        for i in &mut self.0 {
            *i *= -F::one();
        }
        self
    }
}
impl<const DIM: usize, F: Copy + RealField> Mul for AnyCliff<DIM, F>
where
    [(); 1 << DIM]:,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        a!((DIM) => (mul self, rhs) => [(0, GG0),(1, GG1),(2, GG2),(3, GG3),(4, GG4),(5, GG5),(6, GG6),(7, GG7),(8, GG8)])
    }
}
impl<const DIM: usize, F: Copy + RealField> CliffAlgebra<DIM, F> for AnyCliff<DIM, F> where
    [(); 1 << DIM]:
{
}
pub trait CliffAlgebra<const DIM: usize, F: Copy + RealField>:
    Sized
    + Debug
    + GradedSpace<DIM, F>
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Mul<F, Output = Self>
    + Sub<Self, Output = Self>
    + Neg<Output = Self>
    + Div<F, Output = Self>
{
    fn grade_involution<I: IntoIterator<Item = usize>>(self, grades: I) -> Self {
        grades.into_iter().fold(self, |this, grade| {
            this.multi_grade_map(Self::Index::iter_grade(grade), &Neg::neg)
        })
    }
    fn involution(self) -> Self {
        self.grade_involution(rep([1], 2, DIM))
    }
    fn reversion(self) -> Self {
        self.grade_involution(rep([2, 3], 4, DIM))
    }
    fn conjugation(self) -> Self {
        self.grade_involution(rep([1, 2], 4, DIM))
    }
    /// https://math.stackexchange.com/questions/443555/calculating-the-inverse-of-a-multivector
    fn inverse(self) -> Option<Self> {
        let xd = match DIM {
            0 => Self::nscalar(self.clone().square_norm()),
            1 | 2 => self.clone().conjugation(),
            3 => self.clone().conjugation() * self.clone().involution() * self.clone().reversion(),
            4 => {
                let a = self.clone() * self.clone().conjugation();
                self.clone().conjugation() * a.clone().grade_involution([3, 4])
            }
            5 => {
                let a = self.clone().conjugation()
                    * self.clone().involution()
                    * self.clone().reversion();
                a.clone() * (self.clone() * a).grade_involution([1, 4])
            }
            _ => unimplemented!(),
        };
        let (q, p) = (xd.clone(), (self * xd).project([]));
        //(!p.is_zero()).then(|| q / p)
        Some(q / p)
    }
    fn pt_vec(&self) -> SVector<F, DIM> {
        let mut a = [F::zero(); DIM];
        for i in 0..DIM {
            a[i] = self.project([i]);
        }
        SVector::from_column_slice(&a)
    }
    fn nscalar(a: F) -> Self {
        Self::new(a, [])
    }
    fn npscalar(a: F) -> Self {
        Self::new(a, Subbin::bits(!(usize::MAX << DIM)))
    }
    fn uscalar() -> Self {
        Self::nscalar(F::one())
    }
    fn upscalar() -> Self {
        Self::npscalar(F::one())
    }
    fn rdual(self) -> Self {
        self * Self::upscalar()
    }
    fn ldual(self) -> Self {
        Self::upscalar() * Self::upscalar() * Self::upscalar() * self
    }
    fn nvec(v: &[F]) -> Self {
        v.iter()
            .zip(0..DIM)
            .fold(Self::zero(), |acc, (&val, index)| {
                acc + Self::new(val, [index])
            })
    }
    fn wedge(self, rhs: Self) -> Self {
        (0..DIM)
            .flat_map(|i| (0..DIM).map(move |j| (i, j)))
            .map(|(i, j)| (self.grade_select(i) * rhs.grade_select(j)).grade_select(i + j))
            .fold(Self::zero(), Add::add)
    }
    fn lcont(self, rhs: Self) -> Self {
        (self.ldual().wedge(rhs)).rdual()
    }
    fn rcont(self, lhs: Self) -> Self {
        (self.wedge(lhs.rdual())).ldual()
    }
    fn square_norm(self) -> F {
        self.clone().scalar_product(self)
    }
    fn scalar_product(self, rhs: Self) -> F {
        (self * rhs).project([])
    }
    fn plane_rotor(self, half_angle: F) -> Self {
        let bivec = self.multi_select(Subbin::iter_grade(2));
        let sn = bivec.clone().square_norm();
        let (s, c) = if sn.is_positive() {
            (half_angle.sinh(), half_angle.cosh())
        } else {
            half_angle.sin_cos()
        };
        Self::nscalar(c) + bivec * s
    }
    fn sandwich(self, mhs: Self) -> Self {
        (self.clone() * mhs) * self.inverse().unwrap()
    }
    fn print(&self) -> String {
        let mut out = vec![];
        for i in Self::iter_basis() {
            let val = self.project(i.clone());
            if !val.is_zero() {
                let basis_indices = i
                    .iter_elems()
                    .fold(String::new(), |acc, nindex| format!("{acc}{}", nindex));
                out.push(format!("{val}e{basis_indices}"));
            }
        }
        out.into_iter()
            .intersperse_with(|| " + ".to_string())
            .collect()
    }
}

impl<T: RealField + Copy> GradedSpace<0, T> for T {
    type Index = Subbin<0>;

    fn zero() -> Self {
        Zero::zero()
    }

    fn assign(&mut self, elem: T, _: impl IndexSet<0>) {
        *self = elem;
    }

    fn project(&self, _: impl IndexSet<0>) -> T {
        *self
    }
}
impl<T: RealField + Copy> CliffAlgebra<0, T> for T {}
