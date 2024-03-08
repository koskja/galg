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
    f32::consts::PI, fmt::Debug, ops::{Add, Div, Mul, Neg, Sub}
};

use nalgebra::RealField;
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
    (0..).flat_map(move |i| a.map(|x| x + step * i)).take_while(move |&i| i <= n)
}
pub trait CliffAlgebra<const DIM: usize, F: Copy + RealField>:
    Sized + Debug
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
                let basis_indices = i.iter_elems().fold(String::new(), |acc, nindex| format!("{acc}{}", nindex));
                out.push(format!("{val}e{basis_indices}"));
            }
        }
        out.into_iter().intersperse_with(|| " + ".to_string()).collect()
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