#![allow(dead_code)]
#![feature(generic_const_exprs)]
#![feature(anonymous_lifetime_in_impl_trait)]
#![feature(associated_const_equality)]
#![feature(closure_lifetime_binder)]
#![feature(return_position_impl_trait_in_trait)]
#![feature(coroutines)]
#![feature(coroutine_trait)]
#![feature(const_mut_refs)]
#![feature(const_trait_impl)]
#![feature(const_slice_first_last)]
#![feature(const_option)]
#![feature(const_for)]
#![feature(effects)]

pub mod matrix;
pub mod plusalg;
pub mod subset;
pub mod test;

use std::{
    f32::consts::PI,
    ops::{Add, Div, Mul, Neg, Sub},
};

use plusalg::PlusAlgebra;
use subset::{GradeStorage, Subbin};

use crate::subset::IndexSubset;

pub type G1 = PlusAlgebra<0, 1, f32>;
//type C = PlusAlgebra<0, -1, f32>;
pub type G2 = PlusAlgebra<1, 1, G1>;
pub type G3 = PlusAlgebra<2, 1, G2>;
pub type G4 = PlusAlgebra<3, 1, G3>;
pub type G5 = PlusAlgebra<4, 1, G4>;
pub type G6 = PlusAlgebra<5, 1, G5>;
pub type G7 = PlusAlgebra<6, 1, G6>;
pub type G8 = PlusAlgebra<7, 1, G7>;

fn main() {
    let a = G3::nvec(&[1., 0., 0.]);
    let b = G3::nvec(&[3., 4., 5.]);
    xd::<3, _>(a);
    println!("{:?}", a.axis_rotor(PI / 4.).sandwich(b));
}

fn d(a: test::ArbitraryCliffordAlgebra<3, G3>) {}

fn xd<const DIM: usize, A: CliffAlgebra<DIM>>(_: A) {}
fn rep<const K: usize>(a: [usize; K], step: usize, n: usize) -> impl Iterator<Item = usize> {
    (0..).flat_map(move |i| a.map(|x| x + step * i)).take(n)
}
pub trait CliffAlgebra<const DIM: usize>:
    Sized
    + GradeStorage<DIM, f32>
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Mul<f32, Output = Self>
    + Sub<Self, Output = Self>
    + Neg<Output = Self>
    + Div<f32, Output = Self>
{
    fn spec_involution<S: IndexSubset<DIM>, I: IntoIterator<Item = S>>(
        mut self,
        grades: I,
    ) -> Self {
        self.multi_grade_map(grades, &Neg::neg);
        self
    }
    fn grade_involution<I: IntoIterator<Item = usize>>(self, grades: I) -> Self {
        grades.into_iter().fold(self, |this, grade| {
            this.spec_involution(Subbin::<DIM>::iter_grade(grade))
        })
    }
    fn involution(self) -> Self {
        self.grade_involution(rep([1], 2, DIM + 1))
    }
    fn reversion(self) -> Self {
        self.grade_involution(rep([2, 3], 4, DIM + 1))
    }
    fn conjugation(self) -> Self {
        self.grade_involution(rep([1, 2], 4, DIM + 1))
    }
    /// https://math.stackexchange.com/questions/443555/calculating-the-inverse-of-a-multivector
    fn inverse(self) -> Option<Self> {
        let xd = match DIM {
            0 => Self::nscalar(self.clone().square_norm()),
            1 | 2 => self.clone().conjugation(),
            3 => self.clone().conjugation() * self.clone().involution() * self.clone().reversion(),
            4 => {
                self.clone().conjugation()
                    * (self.clone() * self.clone().conjugation()).grade_involution([3, 4])
            }
            5 => {
                let a = self.clone().conjugation()
                    * self.clone().involution()
                    * self.clone().reversion();
                a.clone() * (self.clone() * a).grade_involution([1, 4])
            }
            _ => unimplemented!(),
        };
        let (q, p) = (xd.clone(), (self * xd).project(&[]));
        (p.abs() > f32::EPSILON).then(|| q / p)
    }
    fn nscalar(a: f32) -> Self {
        Self::new(a, &[])
    }
    fn npscalar(a: f32) -> Self {
        Self::new(a, &Subbin::bits(!(usize::MAX << DIM)))
    }
    fn uscalar() -> Self {
        Self::nscalar(1.)
    }
    fn upscalar() -> Self {
        Self::npscalar(1.)
    }
    fn rdual(self) -> Self {
        self * Self::upscalar()
    }
    fn ldual(self) -> Self {
        Self::upscalar() * Self::upscalar() * Self::upscalar() * self
    }
    fn nvec(v: &[f32]) -> Self {
        v.into_iter()
            .zip(0..DIM)
            .fold(Self::zero(), |acc, (&val, index)| {
                acc + Self::new(val, &[index])
            })
    }
    fn zero() -> Self {
        Self::nscalar(0.)
    }
    fn wedge(self, rhs: Self) -> Self {
        (0..DIM)
            .flat_map(|i| (0..DIM).map(move |j| (i, j)))
            .map(|(i, j)| (self.select_grade(i) * rhs.select_grade(j)).select_grade(i + j))
            .fold(Self::default(), Add::add)
    }
    fn lcont(self, rhs: Self) -> Self {
        (self.ldual().wedge(rhs)).rdual()
    }
    fn rcont(self, lhs: Self) -> Self {
        (self.wedge(lhs.rdual())).ldual()
    }
    fn square_norm(self) -> f32 {
        self.clone().scalar_product(self.reversion())
    }
    fn scalar_product(self, rhs: Self) -> f32 {
        (self * rhs).project(&[])
    }
    fn axis_rotor(self, half_angle: f32) -> Self {
        let (s, c) = half_angle.sin_cos();
        let bivec = self.rdual().multi_select(Subbin::iter_grade(2));
        Self::nscalar(c) + bivec * s
    }
    fn plane_rotor(self, half_angle: f32) -> Self {
        let (s, c) = half_angle.sin_cos();
        let bivec = self.multi_select(Subbin::iter_grade(2));
        Self::nscalar(c) + bivec * s
    }
    fn sandwich(self, mhs: Self) -> Self {
        self.clone() * mhs * self.inverse().unwrap()
    }
}
