use std::ops::{Add, Div, Mul, Neg, Sub};
use derive_more::{AddAssign, SubAssign, MulAssign, DivAssign, RemAssign};
use crate::{
    subset::{GradeStorage, IndexSubset, Subbin, SubsetCollection},
    CliffAlgebra,
};

impl SubsetCollection<0, f32> for f32 {
    type Index = Subbin<0>;

    fn assign(&mut self, elem: f32, _: &impl IndexSubset<0>) {
        *self = elem;
    }

    fn project(&self, _: &impl IndexSubset<0>) -> f32 {
        *self
    }

    fn iter(&self) -> impl Iterator<Item = (f32, Self::Index)> {
        Some((*self, Subbin::bits(0))).into_iter()
    }

    fn include_other(&mut self, other: &Self) {
        *self += *other;
    }
}
impl GradeStorage<0, f32> for f32 {}
impl CliffAlgebra<0> for f32 {
    fn involution(self) -> Self {
        self
    }
    fn reversion(self) -> Self {
        self
    }
    fn conjugation(self) -> Self {
        self
    }
}
impl SubsetCollection<0, f64> for f64 {
    type Index = Subbin<0>;

    fn assign(&mut self, elem: f64, _: &impl IndexSubset<0>) {
        *self = elem;
    }

    fn project(&self, _: &impl IndexSubset<0>) -> f64 {
        *self
    }

    fn iter(&self) -> impl Iterator<Item = (f64, Self::Index)> {
        Some((*self, Subbin::bits(0))).into_iter()
    }

    fn include_other(&mut self, other: &Self) {
        *self += *other;
    }
}
impl GradeStorage<0, f64> for f64 {}
/// Creates a new Clifford algebra by extending an algebra `A` with a new vector orthogonal to all its elements, e<sub>n</sub>. e<sub>n</sub><sup>2</sup> = `EN2`. `n = DIM + 1`
#[derive(Default, Clone, Copy, AddAssign, SubAssign, MulAssign, DivAssign, RemAssign)]
pub struct PlusAlgebra<const DIM: usize, const EN2: i8, A: CliffAlgebra<DIM>>(A, A);
impl<const DIM: usize, const EN2: i8, A: CliffAlgebra<DIM>> core::fmt::Debug
    for PlusAlgebra<DIM, EN2, A>
where
    [(); DIM + 1]:,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for i in (0..=(DIM + 1)).flat_map(|k| Subbin::iter_grade(k)) {
            //println!("{i:?}");
            write!(
                f,
                "| {} |",
                <Self as SubsetCollection::<{ DIM + 1 }, f32>>::project(&self, &i)
            )?;
        }
        write!(f, "]")
    }
}
impl<const DIM: usize, const EN2: i8, A: CliffAlgebra<DIM>> Add<Self> for PlusAlgebra<DIM, EN2, A> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}
impl<const DIM: usize, const EN2: i8, A: CliffAlgebra<DIM>> Sub<Self> for PlusAlgebra<DIM, EN2, A> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}
impl<const DIM: usize, const EN2: i8, A: CliffAlgebra<DIM>> Neg for PlusAlgebra<DIM, EN2, A> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0, -self.1)
    }
}
impl<const DIM: usize, const EN2: i8, A: CliffAlgebra<DIM>> Mul<f32> for PlusAlgebra<DIM, EN2, A> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0 * rhs, self.1 * rhs)
    }
}
impl<const DIM: usize, const EN2: i8, A: CliffAlgebra<DIM>> Div<f32> for PlusAlgebra<DIM, EN2, A> {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self(self.0 / rhs, self.1 / rhs)
    }
}
impl<const DIM: usize, const EN2: i8, A: CliffAlgebra<DIM>> Mul<Self> for PlusAlgebra<DIM, EN2, A> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let (_self, _rhs) = (self.clone(), rhs.clone());
        Self(
            _self.0 * _rhs.0 + _self.1 * _rhs.1.involution() * EN2 as f32,
            self.0 * rhs.1 + self.1 * rhs.0.involution(),
        )
    }
}
impl<const DIM: usize, const EN2: i8, A: CliffAlgebra<DIM>> SubsetCollection<{ DIM + 1 }, f32>
    for PlusAlgebra<DIM, EN2, A>
{
    type Index = Subbin<{ DIM + 1 }>;

    fn assign(&mut self, elem: f32, i: &impl IndexSubset<{ DIM + 1 }>) {
        let bin = Subbin::convert_from(i).to_bits();
        let lower_bits = DIM;
        let mask = usize::MAX << lower_bits; // lowest significant one overlaps precisely the added vector
        if bin & mask != 0 {
            self.1.assign(elem, &Subbin::bits(bin & !mask))
        } else {
            self.0.assign(elem, &Subbin::bits(bin & !mask))
        }
    }

    fn project(&self, i: &impl IndexSubset<{ DIM + 1 }>) -> f32 {
        let bin = Subbin::convert_from(i).to_bits();
        let lower_bits = DIM;
        let mask = usize::MAX << lower_bits; // lowest significant one overlaps precisely the added vector
        if bin & mask != 0 {
            self.1.project(&Subbin::bits(bin & !mask))
        } else {
            self.0.project(&Subbin::bits(bin & !mask))
        }
    }

    fn include_other(&mut self, other: &Self) {
        *self = self.clone() + other.clone();
    }
}
impl<const DIM: usize, const EN2: i8, A: CliffAlgebra<DIM>> GradeStorage<{ DIM + 1 }, f32>
    for PlusAlgebra<DIM, EN2, A>
where
    [(); DIM + 1]:,
{
}
impl<const DIM: usize, const EN2: i8, A: CliffAlgebra<DIM>> CliffAlgebra<{ DIM + 1 }>
    for PlusAlgebra<DIM, EN2, A>
where
    [(); DIM + 1]:,
{
    /*fn scalar_product(self, rhs: Self) -> f32 {
        self.0.scalar_product(rhs.0) + self.1.scalar_product(rhs.1.involution()) * EN2 as f32
    }

    fn involution(self) -> Self { // cmon bruh
        Self(self.0.involution(), -self.1.involution())
    }

    fn reversion(self) -> Self {
        Self(self.0.reversion(), self.1.conjugation())
    }

    fn conjugation(self) -> Self {
        Self(self.0.conjugation(), -self.1.reversion())
    } */

    fn nscalar(a: f32) -> Self {
        Self(A::nscalar(a), A::nscalar(0.))
    }

    fn nvec(v: &[f32]) -> Self {
        let (l, r) = v.split_at(DIM);
        println!("{DIM} {v:?} {l:?} {r:?}");
        Self(A::nvec(l), A::nscalar(r[0]))
    }
}
