use crate::{
    impl_num_traits,
    subset::{GradedSpace, IndexSet, Subbin},
    CliffAlgebra,
};
use nalgebra::RealField;
use std::{
    fmt::Display,
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
/// Creates a new Clifford algebra by extending an algebra `A` with a new vector orthogonal to all its elements, e<sub>n</sub>. e<sub>n</sub><sup>2</sup> = `EN2`. `n = DIM + 1`
#[derive(Clone, Copy)]
pub struct PlusAlgebra<
    const DIM: usize,
    const EN2: i8,
    F: Copy + RealField,
    A: CliffAlgebra<DIM, F>,
>(A, A, PhantomData<F>);
impl<const DIM: usize, const EN2: i8, F: Display + Copy + RealField, A: CliffAlgebra<DIM, F>> core::fmt::Debug
    for PlusAlgebra<DIM, EN2, F, A>
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
                <Self as GradedSpace::<{ DIM + 1 }, F>>::project(&self, i)
            )?;
        }
        write!(f, "]")
    }
}
impl_num_traits! {
    impl[const DIM: usize, const EN2: i8, F: Copy + RealField, A: CliffAlgebra<DIM, F>] ... for PlusAlgebra<DIM, EN2, F, A> {
        Add(defo; [self, rhs] => Self(self.0 + rhs.0, self.1 + rhs.1, PhantomData)),
        Sub(defo; [self, rhs] => Self(self.0 - rhs.0, self.1 + rhs.1, PhantomData)),
        Neg(defo; [self] => Self(-self.0, -self.1, PhantomData)),
        Mul[F](defo; [self, rhs] => Self(self.0 * rhs, self.1 * rhs, PhantomData)),
        Div[F](defo; [self, rhs] => Self(self.0 / rhs, self.1 / rhs, PhantomData)),
        Default([] => Self(A::zero(), A::zero(), PhantomData)),
        AddAssign(), SubAssign(), MulAssign(), MulAssign[F](), DivAssign[F]()
    }
}
impl<const DIM: usize, const EN2: i8, F: Copy + RealField, A: CliffAlgebra<DIM, F>> Mul<Self>
    for PlusAlgebra<DIM, EN2, F, A>
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let (_self, _rhs) = (self.clone(), rhs.clone());
        let a = _self.0 * _rhs.0 + _self.1 * _rhs.1.involution() * F::from_f32(EN2 as f32).unwrap();
        let b = self.0.clone() * rhs.1.clone() + self.1.clone() * rhs.0.clone().involution();
        if !a.iter().all(|(val, _)| val.is_zero()) || !b.iter().all(|(val, _)| val.is_zero()) {
            //println!("({}+({})e{n})*({}+({})e{n})=({}+({})e{n})", self.0.print(), self.1.print(), rhs.0.print(), rhs.1.print(), a.print(), b.print());
        }

        Self(a, b, PhantomData)
    }
}
impl<const DIM: usize, const EN2: i8, F: Copy + RealField, A: CliffAlgebra<DIM, F>>
    GradedSpace<{ DIM + 1 }, F> for PlusAlgebra<DIM, EN2, F, A>
{
    type Index = Subbin<{ DIM + 1 }>;

    fn assign(&mut self, elem: F, i: impl IndexSet<{ DIM + 1 }>) {
        let bin = Subbin::convert_from(i).to_bits();
        let lower_bits = DIM;
        let mask = usize::MAX << lower_bits; // lowest significant one overlaps precisely the added vector
        if bin & mask != 0 {
            self.1.assign(elem, Subbin::bits(bin & !mask))
        } else {
            self.0.assign(elem, Subbin::bits(bin & !mask))
        }
    }

    fn project(&self, i: impl IndexSet<{ DIM + 1 }>) -> F {
        let bin = Subbin::convert_from(i).to_bits();
        let lower_bits = DIM;
        let mask = usize::MAX << lower_bits; // lowest significant one overlaps precisely the added vector
        if bin & mask != 0 {
            self.1.project(Subbin::bits(bin & !mask))
        } else {
            self.0.project(Subbin::bits(bin & !mask))
        }
    }

    fn zero() -> Self {
        Self(A::zero(), A::zero(), PhantomData)
    }
}
impl<const DIM: usize, const EN2: i8, F: Copy + RealField, A: CliffAlgebra<DIM, F>>
    CliffAlgebra<{ DIM + 1 }, F> for PlusAlgebra<DIM, EN2, F, A>
where
    [(); DIM + 1]:,
{
    /*fn scalar_product(self, rhs: Self) -> f32 {
        self.0.scalar_product(rhs.0) + self.1.scalar_product(rhs.1.involution()) * EN2 as f32
    } */

    fn involution(self) -> Self {
        // cmon bruh
        Self(self.0.involution(), -self.1.involution(), PhantomData)
    }

    fn reversion(self) -> Self {
        Self(self.0.reversion(), self.1.conjugation(), PhantomData)
    }

    fn conjugation(self) -> Self {
        Self(self.0.conjugation(), -self.1.reversion(), PhantomData)
    }

    fn nscalar(a: F) -> Self {
        Self(A::nscalar(a), A::nscalar(F::zero()), PhantomData)
    }

    fn nvec(v: &[F]) -> Self {
        let (l, r) = v.split_at(DIM);
        Self(A::nvec(l), A::nscalar(r[0]), PhantomData)
    }
}
