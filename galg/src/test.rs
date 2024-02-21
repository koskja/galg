use crate::subset::{IndexSubset, Subbin};
use crate::CliffAlgebra;
use arbitrary::{self, Arbitrary};
use std::fmt::Debug;

#[derive(Debug)]
pub struct ArbitraryCliffordAlgebra<const DIM: usize, A: CliffAlgebra<DIM>>(pub A);
impl<'a, const DIM: usize, A: CliffAlgebra<DIM>> Arbitrary<'a>
    for ArbitraryCliffordAlgebra<DIM, A>
{
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        (0..=DIM)
            .flat_map(|k| A::Index::iter_grade(k))
            .try_fold(A::zero(), |acc, i| {
                Ok(acc + A::new(u.arbitrary::<i16>()? as f32, &i))
            })
            .map(Self)
    }
}
#[derive(Debug)]
pub struct Acap<const DIM: usize, A: CliffAlgebra<DIM>, B: CliffAlgebra<DIM>>(pub A, pub B);
impl<'a, const DIM: usize, A: CliffAlgebra<DIM>, B: CliffAlgebra<DIM>> Arbitrary<'a>
    for Acap<DIM, A, B>
{
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Iterator::zip(A::iter_slots(), B::iter_slots())
            .try_fold((A::zero(), B::zero()), |(a, b), (i1, i2)| {
                let val: i16 = u.arbitrary()?;
                Ok((a + A::new(val as f32, &i1), b + B::new(val as f32, &i2)))
            })
            .map(|(a, b)| Self(a, b))
    }
}

pub type ArbitraryPlusG1 = ArbitraryCliffordAlgebra<1, crate::G1>;
pub type ArbitraryPlusG2 = ArbitraryCliffordAlgebra<2, crate::G2>;
pub type ArbitraryPlusG3 = ArbitraryCliffordAlgebra<3, crate::G3>;
pub type ArbitraryPauli = ArbitraryCliffordAlgebra<3, crate::matrix::MatrixG3>;

pub fn is_close(a: f32, b: f32, eps: f32) -> bool {
    within_eps(a, b, eps) || within_eps((a / b).abs(), 1., eps)
}
pub fn within_eps(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}
pub fn check_equality<
    const DIM: usize,
    A: Debug + CliffAlgebra<DIM>,
    B: Debug + CliffAlgebra<DIM>,
>(
    a: A,
    b: B,
    s: &str,
) {
    let eps = 0.1;
    let iter = || Iterator::zip(A::iter_slots(), B::iter_slots());
    let (asize, bsize) = iter().fold((0., 0.), |(a_, b_), (i, j)| {
        (a_ + a.project(&i).abs(), b_ + b.project(&j).abs())
    });
    assert!(is_close(asize, bsize, eps), "A({a:?}) is different from B({b:?}) in {s} of eps {eps}\n asize({asize}) != bsize({bsize})");
    for (i1, i2) in iter() {
        let a_ = a.project(&i1);
        let b_ = b.project(&i2);
        let i1 = Subbin::convert_from(&i1);
        let i2 = Subbin::convert_from(&i2);
        if a_.is_finite() && b_.is_finite() {
            assert!(
                within_eps(a_, b_, f32::max(eps, eps * asize)),
                "A({a_}) is different from B({b_}) @ {i1:?} == {i2:?} in {s}; \n {a:?} != {b:?} of eps {}", f32::max(eps, eps * asize)
            )
        }
    }
}
