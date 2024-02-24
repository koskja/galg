#![allow(unused_mut)]
#![allow(incomplete_features)]
#![no_main]
#![feature(generic_const_exprs)]

use galg::{
    matrix::MatrixG3,
    test::{check_equality, Acap},
    CliffAlgebra, G3,
};
use libfuzzer_sys::fuzz_target;
fn test<
    const DIM: usize,
    A: std::fmt::Debug + CliffAlgebra<DIM> + Clone,
    B: std::fmt::Debug + CliffAlgebra<DIM> + Clone,
>(
    data: Vec<[Acap<DIM, A, B>; 2]>,
) {
    for [Acap(l1, l2), Acap(r1, r2)] in data {
        check_equality(l1.clone() + r1.clone(), l2.clone() + r2.clone(), "addition");
        check_equality(
            l1.clone() - r1.clone(),
            l2.clone() - r2.clone(),
            "subtraction",
        );
        check_equality(
            l1.clone() * r1.clone(),
            l2.clone() * r2.clone(),
            "multiplication",
        );
        check_equality(l1.clone().reversion(), l2.clone().reversion(), "reversion");
        check_equality(
            l1.clone().involution(),
            l2.clone().involution(),
            "involution",
        );
        check_equality(
            l1.clone().conjugation(),
            l2.clone().conjugation(),
            "conjugation",
        );
        check_equality(
            l1.clone() * r1.clone().reversion(),
            l2.clone() * r2.clone().reversion(),
            "rev mul",
        );
        /*let (a, b) = (l1.clone().inverse(), l2.clone().inverse());
        assert!((a.is_none() && b.is_none()) || (a.is_some() && b.is_some()), "a({a:?}) b({b:?})");
        if let (Some(a), Some(b)) = (a, b) {
            check_equality(a, b, "inverse")
        }
        if let Some(a) = l1.clone().inverse() {
            check_equality(a.clone() * l1, A::nscalar(1.), &format!("self mul inverse({a:?}) eq identity of A"))
        }
        if let Some(a) = l2.clone().inverse() {
            check_equality(a.clone() * l2, B::nscalar(1.), &format!("self mul inverse({a:?}) eq identity of B"))
        }*/
    }
}

fuzz_target!(|data: Vec<[Acap<3, MatrixG3, G3>; 2]>| test(data));
