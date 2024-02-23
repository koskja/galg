#![allow(unused_mut)]
#![no_main]
#![feature(generic_const_exprs)]

use std::any::Any;

use galg::{matrix::MatrixG3, subset::IndexSet, test::check_equality, CliffAlgebra, G3};
use libfuzzer_sys::fuzz_target;
fn test<const DIM: usize, A: std::fmt::Debug + CliffAlgebra<DIM> + Clone>(data: f32)
where
    A::Index: Clone,
{
    for i in A::iter_slots() {
        let c = i.clone().complement();
        let v = A::new(data, i);
        let rdual = v.clone().rdual();
        let ldual = v.clone().ldual();
        let r_norm = v.clone() * rdual;
        let l_norm = v.clone() * ldual;
        check_equality(v.clone().rdual().ldual(), v.clone(), "duality inverse");
        check_equality(
            r_norm,
            A::npscalar((v.clone() * v.clone()).project([])),
            "right norm",
        );
        check_equality(
            l_norm,
            A::npscalar((v.clone() * v.clone()).project([])).involution(),
            "left norm",
        );
    }
}

fuzz_target!(|data: i16| {
    test::<3, MatrixG3>(data as f32);
    test::<3, G3>(data as f32);
});
