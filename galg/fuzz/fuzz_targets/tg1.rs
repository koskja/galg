#![allow(unused_mut)]
#![allow(incomplete_features)]
#![no_main]
#![feature(generic_const_exprs)]

use galg::{
    matrix::MatrixG3,
    subset::{IndexSet, Subbin},
    test::is_close,
    CliffAlgebra, G3,
};
use libfuzzer_sys::fuzz_target;

fn test<
    const DIM: usize,
    A: std::fmt::Debug + CliffAlgebra<DIM, f32>,
    B: std::fmt::Debug + CliffAlgebra<DIM, f32>,
>(
    val: [f32; 1 << DIM],
    eps: f32,
) where
    A::Index: std::fmt::Debug,
{
    let slots: Vec<_> = A::iter_basis().collect();
    let a = A::mass_new(val.into_iter().zip(slots.clone()));
    let b = B::mass_new(val.into_iter().zip(slots.clone()));
    for (val, i) in val.into_iter().zip(slots.iter()) {
        let a_reprojected = a.project(i.clone());
        let b_reprojected = a.project(i.clone());
        let i = Subbin::convert_from(i.clone()).0;
        assert!(
            is_close(val, a_reprojected, eps),
            "A: original {val} is different from reprojected {a_reprojected} @ {i:?}: {a:?} {slots:?}"
        );
        assert!(
            is_close(val, b_reprojected, eps),
            "B: original {val} is different from reprojected {b_reprojected} @ {i:?}: {b:?} {slots:?}"
        );
        assert!(
            is_close(a_reprojected, b_reprojected, eps),
            "original {val}: A({a_reprojected}) != B({b_reprojected}) @ {i:?}: {b:?} {slots:?}"
        );
    }
}

fuzz_target!(|data: Vec<i16>| {
    let eps = f32::EPSILON * 100.;
    for val in data.chunks_exact(8) {
        let val: [i16; 8] = val.try_into().unwrap();
        let val = val.map(|x| x as f32);
        if !val.into_iter().all(f32::is_normal) {
            continue;
        }
        test::<3, MatrixG3, G3>(val, eps);
    }
});
