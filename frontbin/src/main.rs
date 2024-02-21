#![feature(generic_const_exprs)]
use galg::{CliffAlgebra, G3};
use core::f32::consts::PI;
fn main() {
    let a = G3::nvec(&[1., 0., 0.]);
    let b = G3::nvec(&[3., 4., 5.]);
    println!("{:?}", a.axis_rotor(PI / 4.).sandwich(b));
}
