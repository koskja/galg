#![allow(incomplete_features)]
#![allow(dead_code)]
#![feature(generic_const_exprs)]
#![feature(iter_intersperse)]

mod render;

use galg::{
    subset::{GradedSpace, Subbin}, variable::{Expr, ExprVal}, CliffAlgebra, G3Var, G4Var, G5Var, G8Var, M3Var, M1
};
use nalgebra::{RealField, SMatrix, SVector, Vector2};
use num_traits::One;
use pyo3::Python;
use std::f32::consts::PI;
use svg::{
    node::element::Circle,
    Document,
};

fn create_matrix<const DIM: usize, F: RealField + Copy, A: CliffAlgebra<DIM, F>>(versor: A) -> SMatrix<F, DIM, DIM> {
    let mut columns = vec![];
    for i in 0..DIM {
        let mut ovec = SVector::zeros();
        let vec = A::new(One::one(), [i]);
        let res = versor.clone().sandwich(vec);
        for j in 0..DIM {
            ovec[j] = res.project([j]);
        }
        columns.push(ovec);
    }
    SMatrix::from_columns(&columns)
}

fn main() {
    let a = Expr::nvar("a");
    let b = Expr::nvar("b");
    let s1 = G4Var::new(Expr::nconst(1.), Subbin::bits(0b0011));
    let s2 = G4Var::new(Expr::nconst(1.), Subbin::bits(0b1100));
    let s3 = G4Var::new(Expr::nconst(1.), Subbin::bits(0b0110));
    let s4 = G4Var::new(Expr::nconst(1.), Subbin::bits(0b1001));
    let r1 = s1.plane_rotor(a);
    let r2 = s2.plane_rotor(b);
    let r3 = s3.plane_rotor(a);
    let r4 = s4.plane_rotor(b);
    let r = r1 * r2 * r3 * r4;
    println!("exp({:?})={:?}", s1 + s2, r);
    //println!("{}", create_matrix(r))
}

fn space_main() {
    let bivec = M1::new(1., Subbin::bits(0b0011));
    println!("Accelerating through {}", bivec.square_norm());
    let n = 40;
    let max = 2. * PI;
    let doc = Document::new().set("viewBox", (0, 0, 4, 4));
    for i in 0..=n {
        let angle = max * i as f32 / n as f32;
        let rotor = bivec.plane_rotor(angle / 2.);
        let mat = create_matrix(rotor);
        println!("phi = {angle}; {}", mat); //rotor.sandwich(event))
        let doc = render::visualise_map(&|a| mat * a).fold(doc.clone(), Document::add);
        let transformed_event = mat * Vector2::new(1., 2.);

        let x = transformed_event[1];
        let y = transformed_event[0];

        let circle = Circle::new()
            .set("cx", x)
            .set("cy", y)
            .set("r", 0.05)
            .set("fill", "red");

        let doc = doc.add(circle);
        svg::save(format!("/home/koskja/svg/phi-{i}.svg"), &doc).unwrap();
    }

    // let rotor2 = bivec.plane_rotor(-PI / 6.);
    // println!("rotor1 {rotor:?} = exp({bivec:?}, PI / 6)");
    // println!("rotor2 {rotor2:?} = exp({bivec:?}, -PI / 6)");
    // println!("r1 {:?}", rotor * rotor.conjugation());
    // println!("r2 {:?}", rotor2 * rotor2.conjugation());
    // let mat = create_matrix(rotor);
    // println!("det |{}| = {}", mat, mat.determinant());

    //println!("{rotor:?} -> {:?}", rotor.clone().inverse());
    //let i = rotor.clone().inverse().unwrap();
    //println!("invself {rotor:?} * {i:?} = {:?}", rotor.clone() * i);
    //println!("rot({event:?}, {bivec:?}) = conjugate({event:?}, {rotor:?}) = {res:?}");
}
