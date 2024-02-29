#![allow(incomplete_features)]
#![allow(dead_code)]
#![feature(generic_const_exprs)]
#![feature(iter_intersperse)]

mod render;

use galg::{
    subset::{GradedSpace, Subbin},
    CliffAlgebra, M1
};
use nalgebra::{SMatrix, SVector, Vector2};
use std::f32::consts::PI;
use svg::{
    node::element::Circle,
    Document,
};

fn create_matrix<const DIM: usize, A: CliffAlgebra<DIM, f32>>(versor: A) -> SMatrix<f32, DIM, DIM> {
    let mut columns = vec![];
    for i in 0..DIM {
        let mut ovec = SVector::zeros();
        let vec = A::new(1., [i]);
        let res = versor.clone().sandwich(vec);
        for j in 0..DIM {
            ovec[j] = res.project([j]);
        }
        columns.push(ovec);
    }
    SMatrix::from_columns(&columns)
}

fn main() {
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
