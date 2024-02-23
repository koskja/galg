#![feature(generic_const_exprs)]
use galg::{
    subset::{Subbin, SubsetCollection},
    CliffAlgebra, G3, G4, G8,
};

fn main() {
    let g = G3::mass_new(G3::iter_slots().enumerate().map(|(x, i)| (x as f32, i)));
    let g = G3::mass_new([
        (288527., Subbin::bits(0)),
        (288527., Subbin::bits(1)),
        (40., Subbin::bits(2)),
    ]);
    let a = g.conjugation() * g.involution() * g.reversion();
    let proj = (g * a).project([]);
    println!(
        "{:?} * {:?} => {:?} => {a:?}",
        g.conjugation(),
        g.involution(),
        g.conjugation() * g.involution()
    );
    println!("a({a:?}) / ga({:?})={:?}", proj, a / proj,);
}
