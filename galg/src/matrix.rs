use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use nalgebra::{Complex, Matrix2, Vector3};

use crate::{
    impl_num_traits,
    subset::{GradedSpace, IndexSet, Subbin},
    CliffAlgebra,
};
pub type Pauli2<F> = Matrix2<Complex<F>>;
/// This is an element of the full Clifford algebra of real 3-space. `Cl(3, 0)` can be embedded into GL<sub>2</sub>(ℂ) as the Pauli matrices.
/// Scalars are multiples of the identity matrix, vectors can be seen in the `From<Vector3>` implementation below.
/// The pseudoscalar is an imaginary multiple of the identity matrix. By the pseudoscalar's commutativity in R3, bivectors are vectors multiplied by `i`.
#[derive(Default, Clone, Copy)]
pub struct MatrixG3(pub Pauli2<f32>);
impl MatrixG3 {
    pub const ONE: Self = Self(Pauli2::new(
        Complex::new(1., 0.),
        Complex::new(0., 0.),
        Complex::new(0., 0.),
        Complex::new(1., 0.),
    ));
    pub const I: Self = Self(Pauli2::new(
        Complex::new(0., 1.),
        Complex::new(0., 0.),
        Complex::new(0., 0.),
        Complex::new(0., 1.),
    ));
    pub fn unary_map(self, map: impl FnOnce(Pauli2<f32>) -> Pauli2<f32>) -> Self {
        Self(map(self.0))
    }
    pub fn binary_map(
        self,
        other: Self,
        map: impl FnOnce(Pauli2<f32>, Pauli2<f32>) -> Pauli2<f32>,
    ) -> Self {
        Self(map(self.0, other.0))
    }
    fn dual(self) -> Self {
        self * (-Self::I)
    }
    fn rotor(axis: Vector3<f32>, rad: f32) -> Self {
        let bi = Self::from(axis.normalize()).dual();
        let (s, c) = rad.div(2.).sin_cos();
        bi * s + c.into()
    }
    fn pt_scalar(self) -> f32 {
        s1(self[(0, 0)].re, self[(1, 1)].re)
    }
    fn pt_pscalar(self) -> f32 {
        s1(self[(0, 0)].im, self[(1, 1)].im)
    }
    fn pt_vec(self) -> Vector3<f32> {
        Vector3::new(
            s1(self[(1, 0)].re, self[(0, 1)].re),
            s2(self[(1, 0)].im, self[(0, 1)].im),
            s2(self[(0, 0)].re, self[(1, 1)].re),
        )
    }
    fn pt_bivec(mut self) -> Vector3<f32> {
        self *= Self::I;
        self.pt_vec()
    }
    fn norm_squared(self) -> f32 {
        self.pt_scalar().powi(2)
            + self.pt_vec().norm_squared()
            + self.pt_bivec().norm_squared()
            + self.pt_pscalar().powi(2)
    }
}
impl core::fmt::Debug for MatrixG3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for i in (0..=3).flat_map(|k| Subbin::iter_grade(k)) {
            //println!("{i:?}");
            write!(
                f,
                "| {} |",
                <Self as GradedSpace::<3, f32>>::project(&self, i)
            )?;
        }
        write!(f, "]")
    }
}
impl crate::subset::GradedSpace<3, f32> for MatrixG3 {
    type Index = Subbin<3>;

    fn new(elem: f32, i: impl IndexSet<3>) -> Self {
        let a = Subbin::convert_from(i).to_bits();
        match a {
            0 => Self::ONE * elem,
            0b001 => Self::from(Vector3::new(elem, 0., 0.)),
            0b010 => Self::from(Vector3::new(0., elem, 0.)),
            0b100 => Self::from(Vector3::new(0., 0., elem)),
            0b011 => Self::from(Vector3::new(0., 0., elem)).rdual(),
            0b101 => -Self::from(Vector3::new(0., elem, 0.)).rdual(),
            0b110 => Self::from(Vector3::new(elem, 0., 0.)).rdual(),
            0b111 => Self::I * elem,
            _ => panic!(),
        }
    }

    fn assign(&mut self, elem: f32, i: impl IndexSet<3>) {
        *self = *self - self.select(i.clone()) + Self::new(elem, i)
    }

    fn project(&self, i: impl IndexSet<3>) -> f32 {
        let a = Subbin::convert_from(i).to_bits();
        match a {
            0 => self.pt_scalar(),
            0b001 => self.pt_vec()[0],
            0b010 => self.pt_vec()[1],
            0b100 => self.pt_vec()[2],
            0b011 => -self.rdual().pt_vec()[2],
            0b101 => self.rdual().pt_vec()[1],
            0b110 => -self.rdual().pt_vec()[0],
            0b111 => self.pt_pscalar(),
            _ => 0.,
        }
    }

    fn mass_new<S: IndexSet<3>, I: IntoIterator<Item = (f32, S)>>(elements: I) -> Self {
        elements
            .into_iter()
            .map(|(val, elem)| Self::new(val, elem))
            .fold(Self::default(), Add::add)
    }
}
impl CliffAlgebra<3, f32> for MatrixG3 {
    /*fn reversion(mut self) -> Self {
        self.0.swap((0, 1), (1, 0));
        Self(self.0.conjugate())
    }
    fn involution(self) -> Self {
        let mut this = self.reversion();
        this.0.swap((0, 0), (1, 1));
        this.0[(0, 1)] *= -1.;
        this.0[(1, 0)] *= -1.;
        this
    }
    fn conjugation(self) -> Self {
        self.reversion().involution()
    } */

    fn inverse(self) -> Option<Self> {
        self.0.try_inverse().map(Self)
    }
}
impl core::fmt::Display for MatrixG3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{{:?} / ", self.pt_scalar())?;
        let v: [_; 3] = self.pt_vec().into();
        let b: [_; 3] = self.pt_bivec().into();
        write!(f, "{:?} / ", v)?;
        write!(f, "{:?} / ", b)?;
        write!(f, "{:?}}}", self.pt_pscalar())?;
        Ok(())
    }
}
fn s1(a: f32, b: f32) -> f32 {
    a / 2. + b / 2.
}
fn s2(a: f32, b: f32) -> f32 {
    a / 2. - b / 2.
}

impl From<Vector3<f32>> for MatrixG3 {
    fn from(value: Vector3<f32>) -> Self {
        let [a, b, c]: [f32; 3] = value.into();
        let (ze, re, im) = (
            Complex::new(0., 0.),
            Complex::new(1., 0.),
            Complex::new(0., 1.),
        );
        Self(Pauli2::new(ze, re, re, ze)) * a
            + Self(Pauli2::new(ze, -im, im, ze)) * b
            + Self(Pauli2::new(re, ze, ze, -re)) * c
    }
}
impl_num_traits! {
    impl ... for MatrixG3 {
        Add(defo; [self, rhs] => self.binary_map(rhs, Add::add)),
        Sub(defo; [self, rhs] => self.binary_map(rhs, Sub::sub)),
        Mul(defo; [self, rhs] => self.binary_map(rhs, Mul::mul)),
        Neg(defo; [self] => self.unary_map(Neg::neg)),
        Index[(usize, usize)](Complex<f32>; [self, index] => &self.0[index]),
        IndexMut[(usize, usize)]([self, index] => &mut self.0[index]),
        From[Complex<f32>]([value] => Self(Pauli2::identity() * value)),
        From[f32]([value] => Self::from(Complex::from(value))),
        AddAssign(), SubAssign(), MulAssign()
    }
}
impl_num_traits! {
    impl[G: Into<Complex<f32>>] ... for MatrixG3 {
        Mul[G](defo; [self, rhs] => Self(self.0 * rhs.into())),
        Div[G](defo; [self, rhs] => Self(self.0 / rhs.into())),
        MulAssign[G]([self, rhs] => { self.0 *= rhs.into(); }),
        DivAssign[G]([self, rhs] => { self.0 /= rhs.into(); }),
    }
}
