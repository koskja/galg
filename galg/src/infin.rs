use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Rem, RemAssign, Sub,
    SubAssign,
};

use approx::RelativeEq;
use nalgebra::{ComplexField, Field, RealField, SimdValue};
use num_traits::{FromPrimitive, Num, One, Signed, Zero};
use simba::scalar::{SubsetOf, SupersetOf};

use crate::{impl_num_traits, variable::RealFunction};

impl<F: RealField + Copy> Cinf<F> for RealFunction {
    fn nth_derivative(&self, n: usize, val: F) -> F {
        match self {
            RealFunction::Sin => match n % 4 {
                0 => val.sin(),
                1 => val.cos(),
                2 => -val.sin(),
                3 => -val.cos(),
                _ => unreachable!(),
            },
            RealFunction::Cos => match n % 4 {
                0 => val.cos(),
                1 => -val.sin(),
                2 => -val.cos(),
                3 => val.sin(),
                _ => unreachable!(),
            },
            RealFunction::Tan => match n {
                0 => val.tan(),
                1 => val.cos().powi(-2),
                2 => F::from_subset(&2.) * val.tan() * val.cos().powi(-2),
                _ => unimplemented!(),
            },
            RealFunction::Sinh => match n % 2 {
                0 => val.sinh(),
                1 => val.cosh(),
                _ => unimplemented!(),
            },
            RealFunction::Cosh => match n % 2 {
                0 => val.cosh(),
                1 => val.sinh(),
                _ => unimplemented!(),
            },
            RealFunction::Tanh => match n {
                0 => val.tanh(),
                1 => val.cos().powi(-2),
                2 => -F::from_subset(&2.) * val.tanh() * val.cosh().powi(-2),
                _ => unimplemented!(),
            },
            RealFunction::Csc => match n {
                0 => val.sin().powi(-1),
                1 => val.cos().powi(-2),
                2 => -F::from_subset(&2.) * val.tanh() * val.cosh().powi(-2),
                _ => unimplemented!(),
            },
            RealFunction::Sec => match n {
                0 => val.tanh(),
                1 => val.cos() * val.sin().powi(-2),
                2 => -F::from_subset(&2.) * val.tanh() * val.cosh().powi(-2),
                _ => unimplemented!(),
            },
            RealFunction::Cot => match n {
                0 => val.tanh(),
                1 => val.cos().powi(-2),
                2 => val.sin().powi(-1) * ((val.cos() / val.sin()).powi(2)),
                _ => unimplemented!(),
            },
            RealFunction::Exp => val.exp(),
            RealFunction::Ln => match n {
                0 => val.ln(),
                1 => F::one() / val,
                _ => unimplemented!(),
            },
            RealFunction::Abs => match n {
                0 => val.abs(),
                1 => val.signum(),
                _ => F::zero(),
            },
            RealFunction::Sign => match n {
                0 => val.signum(),
                _ => F::zero(),
            },
            &RealFunction::Powi(i) => {
                let mul = ((i - (n as i32) + 1)..=i)
                    .map(|x| F::from_subset(&(x as f64)))
                    .fold(F::one(), Mul::mul);
                if i >= 0 && n as i32 > i {
                    F::zero()
                } else {
                    mul * val.powi(i - n as i32)
                }
            }
            RealFunction::Expression(_) => todo!(),
        }
    }
}

pub trait Cinf<F> {
    fn apply(&self, val: F) -> F {
        self.nth_derivative(0, val)
    }
    fn nth_derivative(&self, n: usize, val: F) -> F;
}
pub type BoxDynFunction<F> = Box<dyn Fn(F) -> F>;
pub struct FakeCinf<F> {
    f: BoxDynFunction<F>,
    f_prime: BoxDynFunction<F>,
}
impl<F> FakeCinf<F> {
    pub fn new(f: BoxDynFunction<F>, f_prime: BoxDynFunction<F>) -> Self {
        FakeCinf { f, f_prime }
    }
}
impl<F> Cinf<F> for FakeCinf<F> {
    fn nth_derivative(&self, n: usize, val: F) -> F {
        (match n {
            0 => &self.f,
            1 => &self.f_prime,
            _ => panic!("this function does not support derivatives of order {n}"),
        })(val)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Infin<const N: usize, F>([F; N]);
impl<const N: usize, F: RealField + Copy> Infin<N, F> {
    pub fn new(val: [F; N]) -> Self {
        Self(val)
    }
    pub fn nconst(val: F) -> Self {
        let mut a = [F::zero(); N];
        a[0] = val;
        Self(a)
    }
    pub fn apply_fn<T: Cinf<F>>(self, f: T) -> Self {
        let mut res = Self::new([F::zero(); N]);
        res[0] = f.apply(self[0]);
        let a = f.nth_derivative(1, self[0]);
        for i in 1..N {
            res[i] = a * self[i];
        }
        res
    }
    fn map1(mut self, f: &impl Fn(F) -> F) -> Self {
        for i in &mut self.0 {
            *i = f(*i)
        }
        self
    }
    fn map2(mut self, rhs: Self, f: &impl Fn(F, F) -> F) -> Self {
        for (i, j) in (&mut self.0).into_iter().zip(rhs.0.into_iter()) {
            *i = f(*i, j)
        }
        self
    }
}
impl_num_traits! {
    impl[const N: usize, F: Copy + RealField] ... for Infin<N, F> {
        Add(defo; [self, rhs] => self.map2(rhs, &Add::add)),
        Sub(defo; [self, rhs] => self.map2(rhs, &Sub::sub)),
        Mul(defo; [self, rhs] => {
            let mut out = [F::zero(); N];
            out[0] = self[0] * rhs[0];
            for i in 1..N {
                out[i] = self[0] * rhs[i] + self[i] * rhs[0];
            }
            Self(out)
        }),
        Div(defo; [self, rhs] => {
            self * rhs.apply_fn(RealFunction::Powi(-1))
        }),
        Rem(defo; [self, _] => todo!()),
        Neg(defo; [self] => self.map1(&Neg::neg)),
        Mul[F](defo; [self, rhs] => self.map1(&|x| x * rhs)),
        Div[F](defo; [self, rhs] => self.map1(&|x| x / rhs)),
        Index[usize](F; [self, index] => &self.0[index]),
        IndexMut[usize]([self, index] => &mut self.0[index]),
        AddAssign(), SubAssign(), MulAssign[F](), DivAssign[F](), RemAssign(), MulAssign(), DivAssign(),
        SubsetOf[Infin<N, F>](), SimdValue()
    }
}
impl<const N: usize, F: RealField + Copy> SupersetOf<f64> for Infin<N, F> {
    fn is_in_subset(&self) -> bool {
        true
    }

    fn to_subset_unchecked(&self) -> f64 {
        self[0].to_subset_unchecked()
    }

    fn from_subset(element: &f64) -> Self {
        Self::nconst(F::from_subset(element))
    }
}
impl<const N: usize, F: RealField + Copy> PartialOrd for Infin<N, F> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self[0].partial_cmp(&other[0])
    }
}
impl<const N: usize, F: RealField + Copy> PartialEq for Infin<N, F> {
    fn eq(&self, other: &Self) -> bool {
        self[0].eq(&other[0])
    }
}
impl<const N: usize, F: RealField + Copy> approx::AbsDiffEq for Infin<N, F> {
    type Epsilon = Self;

    fn default_epsilon() -> Self::Epsilon {
        Self::nconst(F::default_epsilon())
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self[0].abs_diff_eq(&other[0], epsilon[0])
    }
}
impl<const N: usize, F: RealField + Copy> approx::UlpsEq for Infin<N, F> {
    fn default_max_ulps() -> u32 {
        F::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, eps: Self::Epsilon, ulps: u32) -> bool {
        self[0].ulps_eq(&other[0], eps[0], ulps)
    }
}
impl<const N: usize, F: RealField + Copy> FromPrimitive for Infin<N, F> {
    fn from_i64(n: i64) -> Option<Self> {
        F::from_i64(n).map(Self::nconst)
    }

    fn from_u64(n: u64) -> Option<Self> {
        F::from_u64(n).map(Self::nconst)
    }
}
impl<const N: usize, F: RealField + Copy> Signed for Infin<N, F> {
    fn abs(&self) -> Self {
        self.apply_fn(RealFunction::Abs)
    }

    fn abs_sub(&self, other: &Self) -> Self {
        (*self - *other).abs()
    }

    fn signum(&self) -> Self {
        self.apply_fn(RealFunction::Sign)
    }

    fn is_positive(&self) -> bool {
        self.signum()[0] > F::zero()
    }

    fn is_negative(&self) -> bool {
        self.signum()[0] < F::zero()
    }
}
impl<const N: usize, F: RealField + Copy> One for Infin<N, F> {
    fn one() -> Self {
        Self::nconst(F::one())
    }
}
impl<const N: usize, F: RealField + Copy> Zero for Infin<N, F> {
    fn zero() -> Self {
        Self::nconst(F::zero())
    }

    fn is_zero(&self) -> bool {
        self[0] == F::zero()
    }
}
impl<const N: usize, F: RealField + Copy> Num for Infin<N, F> {
    type FromStrRadixErr = <F as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        F::from_str_radix(str, radix).map(Self::nconst)
    }
}
impl<const N: usize, F: RealField + Copy> RelativeEq for Infin<N, F> {
    fn default_max_relative() -> Self::Epsilon {
        Self::nconst(F::default_max_relative())
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self[0].relative_eq(&other[0], epsilon[0], max_relative[0])
    }
}
impl<const N: usize, F: RealField + Copy> Field for Infin<N, F> {}
impl<const N: usize, F: RealField + Copy> core::fmt::Display for Infin<N, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self[0])
    }
}
impl<const N: usize, F: RealField + Copy> RealField for Infin<N, F> {
    fn is_sign_positive(&self) -> bool {
        self.is_positive()
    }

    fn is_sign_negative(&self) -> bool {
        self.is_negative()
    }

    fn copysign(self, sign: Self) -> Self {
        self.abs() * sign.abs()
    }

    fn max(self, _: Self) -> Self {
        todo!()
    }

    fn min(self, _: Self) -> Self {
        todo!()
    }

    fn clamp(self, _: Self, _: Self) -> Self {
        todo!()
    }

    fn atan2(self, _: Self) -> Self {
        todo!()
    }

    fn min_value() -> Option<Self> {
        todo!()
    }

    fn max_value() -> Option<Self> {
        todo!()
    }

    fn pi() -> Self {
        todo!()
    }

    fn two_pi() -> Self {
        todo!()
    }

    fn frac_pi_2() -> Self {
        todo!()
    }

    fn frac_pi_3() -> Self {
        todo!()
    }

    fn frac_pi_4() -> Self {
        todo!()
    }

    fn frac_pi_6() -> Self {
        todo!()
    }

    fn frac_pi_8() -> Self {
        todo!()
    }

    fn frac_1_pi() -> Self {
        todo!()
    }

    fn frac_2_pi() -> Self {
        todo!()
    }

    fn frac_2_sqrt_pi() -> Self {
        todo!()
    }

    fn e() -> Self {
        todo!()
    }

    fn log2_e() -> Self {
        todo!()
    }

    fn log10_e() -> Self {
        todo!()
    }

    fn ln_2() -> Self {
        todo!()
    }

    fn ln_10() -> Self {
        todo!()
    }
}
impl<const N: usize, F: RealField + Copy> ComplexField for Infin<N, F> {
    type RealField = Self;

    #[inline]
    fn from_real(re: Self::RealField) -> Self {
        re
    }

    #[inline]
    fn real(self) -> Self::RealField {
        self
    }

    #[inline]
    fn imaginary(self) -> Self::RealField {
        Self::zero()
    }

    #[inline]
    fn norm1(self) -> Self::RealField {
        self.abs()
    }

    #[inline]
    fn modulus(self) -> Self::RealField {
        self.abs()
    }

    #[inline]
    fn modulus_squared(self) -> Self::RealField {
        self * self
    }

    #[inline]
    fn argument(self) -> Self::RealField {
        if self >= Self::zero() {
            Self::zero()
        } else {
            Self::pi()
        }
    }

    #[inline]
    fn to_exp(self) -> (Self, Self) {
        if self >= Self::zero() {
            (self, Self::one())
        } else {
            (-self, -Self::one())
        }
    }

    #[inline]
    fn recip(self) -> Self {
        self.powi(-1)
    }

    #[inline]
    fn conjugate(self) -> Self {
        self
    }

    #[inline]
    fn scale(self, factor: Self::RealField) -> Self {
        self * factor
    }

    #[inline]
    fn unscale(self, factor: Self::RealField) -> Self {
        self / factor
    }

    #[inline]
    fn floor(self) -> Self {
        todo!()
    }

    #[inline]
    fn ceil(self) -> Self {
        todo!()
    }

    #[inline]
    fn round(self) -> Self {
        todo!()
    }

    #[inline]
    fn trunc(self) -> Self {
        todo!()
    }

    #[inline]
    fn fract(self) -> Self {
        todo!()
    }

    #[inline]
    fn abs(self) -> Self {
        Signed::abs(&self)
    }

    #[inline]
    fn signum(self) -> Self {
        Signed::signum(&self)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        self.apply_fn(RealFunction::Powi(n))
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        (self.ln() * n).exp()
    }

    #[inline]
    fn powc(self, n: Self) -> Self {
        self.powf(n)
    }

    #[inline]
    fn sqrt(self) -> Self {
        self.powf(Self::nconst(F::one() / (F::one() + F::one())))
    }

    #[inline]
    fn try_sqrt(self) -> Option<Self> {
        if self >= Self::zero() {
            Some(self.sqrt())
        } else {
            None
        }
    }

    #[inline]
    fn exp(self) -> Self {
        self.apply_fn(RealFunction::Exp)
    }

    #[inline]
    fn exp2(self) -> Self {
        (self * Self::ln_2()).exp()
    }

    #[inline]
    fn exp_m1(self) -> Self {
        todo!()
    }

    #[inline]
    fn ln_1p(self) -> Self {
        (self + Self::nconst(F::one())).ln()
    }

    #[inline]
    fn ln(self) -> Self {
        self.apply_fn(RealFunction::Ln)
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    #[inline]
    fn log2(self) -> Self {
        self.log(Self::nconst(F::one() + F::one()))
    }

    #[inline]
    fn log10(self) -> Self {
        self.log(Self::nconst(F::from_subset(&10.)))
    }

    #[inline]
    fn cbrt(self) -> Self {
        todo!()
    }

    #[inline]
    fn hypot(self, other: Self) -> Self::RealField {
        (self.conjugate() * self + other.conjugate() * other).sqrt()
    }

    #[inline]
    fn sin(self) -> Self {
        self.apply_fn(RealFunction::Sin)
    }

    #[inline]
    fn cos(self) -> Self {
        self.apply_fn(RealFunction::Cos)
    }

    #[inline]
    fn tan(self) -> Self {
        self.apply_fn(RealFunction::Tan)
    }

    #[inline]
    fn asin(self) -> Self {
        todo!()
    }

    #[inline]
    fn acos(self) -> Self {
        todo!()
    }

    #[inline]
    fn atan(self) -> Self {
        todo!()
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    //            #[inline]
    //            fn exp_m1(self) -> Self {
    //                $libm::exp_m1(self)
    //            }
    //
    //            #[inline]
    //            fn ln_1p(self) -> Self {
    //                $libm::ln_1p(self)
    //            }
    //
    #[inline]
    fn sinh(self) -> Self {
        self.apply_fn(RealFunction::Sinh)
    }

    #[inline]
    fn cosh(self) -> Self {
        self.apply_fn(RealFunction::Cosh)
    }

    #[inline]
    fn tanh(self) -> Self {
        todo!()
    }

    #[inline]
    fn asinh(self) -> Self {
        todo!()
    }

    #[inline]
    fn acosh(self) -> Self {
        todo!()
    }

    #[inline]
    fn atanh(self) -> Self {
        todo!()
    }

    #[inline]
    fn is_finite(&self) -> bool {
        true
    }
}
