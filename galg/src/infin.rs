use core::ops;

use nalgebra::RealField;

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
        FakeCinf {
            f,
            f_prime,
        }
    }
}
impl<F> Cinf<F> for FakeCinf<F> {
    fn nth_derivative(&self, n: usize, val: F) -> F {
        (match n {
            0 => &self.f,
            1 => &self.f_prime,
            _ => panic!("this function does not support derivatives of order {n}")
        })(val)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Infin<const N: usize, F>([F; N]);
impl<const N: usize, F: RealField + Copy> Infin<N, F> {
    pub fn new(val: [F; N]) -> Self {
        Self(val)
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
}
impl<const N: usize, F: RealField + Copy> ops::Add for Infin<N, F> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..N {
            self.0[i] += rhs.0[i];
        }
        self
    }
}
impl<const N: usize, F: RealField + Copy> ops::Sub for Infin<N, F> {
    type Output = Self;
    
    fn sub(mut self, rhs: Self) -> Self::Output {
        for i in 0..N {
            self.0[i] += rhs.0[i];
        }
        self
    }
}
impl<const N: usize, F: RealField + Copy> ops::Mul for Infin<N, F> {
    type Output = Self;
    
    fn mul(self, rhs: Self) -> Self::Output {
        let mut out = [F::zero(); N];
        for i in 1..N {
            out[i] = self[0] * rhs[i] + self[i] * rhs[0];
        }
        Self(out)
    }
}
impl<const N: usize, F> ops::Index<usize> for Infin<N, F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl<const N: usize, F> ops::IndexMut<usize> for Infin<N, F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
impl<const N: usize, F: RealField + Copy> PartialEq for Infin<N, F> {
    fn eq(&self, other: &Self) -> bool {
        self[0] == other[0]
    }
}
impl<const N: usize, F: RealField + Copy> PartialOrd for Infin<N, F> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self[0].partial_cmp(&other[0])
    }
}