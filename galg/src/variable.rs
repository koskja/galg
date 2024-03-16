use std::{
    cell::RefCell,
    cmp,
    ops::{
        Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Rem, RemAssign, Sub,
        SubAssign,
    },
};

use approx::{AbsDiffEq, RelativeEq};
use lazy_static::lazy_static;
use nalgebra::{ComplexField, Field, RealField, SimdValue};
use num_traits::{FromPrimitive, Num, One, Signed, Zero};
use pyo3::{
    exceptions,
    types::{IntoPyDict, PyDict},
    PyAny, PyErr, PyObject, PyResult, Python, ToPyObject,
};
use simba::scalar::SubsetOf;

use std::sync::Once;

static INIT: Once = Once::new();
static mut VALUE: Option<RefCell<ExprStorage>> = None;

fn get_storage() -> &'static RefCell<ExprStorage> {
    unsafe {
        INIT.call_once(|| {
            VALUE = Some(RefCell::new(ExprStorage::new()));
        });
        VALUE.as_ref().unwrap()
    }
}

pub struct ExprStorage {
    blocks: Vec<Vec<ExprVal>>,
    len: usize,
}
impl ExprStorage {
    pub fn new() -> Self {
        Self {
            blocks: vec![],
            len: 0,
        }
    }
    pub fn push_expr(&mut self, expr: ExprVal) -> Expr {
        if self
            .blocks
            .last()
            .map(|x| x.capacity() - x.len())
            .unwrap_or(0)
            == 0
        {
            self.add_block();
        }
        self.blocks
            .last_mut()
            .as_mut()
            .unwrap()
            .push_within_capacity(expr)
            .unwrap();
        self.len += 1;
        Expr(self.len - 1)
    }
    pub fn add_block(&mut self) {
        let nlen = self.len.max(1);
        self.blocks.push(Vec::with_capacity(nlen))
    }
}
impl Index<usize> for ExprStorage {
    type Output = ExprVal;

    fn index(&self, mut index: usize) -> &Self::Output {
        assert!(index < self.len);
        let mut i = 0;
        while index >= self.blocks[i].len() {
            index -= self.blocks[i].len();
            i += 1;
        }
        &self.blocks[i][index]
    }
}
impl IndexMut<usize> for ExprStorage {
    fn index_mut(&mut self, mut index: usize) -> &mut Self::Output {
        assert!(index < self.len);
        let mut i = 0;
        while index >= self.blocks[i].len() {
            index -= self.blocks[i].len();
            i += 1;
        }
        &mut self.blocks[i][index]
    }
}

pub fn rnexpr(e: ExprVal) -> Expr {
    get_storage().borrow_mut().push_expr(e)
}
pub fn nexpr(e: ExprVal) -> Expr {
    let a = get_storage().borrow_mut().push_expr(e);
    optimise(a).unwrap();
    a
}
pub fn expr(i: Expr) -> ExprVal {
    get_storage().borrow()[i.0]
}
pub fn optimise(i: Expr) -> PyResult<()> {
    Python::with_gil(|py| {
        let new = Expr::from_string(py, &i.to_string(py)?)?;
        let mut storage = get_storage().borrow_mut();
        storage[i.0] = storage[new.0];
        Ok(())
    })
}
#[derive(Debug, Clone, Copy)]
pub enum RealFunction {
    Sin,
    Cos,
    Tan,
    Csc,
    Sec,
    Cot,
    Exp,
    Ln,
    Abs,
    Sign,
    Expression(Expr),
}
#[derive(Debug, Clone, Copy)]
pub enum ExprVal {
    Constant(f32),
    Variable(&'static str),
    Add(Expr, Expr),
    Sub(Expr, Expr),
    Mul(Expr, Expr),
    Div(Expr, Expr),
    Power(Expr, Expr),
    Unary(RealFunction, Expr),
}
impl Expr {
    pub fn nvar(name: &str) -> Self {
        let name: String = name.to_owned();
        rnexpr(ExprVal::Variable(Box::leak(name.into_boxed_str())))
    }
    pub fn nconst(value: f32) -> Self {
        rnexpr(ExprVal::Constant(value))
    }
    pub fn from_sympy_expr(py: Python, sympy_expr: &PyAny) -> PyResult<Self> {
        let bin_args = |f: &dyn Fn(Self, Self) -> ExprVal| -> PyResult<Self> {
            let args: Vec<PyObject> = sympy_expr.getattr("args")?.extract()?;
            let e1 = Self::from_sympy_expr(py, args[0].as_ref(py))?;
            let e2 = Self::from_sympy_expr(py, args[1].as_ref(py))?;
            Ok(rnexpr(f(e1, e2)))
        };
        let unary_arg = |f: &dyn Fn(Self) -> ExprVal| -> PyResult<Self> {
            let args: Vec<PyObject> = sympy_expr.getattr("args")?.extract()?;
            let e1 = Self::from_sympy_expr(py, args[0].as_ref(py))?;
            Ok(rnexpr(f(e1)))
        };
        let expr = match sympy_expr
            .get_type()
            .getattr("__name__")?
            .extract::<&str>()?
        {
            "Add" => bin_args(&|a, b| ExprVal::Add(a, b))?,
            "Sub" => bin_args(&|a, b| ExprVal::Sub(a, b))?,
            "Div" => bin_args(&|a, b| ExprVal::Div(a, b))?,
            "Mul" => bin_args(&|a, b| ExprVal::Mul(a, b))?,
            "Pow" => bin_args(&|a, b| ExprVal::Power(a, b))?,
            "sin" => unary_arg(&|a| ExprVal::Unary(RealFunction::Sin, a))?,
            "cos" => unary_arg(&|a| ExprVal::Unary(RealFunction::Cos, a))?,
            "tan" => unary_arg(&|a| ExprVal::Unary(RealFunction::Tan, a))?,
            "csc" => unary_arg(&|a| ExprVal::Unary(RealFunction::Csc, a))?,
            "sec" => unary_arg(&|a| ExprVal::Unary(RealFunction::Sec, a))?,
            "cot" => unary_arg(&|a| ExprVal::Unary(RealFunction::Cot, a))?,
            "Symbol" => {
                let name: String = sympy_expr.getattr("name")?.extract()?;
                Self::nvar(&name)
            }
            "Number" | "Integer" | "NegativeOne" | "Zero" | "One" | "Rational" | "Float"
            | "Half" => {
                let value: f32 = sympy_expr.extract()?;
                Self::nconst(value)
            }
            s => {
                return Err(PyErr::new::<exceptions::PyValueError, &'static str>(
                    Box::leak(format!("Unsupported sympy expression `{s}`").into_boxed_str()),
                ))
            }
        };
        Ok(expr)
    }
    pub fn to_string(&self, py: Python) -> PyResult<String> {
        Ok(match expr(*self) {
            ExprVal::Constant(c) => c.to_string(),
            ExprVal::Variable(v) => v.to_string(),
            ExprVal::Add(e1, e2) => format!("({}) + ({})", e1.to_string(py)?, e2.to_string(py)?),
            ExprVal::Sub(e1, e2) => format!("({}) - ({})", e1.to_string(py)?, e2.to_string(py)?),
            ExprVal::Mul(e1, e2) => format!("({}) * ({})", e1.to_string(py)?, e2.to_string(py)?),
            ExprVal::Div(e1, e2) => format!("({}) / ({})", e1.to_string(py)?, e2.to_string(py)?),
            ExprVal::Power(e1, e2) => format!("({}) ** ({})", e1.to_string(py)?, e2.to_string(py)?),
            ExprVal::Unary(op, x) => {
                let op_str = match op {
                    RealFunction::Sin => "sin",
                    RealFunction::Cos => "cos",
                    RealFunction::Tan => "tan",
                    RealFunction::Csc => "csc",
                    RealFunction::Sec => "sec",
                    RealFunction::Cot => "cot",
                    RealFunction::Exp => "exp",
                    RealFunction::Ln => "log",
                    RealFunction::Abs => "abs",
                    RealFunction::Expression(_) => unimplemented!(),
                    RealFunction::Sign => "sign",
                };
                format!("{}({})", op_str, x.to_string(py)?)
            }
        })
    }
    pub fn to_sympy(&self, py: Python) -> PyResult<String> {
        let code = format!("str(sympy.simplify('{}'))", self.to_string(py)?);
        let result: String = geval(py, &code, None)?.extract()?;
        Ok(result)
    }
    pub fn to_object(&self, py: Python) -> PyResult<PyObject> {
        Self::sympy_from_expr(py, &self.to_sympy(py)?)
    }
    pub fn from_string(py: Python, expr_str: &str) -> PyResult<Self> {
        let obj = Self::sympy_from_expr(py, expr_str)?;
        Self::from_sympy_expr(py, obj.as_ref(py))
    }
    pub fn sympy_from_expr(py: Python, expr_str: &str) -> PyResult<PyObject> {
        let code = format!("sympy.simplify('{expr_str}')");
        geval(py, &code, None).map(|x| x.to_object(py))
    }
}

fn module_globals() -> PyObject {
    let list = vec!["sympy"];
    Python::with_gil(|py| {
        let a: Vec<_> = list
            .into_iter()
            .map(|x| (x, py.import(x).expect(&format!("Failed to import {x}"))))
            .collect();
        a.into_py_dict(py).to_object(py)
    })
}

lazy_static! {
    static ref GLOBALS: PyObject = module_globals();
}

fn geval<'py>(py: Python<'py>, code: &str, locals: Option<&PyDict>) -> PyResult<&'py PyAny> {
    py.eval(code, locals, Some(GLOBALS.as_ref(py).downcast().unwrap()))
}

#[derive(Debug, Clone, Copy)]
pub struct Expr(usize);

impl Add for Expr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            return self;
        }
        if self.is_zero() {
            return rhs;
        }
        nexpr(ExprVal::Add(self, rhs))
    }
}

impl Sub for Expr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            return self;
        }
        if self.is_zero() {
            return -rhs;
        }
        nexpr(ExprVal::Sub(self, rhs))
    }
}

impl Mul for Expr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero() {
            return self;
        }
        if rhs.is_zero() {
            return rhs;
        }
        if self.is_one() {
            return rhs;
        }
        if rhs.is_one() {
            return self;
        }
        nexpr(ExprVal::Mul(self, rhs))
    }
}

impl Div for Expr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        if self.is_zero() && !rhs.is_zero() {
            return self;
        }
        nexpr(ExprVal::Div(self, rhs))
    }
}

impl Neg for Expr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.is_zero() {
            return self;
        }
        nexpr(ExprVal::Mul(nexpr(ExprVal::Constant(-1.0)), self))
    }
}

impl SimdValue for Expr {
    type Element = Self;
    type SimdBool = bool;

    #[inline(always)]
    fn lanes() -> usize {
        1
    }

    #[inline(always)]
    fn splat(val: Self::Element) -> Self {
        val
    }

    #[inline(always)]
    fn extract(&self, _: usize) -> Self::Element {
        *self
    }

    #[inline(always)]
    unsafe fn extract_unchecked(&self, _: usize) -> Self::Element {
        *self
    }

    #[inline(always)]
    fn replace(&mut self, _: usize, val: Self::Element) {
        *self = val
    }

    #[inline(always)]
    unsafe fn replace_unchecked(&mut self, _: usize, val: Self::Element) {
        *self = val
    }

    #[inline(always)]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        if cond {
            self
        } else {
            other
        }
    }
}
impl Rem for Expr {
    type Output = Self;

    fn rem(self, _: Self) -> Self::Output {
        todo!()
    }
}
impl PartialOrd for Expr {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self == other {
            return Some(cmp::Ordering::Equal);
        }
        Python::with_gil(|py| -> PyResult<_> {
            let obj = (*self - *other).to_object(py)?;
            let locals = PyDict::new(py);
            locals.set_item("obj", obj)?;
            locals.set_item("zero", py.eval("0", None, None)?)?;
            let result: bool = geval(py, "obj < zero", Some(locals))?.extract()?;
            Ok(result)
        })
        .ok()
        .map(|x| {
            if x {
                cmp::Ordering::Less
            } else {
                cmp::Ordering::Greater
            }
        })
    }
}
impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        (*self - *other).is_zero()
    }
}
pub trait RealStuff: Field + PartialEq + PartialOrd + FromPrimitive {}
impl approx::AbsDiffEq for Expr {
    type Epsilon = Self;

    fn default_epsilon() -> Self::Epsilon {
        Self::nconst(f32::EPSILON)
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        (*self - *other).abs() < epsilon
    }
}
impl approx::UlpsEq for Expr {
    fn default_max_ulps() -> u32 {
        todo!()
    }

    fn ulps_eq(&self, _: &Self, _: Self::Epsilon, _: u32) -> bool {
        todo!()
    }
}
impl SubsetOf<Expr> for Expr {
    fn to_superset(&self) -> Self {
        *self
    }

    fn from_superset_unchecked(element: &Self) -> Self {
        *element
    }

    fn is_in_subset(_: &Self) -> bool {
        true
    }
}
impl FromPrimitive for Expr {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self::nconst(n as f32))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self::nconst(n as f32))
    }
}
impl Signed for Expr {
    fn abs(&self) -> Self {
        nexpr(ExprVal::Unary(RealFunction::Abs, *self))
    }

    fn abs_sub(&self, other: &Self) -> Self {
        (*self - *other).abs()
    }

    fn signum(&self) -> Self {
        nexpr(ExprVal::Unary(RealFunction::Sign, *self))
    }

    fn is_positive(&self) -> bool {
        self.signum() > Self::nconst(0.)
    }

    fn is_negative(&self) -> bool {
        self.signum() < Self::nconst(0.)
    }
}
impl One for Expr {
    fn one() -> Self {
        Self::nconst(1.)
    }
}
impl Zero for Expr {
    fn zero() -> Self {
        Self::nconst(0.)
    }

    fn is_zero(&self) -> bool {
        if matches!(expr(*self), ExprVal::Constant(0.)) {
            return true;
        }
        Python::with_gil(|py| -> PyResult<_> {
            let locals = PyDict::new(py);
            locals.set_item("obj", self.to_object(py)?)?;
            locals.set_item("zero", py.eval("0", None, None)?)?;
            let result: bool = geval(py, "obj == zero", Some(locals))?.extract()?;
            Ok(result)
        })
        .unwrap()
    }
}
impl Num for Expr {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f32::from_str_radix(str, radix)
            .map(ExprVal::Constant)
            .map(nexpr)
    }
}
impl RelativeEq for Expr {
    fn default_max_relative() -> Self::Epsilon {
        Self::nconst(f32::EPSILON)
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        if self.abs_diff_eq(other, epsilon) {
            return true;
        }

        let abs_self = Signed::abs(self);
        let abs_other = Signed::abs(other);
        let largest = Self::max(abs_other, abs_self);
        Signed::abs_sub(self, other) <= largest * max_relative
    }
}
impl AddAssign for Expr {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}
impl SubAssign for Expr {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl MulAssign for Expr {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl DivAssign for Expr {
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

impl RemAssign for Expr {
    fn rem_assign(&mut self, other: Self) {
        *self = *self % other;
    }
}
impl Field for Expr {}
impl core::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Python::with_gil(|py| self.to_sympy(py).unwrap()))
    }
}
impl SubsetOf<Expr> for f64 {
    fn to_superset(&self) -> Expr {
        Expr::nconst(*self as f32)
    }

    fn from_superset_unchecked(_: &Expr) -> Self {
        todo!()
    }

    fn is_in_subset(_: &Expr) -> bool {
        true
    }
}
impl RealField for Expr {
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
impl ComplexField for Expr {
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
        nexpr(ExprVal::Power(self, Self::nconst(n as f32)))
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        nexpr(ExprVal::Power(self, n))
    }

    #[inline]
    fn powc(self, n: Self) -> Self {
        self.powf(n)
    }

    #[inline]
    fn sqrt(self) -> Self {
        nexpr(ExprVal::Power(self, Self::nconst(0.5)))
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
        nexpr(ExprVal::Unary(RealFunction::Exp, self))
    }

    #[inline]
    fn exp2(self) -> Self {
        Self::nconst(2.).powf(self)
    }

    #[inline]
    fn exp_m1(self) -> Self {
        nexpr(ExprVal::Unary(RealFunction::Exp, self))
    }

    #[inline]
    fn ln_1p(self) -> Self {
        (self + Self::nconst(1.)).ln()
    }

    #[inline]
    fn ln(self) -> Self {
        nexpr(ExprVal::Unary(RealFunction::Ln, self))
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    #[inline]
    fn log2(self) -> Self {
        self.log(Self::nconst(2.))
    }

    #[inline]
    fn log10(self) -> Self {
        self.log(Self::nconst(10.))
    }

    #[inline]
    fn cbrt(self) -> Self {
        nexpr(ExprVal::Power(self, Self::nconst(1. / 3.)))
    }

    #[inline]
    fn hypot(self, other: Self) -> Self::RealField {
        (self.conjugate() * self + other.conjugate() * other).sqrt()
    }

    #[inline]
    fn sin(self) -> Self {
        nexpr(ExprVal::Unary(RealFunction::Sin, self))
    }

    #[inline]
    fn cos(self) -> Self {
        nexpr(ExprVal::Unary(RealFunction::Cos, self))
    }

    #[inline]
    fn tan(self) -> Self {
        nexpr(ExprVal::Unary(RealFunction::Tan, self))
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
        todo!()
    }

    #[inline]
    fn cosh(self) -> Self {
        todo!()
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
