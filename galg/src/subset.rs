use std::{
    marker::PhantomData,
    ops::{Add, Mul},
};

use crate::defer_default_impl;

pub mod default {
    use super::{IndexSet, Subbin};

    fn unary_map<const DIM: usize, I: IndexSet<DIM>>(
        a: I,
        f: impl FnOnce(Subbin<DIM>) -> Subbin<DIM>,
    ) -> I {
        I::convert_from(f(Subbin::convert_from(a)))
    }
    fn binary_map<const DIM: usize, I: IndexSet<DIM>>(
        a: I,
        b: I,
        f: impl FnOnce(Subbin<DIM>, Subbin<DIM>) -> Subbin<DIM>,
    ) -> I {
        let a = Subbin::convert_from(a);
        let b = Subbin::convert_from(b);
        I::convert_from(f(a, b))
    }
    pub fn symmetric_difference<const DIM: usize, I: IndexSet<DIM>>(a: I, b: I) -> I {
        binary_map(a, b, |a, b| a.symmetric_difference(b))
    }
    pub fn conjunction<const DIM: usize, I: IndexSet<DIM>>(a: I, b: I) -> I {
        binary_map(a, b, |a, b| a.conjunction(b))
    }
    pub fn disjunction<const DIM: usize, I: IndexSet<DIM>>(a: I, b: I) -> I {
        binary_map(a, b, |a, b| a.disjunction(b))
    }
    pub fn complement<const DIM: usize, I: IndexSet<DIM>>(a: I) -> I {
        //I::full().relative_complement(a)
        unary_map(a, |x| x.complement())
    }
    pub fn iter_elems<const DIM: usize, I: IndexSet<DIM>>(
        a: &I,
    ) -> impl '_ + Iterator<Item = usize> {
        (0..=DIM).filter(move |&i| a.contains(i))
    }
    pub fn from_iter<const DIM: usize, I: IndexSet<DIM>>(a: impl Iterator<Item = usize>) -> I {
        a.map(I::new).fold(I::empty(), I::disjunction)
    }
    pub fn empty<const DIM: usize, I: IndexSet<DIM>>() -> I {
        I::from_iter([0usize; 0].into_iter())
    }
    pub fn new<const DIM: usize, I: IndexSet<DIM>>(index: usize) -> I {
        I::from_iter([index].into_iter())
    }
    pub fn full<const DIM: usize, I: IndexSet<DIM>>() -> I {
        I::from_iter(0..=DIM)
    }
    pub fn size<const DIM: usize, I: IndexSet<DIM>>(a: &I) -> usize {
        a.iter_elems().count()
    }
    pub fn contains<const DIM: usize, I: IndexSet<DIM>>(a: &I, index: usize) -> bool {
        // I::new(i).conjunction(a).size() == 1
        a.iter_elems().any(|x| x == index)
    }
    pub fn relative_complement<const DIM: usize, I: IndexSet<DIM>>(a: I, remove: I) -> I {
        a.conjunction(remove.complement())
    }

    pub fn convert_from<const DIM: usize, I: IndexSet<DIM>>(from: impl IndexSet<DIM>) -> I {
        I::from_iter(from.iter_elems())
    }
    pub fn iter_grade<const DIM: usize, I: IndexSet<DIM>>(k: usize) -> impl Iterator<Item = I> {
        let mut v = vec![0; k];
        (0..super::count_combinations(DIM, k)).map(move |x| {
            super::write_kth_tuple(x, &mut v);
            I::from_iter(v.iter().copied())
        })
    }
}

pub trait IndexSet<const DIM: usize>: Sized + PartialEq + Clone {
    fn empty() -> Self;
    fn full() -> Self;
    fn new(index: usize) -> Self;
    fn size(&self) -> usize;
    fn contains(&self, index: usize) -> bool;
    fn from_iter(list: impl Iterator<Item = usize>) -> Self;
    fn iter_elems(&self) -> impl '_ + Iterator<Item = usize>;
    fn conjunction(self, with: Self) -> Self;
    fn disjunction(self, with: Self) -> Self;
    fn symmetric_difference(self, with: Self) -> Self;
    fn relative_complement(self, remove: Self) -> Self;
    fn complement(self) -> Self;
    fn convert_from(t: impl IndexSet<DIM>) -> Self;
    fn iter_grade(k: usize) -> impl Iterator<Item = Self>;
}
pub trait GradedSpace<const DIM: usize, F>:
    Clone + Add<Self, Output = Self> + Mul<F, Output = Self>
{
    type Index: IndexSet<DIM>;
    fn zero() -> Self;
    fn assign(&mut self, elem: F, i: impl IndexSet<DIM>);
    fn project(&self, i: impl IndexSet<DIM>) -> F;
    fn iter_basis() -> impl Iterator<Item = Self::Index> {
        (0..(DIM + 1)).flat_map(Self::Index::iter_grade)
    }
    fn iter(&self) -> impl Iterator<Item = (F, Self::Index)> {
        Self::iter_basis().map(|i| (self.project(i.clone()), i))
    }
    fn iter2<S: IndexSet<DIM>>(&self) -> impl Iterator<Item = (F, S)> {
        self.iter().map(|(a, b)| (a, S::convert_from(b)))
    }
    fn new(elem: F, i: impl IndexSet<DIM>) -> Self {
        let mut this = Self::zero();
        this.assign(elem, i);
        this
    }
    fn mass_new<S: IndexSet<DIM>, I: IntoIterator<Item = (F, S)>>(elements: I) -> Self {
        elements
            .into_iter()
            .map(|(val, elem)| Self::new(val, elem))
            .fold(Self::zero(), Add::add)
    }
    fn select(&self, element: impl IndexSet<DIM>) -> Self {
        Self::new(self.project(element.clone()), element)
    }
    fn grade_select(&self, k: usize) -> Self {
        self.multi_select(Subbin::iter_grade(k))
    }
    fn multi_select<S: IndexSet<DIM>, I: IntoIterator<Item = S>>(&self, grades: I) -> Self {
        grades
            .into_iter()
            .map(|i| self.select(i))
            .fold(Self::zero(), Add::add)
    }
    fn multi_project<S: IndexSet<DIM>, I: IntoIterator<Item = S>>(
        &self,
        grades: I,
    ) -> impl Iterator<Item = F> {
        grades.into_iter().map(|i| self.project(i))
    }
    fn grade_map(mut self, elem: impl IndexSet<DIM>, f: &impl Fn(F) -> F) -> Self {
        self.assign(f(self.project(elem.clone())), elem);
        self
    }
    fn multi_grade_map<S: IndexSet<DIM>, I: IntoIterator<Item = S>>(
        &self,
        elems: I,
        f: &impl Fn(F) -> F,
    ) -> Self {
        elems
            .into_iter()
            .fold(self.clone(), |acc, elem| acc.grade_map(elem, f))
    }
}

impl<const DIM: usize, const K: usize> IndexSet<DIM> for [usize; K] {
    defer_default_impl!(
        empty,
        full,
        contains,
        symmetric_difference,
        relative_complement,
        convert_from,
        iter_grade,
        conjunction,
        disjunction,
        new
    );
    fn size(&self) -> usize {
        K
    }

    fn complement(self) -> Self {
        if IndexSet::<DIM>::size(&self) * 2 == DIM {
            self.map(|x| DIM - x - 1)
        } else {
            unimplemented!()
        }
    }

    fn from_iter(list: impl Iterator<Item = usize>) -> Self {
        let v: Vec<_> = list.collect();
        Self::try_from(&v[..]).unwrap()
    }

    fn iter_elems(&self) -> impl '_ + Iterator<Item = usize> {
        self.iter().copied()
    }
}
impl<'a, const DIM: usize> IndexSet<DIM> for &'a [usize] {
    defer_default_impl!(
        new,
        full,
        from_iter,
        symmetric_difference,
        relative_complement,
        convert_from,
        iter_grade,
        complement,
        conjunction,
        disjunction,
        contains
    );
    fn size(&self) -> usize {
        self.len()
    }

    fn empty() -> Self {
        &[]
    }

    fn iter_elems(&self) -> impl '_ + Iterator<Item = usize> {
        self.iter().copied()
    }
}

pub fn count_permutations(n: u64, r: u64) -> u64 {
    (n - r + 1..=n).product()
}

pub const fn count_combinations(n: usize, r: usize) -> usize {
    if r > n {
        0
    } else {
        let mut acc = 1;
        let mut val = 1;
        while val <= r {
            acc = acc * (n - val + 1) / val;
            val += 1;
        }
        acc
    }
}
pub const fn write_kth_tuple(n: usize, tuple: &mut [usize]) {
    let k = tuple.len();
    if k == 0 {
        return;
    }
    if k == 1 {
        tuple[0] = n;
        return;
    }
    let mut i = k - 1;
    while count_combinations(i + 1, k) <= n {
        i += 1;
    }
    tuple[k - 1] = i;
    write_kth_tuple(
        n - count_combinations(i, k),
        tuple.split_last_mut().unwrap().1,
    );
}
pub const fn get_tuple_k(tuple: &[usize]) -> usize {
    let k = tuple.len();
    if k == 0 {
        return 0;
    }
    tuple
        .into_iter()
        .zip(1..=k)
        .map(|(&n, k)| count_combinations(n, k))
        .fold(0, Add::add)
}
pub const fn kth_tuple<const K: usize>(n: usize) -> [usize; K] {
    let mut result = [0; K];
    write_kth_tuple(n, &mut result);
    result
}
#[derive(Default, Clone, PartialEq, Eq)]
pub struct Subbin<const DIM: usize>(pub usize, PhantomData<[(); DIM]>);
impl<const DIM: usize> Subbin<DIM> {
    pub fn bits(data: usize) -> Self {
        Self(data, PhantomData)
    }
    pub fn to_bits(&self) -> usize {
        self.0
    }
}
impl<const DIM: usize> std::fmt::Debug for Subbin<DIM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bits = (0..DIM).rev().fold(String::new(), |acc, bit| {
            acc + &format!("{}", self.0 >> bit & 1)
        });
        write!(f, "{{{bits}}}")
    }
}
impl<const DIM: usize> IndexSet<DIM> for Subbin<DIM> {
    defer_default_impl!(new, contains, relative_complement, convert_from, iter_grade);

    fn size(&self) -> usize {
        self.0.count_ones() as usize
    }

    fn from_iter(list: impl Iterator<Item = usize>) -> Self {
        Self(list.map(|x| 1 << x).fold(0, Add::add), PhantomData)
    }

    fn iter_elems(&self) -> impl Iterator<Item = usize> {
        (0..usize::BITS as usize).filter(|i| (self.0 >> i) & 1 == 1)
    }

    fn conjunction(self, other: Self) -> Self {
        Self(self.0 & other.0, PhantomData)
    }

    fn disjunction(self, other: Self) -> Self {
        Self(self.0 | other.0, PhantomData)
    }

    fn complement(self) -> Self {
        Self(!self.0, PhantomData)
    }

    fn empty() -> Self {
        Self(0, PhantomData)
    }

    fn full() -> Self {
        if DIM == 0 {
            return Self::empty();
        }
        Self(usize::MAX >> (usize::BITS as usize - DIM), PhantomData)
    }

    fn symmetric_difference(self, with: Self) -> Self {
        Self(self.0 ^ with.0, PhantomData)
    }
}
