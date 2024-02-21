use std::{marker::PhantomData, ops::Add};

pub trait IndexSubset<const DIM: usize>: Sized + PartialEq {
    fn size(&self) -> usize;
    fn from_elems<'a>(list: &'a [usize]) -> Self;
    fn to_elems(&self, to: &mut [usize]);

    fn conjunction(self, other: &Self) -> Self {
        Self::convert_from(&Subbin::convert_from(&self).conjunction(&Subbin::convert_from(other)))
    }

    fn disjunction(self, other: &Self) -> Self {
        Self::convert_from(&Subbin::convert_from(&self).disjunction(&Subbin::convert_from(other)))
    }

    fn inversion(self) -> Self {
        Self::convert_from(&Subbin::convert_from(&self).inversion())
    }

    fn convert_from<T: IndexSubset<DIM>>(t: &T) -> Self {
        let mut vec = vec![0; t.size()];
        t.to_elems(&mut vec);
        Self::from_elems(&vec)
    }

    fn iter_elems(&self) -> impl Iterator<Item = usize> {
        let mut a = vec![0; self.size()];
        self.to_elems(&mut a);
        a.into_iter()
    }

    fn iter_grade(k: usize) -> impl Iterator<Item = Self> {
        let mut v = vec![0; k];
        (0..count_combinations(DIM, k)).map(move |x| {
            write_kth_tuple(x, &mut v);
            Self::from_elems(&v)
        })
    }
}
pub trait SubsetCollection<const DIM: usize, T>: Default {
    type Index: IndexSubset<DIM>;
    fn assign(&mut self, elem: T, i: &impl IndexSubset<DIM>) -> Option<T>;
    fn project(&self, i: &impl IndexSubset<DIM>) -> Option<T>;
    fn include_other(&mut self, other: &Self) {
        for (elem, i) in other.iter() {
            self.assign(elem, &i);
        }
    }
    fn include_inplace(mut self, other: &Self) -> Self {
        self.include_other(other);
        self
    }
    fn iter(&self) -> impl Iterator<Item = (T, Self::Index)>;
    fn iter2<S: IndexSubset<DIM>>(&self) -> impl Iterator<Item = (T, S)> {
        self.iter().map(|(a, b)| {
            let mut v = vec![0; b.size()];
            b.to_elems(&mut v);
            (a, S::from_elems(&v))
        })
    }
    fn new(elem: T, i: &impl IndexSubset<DIM>) -> Self {
        let mut this = Self::default();
        this.assign(elem, i);
        this
    }
    fn mass_new<'a, S: 'a + IndexSubset<DIM>, I: IntoIterator<Item = (T, &'a S)>>(
        elements: I,
    ) -> Self {
        elements
            .into_iter()
            .map(|(val, elem)| Self::new(val, elem))
            .fold(Self::default(), |acc, val| acc.include_inplace(&val))
    }
    fn select(&self, element: &impl IndexSubset<DIM>) -> Option<Self> {
        self.project(element).map(|x| Self::new(x, element))
    }
    fn select_grade(&self, k: usize) -> Self {
        self.multi_select(Subbin::iter_grade(k))
    }
    fn multi_select<S: IndexSubset<DIM>, I: IntoIterator<Item = S>>(&self, grades: I) -> Self {
        grades
            .into_iter()
            .flat_map(|i| self.select(&i))
            .fold(Self::default(), |a, b| a.include_inplace(&b))
    }
    fn multi_project<S: IndexSubset<DIM>, I: IntoIterator<Item = S>>(
        &self,
        grades: I,
    ) -> impl Iterator<Item = T> {
        grades.into_iter().flat_map(|i| self.project(&i))
    }
    fn grade_map<S: IndexSubset<DIM>>(&self, elem: &S, f: &impl Fn(T) -> T) -> Self {
        Self::mass_new(
            self.iter2::<S>()
                .map(|(a, b)| (if &b == elem { f(a) } else { a }, elem)),
        )
    }
    fn multi_grade_map<S: IndexSubset<DIM>, I: IntoIterator<Item = S>>(
        &mut self,
        elems: I,
        f: &impl Fn(T) -> T,
    ) {
        for i in elems {
            self.grade_map(&i, f);
        }
    }
}
pub trait GradeStorage<const DIM: usize, T: 'static + Copy>:
    Sized + Default + Clone + SubsetCollection<DIM, T>
{
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sublist<const DIM: usize>(Vec<usize>, PhantomData<[(); DIM]>);
impl<const DIM: usize> IndexSubset<DIM> for Sublist<DIM> {
    fn size(&self) -> usize {
        self.0.len()
    }

    fn from_elems(list: &[usize]) -> Self {
        Self(list.to_owned(), PhantomData)
    }

    fn to_elems(&self, to: &mut [usize]) {
        to.copy_from_slice(&self.0)
    }

    fn conjunction(self, other: &Self) -> Self {
        let mut common_elems = Vec::new();
        let mut i = 0;
        let mut j = 0;
        while i < self.0.len() && j < other.0.len() {
            if self.0[i] == other.0[j] {
                common_elems.push(self.0[i]);
                i += 1;
                j += 1;
            } else if self.0[i] < other.0[j] {
                i += 1;
            } else {
                j += 1;
            }
        }
        Self(common_elems, PhantomData)
    }

    fn disjunction(self, other: &Self) -> Self {
        let mut all_elems = Vec::new();
        let mut i = 0;
        let mut j = 0;
        while i < self.0.len() && j < other.0.len() {
            if self.0[i] == other.0[j] {
                all_elems.push(self.0[i]);
                i += 1;
                j += 1;
            } else if self.0[i] < other.0[j] {
                all_elems.push(self.0[i]);
                i += 1;
            } else {
                all_elems.push(other.0[j]);
                j += 1;
            }
        }
        while i < self.0.len() {
            all_elems.push(self.0[i]);
            i += 1;
        }
        while j < other.0.len() {
            all_elems.push(other.0[j]);
            j += 1;
        }
        Self(all_elems, PhantomData)
    }

    fn inversion(self) -> Self {
        let mut inverted_elems = Vec::new();
        let mut j = 0;
        for i in 0..DIM {
            if j < self.0.len() && self.0[j] == i {
                j += 1;
            } else {
                inverted_elems.push(i);
            }
        }
        Self(inverted_elems, PhantomData)
    }
}
impl<const DIM: usize, const K: usize> IndexSubset<DIM> for [usize; K] {
    fn size(&self) -> usize {
        K
    }

    fn from_elems(list: &[usize]) -> Self {
        let mut this = [0; K];
        this.copy_from_slice(list);
        this
    }

    fn to_elems(&self, to: &mut [usize]) {
        to.copy_from_slice(self)
    }

    fn conjunction(self, _: &Self) -> Self {
        unimplemented!()
    }

    fn disjunction(self, _: &Self) -> Self {
        unimplemented!()
    }

    fn inversion(self) -> Self {
        if IndexSubset::<DIM>::size(&self) * 2 == DIM {
            self.map(|x| DIM - x - 1)
        } else {
            unimplemented!()
        }
    }
}
impl<'a, const DIM: usize> IndexSubset<DIM> for &'a [usize] {
    fn size(&self) -> usize {
        self.len()
    }

    fn from_elems(_: &[usize]) -> Self {
        unimplemented!("Cannot create a borrowed slice")
    }

    fn to_elems(&self, to: &mut [usize]) {
        to.copy_from_slice(self)
    }

    fn conjunction(self, _: &Self) -> Self {
        unimplemented!()
    }

    fn disjunction(self, _: &Self) -> Self {
        unimplemented!()
    }

    fn inversion(self) -> Self {
        unimplemented!()
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
    return result;
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subcom<const DIM: usize>(usize, usize, PhantomData<[(); DIM]>);
impl<const DIM: usize> IndexSubset<DIM> for Subcom<DIM> {
    fn size(&self) -> usize {
        self.0
    }

    fn from_elems(list: &[usize]) -> Self {
        Self(list.len(), get_tuple_k(list), PhantomData)
    }

    fn to_elems(&self, to: &mut [usize]) {
        write_kth_tuple(self.1, to)
    }
    fn iter_grade(k: usize) -> impl Iterator<Item = Self> {
        (0..count_combinations(DIM, k)).map(move |i| Self(k, i, PhantomData))
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subbin<const DIM: usize>(pub usize, PhantomData<[(); DIM]>);
impl<const DIM: usize> Subbin<DIM> {
    pub fn bits(data: usize) -> Self {
        Self(data, PhantomData)
    }
    pub fn to_bits(&self) -> usize {
        self.0
    }
}
impl<const DIM: usize> IndexSubset<DIM> for Subbin<DIM> {
    fn size(&self) -> usize {
        self.0.count_ones() as usize
    }

    fn from_elems(list: &[usize]) -> Self {
        Self(list.iter().map(|&x| 1 << x).fold(0, Add::add), PhantomData)
    }

    fn iter_elems(&self) -> impl Iterator<Item = usize> {
        (0..usize::BITS as usize).filter(|i| (self.0 >> i) & 1 == 1)
    }

    fn to_elems(&self, to: &mut [usize]) {
        let mut j = 0;
        for i in IndexSubset::<DIM>::iter_elems(self) {
            to[j] = i;
            j += 1;
        }
    }

    fn conjunction(self, other: &Self) -> Self {
        Self(self.0 & other.0, PhantomData)
    }

    fn disjunction(self, other: &Self) -> Self {
        Self(self.0 | other.0, PhantomData)
    }

    fn inversion(self) -> Self {
        Self(!self.0, PhantomData)
    }
}