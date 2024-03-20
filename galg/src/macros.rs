#[macro_export]
macro_rules! defer_default_impl {
    (@name empty) => { defer_default_impl!(@fn empty() -> Self); };
    (@name full) => { defer_default_impl!(@fn full() -> Self); };
    (@name new) => { defer_default_impl!(@fn new(index: [usize]) -> Self); };
    (@name size) => { defer_default_impl!(@fn size(self: [&Self]) -> usize); };
    (@name contains) => { defer_default_impl!(@fn contains(self: [&Self], index: [usize]) -> bool); };
    (@name from_iter) => { defer_default_impl!(@fn from_iter(list: [impl Iterator<Item = usize>]) -> Self); };
    (@name iter_elems) => { defer_default_impl!(@fn iter_elems(self: [&Self]) -> impl '_ + Iterator<Item = usize>); };
    (@name conjunction) => { defer_default_impl!(@fn conjunction(self: [Self], with: [Self]) -> Self); };
    (@name disjunction) => { defer_default_impl!(@fn disjunction(self: [Self], with: [Self]) -> Self); };
    (@name symmetric_difference) => { defer_default_impl!(@fn symmetric_difference(self: [Self], with: [Self]) -> Self); };
    (@name relative_complement) => { defer_default_impl!(@fn relative_complement(self: [Self], remove: [Self]) -> Self); };
    (@name complement) => { defer_default_impl!(@fn complement(self: [Self]) -> Self); };
    (@name convert_from) => { defer_default_impl!(@fn convert_from(t: [impl IndexSet<DIM>]) -> Self); };
    (@name iter_grade) => { defer_default_impl!(@fn iter_grade(k: [usize]) -> impl Iterator<Item = Self>); };
    (@fn $n:ident($($a:ident: [$($ty:tt)*]),*) -> $($t:tt)+) => {
        fn $n($($a: $($ty)*),*) -> $($t)* { $crate::subset::default::$n::<DIM, Self>($($a),*) }
    };
    (, $(t:tt)*) => { $($t)* };
    ($n:ident, $($t:tt)*) => {
        defer_default_impl!(@name $n);
        defer_default_impl!($($t)*);
    };
    ($n:ident $($t:tt)*) => {
        defer_default_impl!(@name $n);
        defer_default_impl!($($t)*);
    };
    () => {};
}
#[macro_export]
macro_rules! impl_num_traits {
    (impl $([$($i:tt)*])? ... for $t:ty $(where [$($w:tt)*])? { $($b:tt)* }) => {
        impl_num_traits!(@2 [impl $(<$($i)*>)?] [for $t $(where $($w)*)?] { $($b)* });
    };
    (@2 $t1:tt $t2:tt { $($trait:ident $([$($g:tt)*])? ($($impl:tt)*)),* $(,)*}) => {
        $(
            impl_num_traits!(@3 $t1 [$trait $([$($g)*])?] $t2 $($impl)*);
        )*
    };
    (@3 [$($t1:tt)*] [$trait:ident $([$($g:tt)*])?] [$($t2:tt)*] $($impl:tt)*) => {
        $($t1)* $trait $(<$($g)*>)? $($t2)* {
            impl_num_traits!(@4 $trait $([$($g)*])?: $($impl)*);
        }
    };
    (@4 Add: [$l:tt, $r:tt] => $($impl:tt)*) => {
        fn add($l, $r: Self) -> Self { $($impl)* }
    };
    (@4 Sub: [$l:tt, $r:tt] => $($impl:tt)*) => {
        fn sub($l, $r: Self) -> Self { $($impl)* }
    };
    (@4 Neg: [$l:tt] => $($impl:tt)*) => {
        fn neg($l) -> Self { $($impl)* }
    };
    (@4 Mul [$t:tt] : [$l:tt, $r:tt] => $($impl:tt)*) => {
        fn mul($l, $r: $t) -> Self { $($impl)* }
    };
    (@4 Mul: [$l:tt, $r:tt] => $($impl:tt)*) => {
        fn mul($l, $r: Self) -> Self { $($impl)* }
    };
    (@4 Div [$t:tt] : [$l:tt, $r:tt] => $($impl:tt)*) => {
        fn div($l, $r: $t) -> Self { $($impl)* }
    };
    (@4 Div: [$l:tt, $r:tt] => $($impl:tt)*) => {
        fn div($l, $r: Self) -> Self { $($impl)* }
    };
    (@4 AddAssign: [$l:tt, $r:tt] => $($impl:tt)*) => {
        fn add_assign(&mut $l, $r: Self) { $($impl)* }
    };
    (@4 SubAssign: [$l:tt, $r:tt] => $($impl:tt)*) => {
        fn sub_assign(&mut $l, $r: Self) { $($impl)* }
    };
    (@4 MulAssign [$t:tt]: [$l:tt, $r:tt] => $($impl:tt)*) => {
        fn mul_assign(&mut $l, $r: $t) { $($impl)* }
    };
    (@4 MulAssign: [$l:tt, $r:tt] => $($impl:tt)*) => {
        fn mul_assign(&mut $l, $r: Self) { $($impl)* }
    };
    (@4 DivAssign [$t:tt]: [$l:tt, $r:tt] => $($impl:tt)*) => {
        fn div_assign(&mut $l, $r: $t) { $($impl)* }
    };
    (@4 DivAssign: [$l:tt, $r:tt] => $($impl:tt)*) => {
        fn div_assign(&mut $l, $r: Self) { $($impl)* }
    };
    (@4 AddAssign:) => {
        fn add_assign(&mut self, rhs: Self) { *self = self.clone() + rhs }
    };
    (@4 SubAssign:) => {
        fn sub_assign(&mut self, rhs: Self) { *self = self.clone() - rhs }
    };
    (@4 MulAssign [$t:tt]:) => {
        fn mul_assign(&mut self, rhs: $t) { *self = self.clone() * rhs }
    };
    (@4 MulAssign [$t:tt]:) => {
        fn mul_assign(&mut self, rhs: $t) { *self = self.clone() * rhs }
    };
    (@4 MulAssign:) => {
        fn mul_assign(&mut self, rhs: Self) { *self = self.clone() * rhs }
    };
    (@4 DivAssign [$t:tt]:) => {
        fn div_assign(&mut self, rhs: $t) { *self = self.clone() / rhs }
    };
    (@4 DivAssign [$t:tt]:) => {
        fn div_assign(&mut self, rhs: $t) { *self = self.clone() / rhs }
    };
    (@4 DivAssign:) => {
        fn div_assign(&mut self, rhs: Self) { *self = self.clone() / rhs }
    };
    (@4 RemAssign:) => {
        fn rem_assign(&mut self, rhs: Self) { *self = self.clone() % rhs }
    };
    (@4 Debug: [$l:tt, $r:tt] => $($impl:tt)*) => {
        fn fmt(&$l, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { $($impl)* }
    };
    (@4 Display: [$l:tt, $r:tt] => $($impl:tt)*) => {
        fn fmt(&$l, $r: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { $($impl)* }
    };
    (@4 Index [$t:ty]: [$l:tt, $r:tt] => $($impl:tt)*) => {
        fn index(&$l, $r: $t) -> &Self::Output { $($impl)* }
    };
    (@4 IndexMut [$t:ty]: [$l:tt, $r:tt] => $($impl:tt)*) => {
        fn index_mut(&mut $l, $r: $t) -> &mut Self::Output { $($impl)* }
    };
    (@4 From [$t:ty]: [$l:tt] => $($impl:tt)*) => {
        fn from($l: $t) -> Self { $($impl)* }
    };
    (@4 Default: [] => $($impl:tt)*) => {
        fn default() -> Self { $($impl)* }
    };
    (@4 SimdValue:) => {
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
    };
    (@4 $trait:ident $([$($t:tt)*])?: defo; $($impl:tt)*) => {
        type Output = Self;
        impl_num_traits!(@4 $trait $([$($t)*])?: $($impl)*);
    };
    (@4 $trait:ident $([$($t:tt)*])?: type $i:ident = $x:ty; $($impl:tt)*) => {
        type $i = $x;
        impl_num_traits!(@4 $trait $([$($t)*])?: $($impl)*);
    };
    (@4 $trait:ident $([$($t:tt)*])?: $x:ty; $($impl:tt)*) => {
        type Output = $x;
        impl_num_traits!(@4 $trait $([$($t)*])?: $($impl)*);
    };
}
