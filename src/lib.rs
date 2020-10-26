use rand::prelude::*;
use rand::distributions::Standard;
use rand::distributions::uniform::{SampleUniform};
use num::{Zero, Bounded};
use std::ops::{Sub, Add, Mul, Div};
use alga::general::{Multiplicative, AbstractField};

// things that I will never get to because this is a one-day learning project:
// TODO: we only really need an integral domain for Schwartz-Zippel lemma, we can relax some types
// TODO: find out if there is any way to remove some traits in definitions of PartialEq and Zero
// TODO: support multivariate polynomials.
// TODO: test lots of fields (maybe a finite field).
// TODO: add different polynomial encodings.

/// Polynomial represents a polynomial with elements of type T.
pub trait Polynomial<T> {
    /// Returns the order of the polynomial.
    fn order(&self) -> usize;

    /// Returns the result of evaluating the polynomial at a certain element. When the polynomial
    /// has no terms, it returns None.
    fn evaluate(&self, element: T) -> Option<T>;
}

/// VecPoly is a struct that represents a polynomial with float coefficients.
///
/// The coefficients are stored in a Vec<f64> and the size of the vec minus one is the order.
/// This means that the following polynomial:
///     x^2 + 1
/// is represented by vec![1,0,1].
#[derive(Debug, Clone)]
struct VecPoly<T> {
    coefficients: Vec<T>,
}

impl<T> Polynomial<T> for VecPoly<T> where
    T: Add<Output=T> + Mul<Output=T> + Copy {

    /// The VecPoly implementation for Polynomial sets the order as the length of the coefficient
    /// vector.
    fn order(&self) -> usize {
        return self.coefficients.len();
    }

    /// Returns the evaluation of the polynomial at element, using Horner's method.
    fn evaluate(&self, element: T) -> Option<T> {
        let coefs = self.coefficients.iter();
        if let Some(result) = self.coefficients.first() {
            let mut accumulated = *result;
            for coef in coefs.skip(1) {
                accumulated = accumulated * element + *coef;
            }
            return Some(accumulated);
        }
        return None
    }
}

/// This implements PartialEq for VecPoly<T>, using the polynomial identity test to determine
/// equality of polynomials.
impl<T: Add<Output=T> + Mul<Output=T> + Div<Output=T> + Sub<Output=T> + Zero + Copy + Bounded + AbstractField + SampleUniform> PartialEq for VecPoly<T> where
    Standard: Distribution<T> {
    fn eq(&self, other: &Self) -> bool {
        // f(x) = g(x) iff f(x) - g(x) = 0
        let self_clone = self.clone();
        let other_clone = other.clone();
        let sub = self_clone - other_clone;
        return sub.is_zero();
    }
}

impl<T: Add<Output=T> + Mul<Output=T> + Div<Output=T> + Sub<Output=T> + Zero + Copy + Bounded + AbstractField + SampleUniform> Eq for VecPoly<T> where
    Standard: Distribution<T> {}

impl<T: Add<Output=T> + Copy> Add for VecPoly<T> {
    type Output=Self;

    fn add(self, other: Self) -> Self {
        return Self {
            coefficients: self.coefficients.iter().zip(other.coefficients.iter()).map(|(a,b)| *a+*b).collect()
        }
    }
}

impl<T: Sub<Output=T> + Copy> Sub for VecPoly<T> {
    type Output=Self;

    fn sub(self, other: Self) -> Self {
        return Self {
            coefficients: self.coefficients.iter().zip(other.coefficients.iter()).map(|(a,b)| *a-*b).collect()
        }
    }
}

impl<T: Add<Output=T> + Mul<Output=T> + Div<Output=T> + Zero + Copy + Bounded + AbstractField + SampleUniform> Zero for VecPoly<T> {
    /// Returns whether or not the polynomial is zero. According to the Schwartz-Zippel lemma, for
    /// a nonzero polynomial with degree d over a field of cardinality N, the probability that the
    /// polynomial is zero if = d/N.
    ///
    /// For floats, N is very large so we can be fairly confident by running it 1 time.
    fn is_zero(&self) -> bool {
        let mut rng = rand::thread_rng();
        // we do the following because the min and max values are large - so assuming the field is
        // not trivial, these values can be used to bound the rng in a generic way. It will not
        // always work but most of the time it should. It would be great if we could sample
        // the entire space uniformly at random in a generic way such that no overflows occur when
        // evaluating the polnomial.
        let min_of_range = T::min_value() / (T::id(Multiplicative) + T::id(Multiplicative));
        let max_of_range = T::max_value() / (T::id(Multiplicative) + T::id(Multiplicative));
        let rand_point = rng.gen_range(min_of_range, max_of_range);
        if let Some(eval_result) = self.evaluate(rand_point) {
            return eval_result.is_zero();
        }
        return true
    }

    fn zero() -> Self {
        return Self{ coefficients: vec![] }
    }
}

#[test]
fn check_nonzero() {
    let nonzero_poly = VecPoly::<f64>{
        coefficients: vec![1.0,0.0,1.0],
    };
    assert_eq!(nonzero_poly.is_zero(), false)
}

#[test]
fn check_zero() {
    let nonzero_poly = VecPoly::<f32>{
        coefficients: vec![0.0, 0.0, 0.0],
    };
    assert_eq!(nonzero_poly.is_zero(), true)
}
