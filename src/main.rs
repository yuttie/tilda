#[macro_use]
extern crate clap;
extern crate libc;
#[macro_use]
extern crate ndarray;
extern crate rand;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;

use std::string::String;
use std::collections::HashMap;
use std::vec::Vec;
use std::fs::{File};
use std::path::{Path};
use std::io::{self, BufReader, BufRead, BufWriter, Write};
use std::f64;
use std::fmt;
use std::str::FromStr;

use clap::{Arg, App, AppSettings};
use ndarray::{aview0, aview1, Array, Array1, Array2, Dimension, LinalgScalar};
use rand::{Closed01, Rng};
use rand::distributions::{IndependentSample, Sample, Gamma, LogNormal, RandSample, Range};
use serde::{Serialize, Deserialize};


type Bag = HashMap<usize, usize>;

fn load_bags<P: AsRef<Path>>(path: P) -> io::Result<(Vec<Bag>, usize)> {
    let mut bags = Vec::new();
    let mut vocab_size = 0;
    let file = try!(File::open(path));
    let file = BufReader::new(file);
    for line in file.lines() {
        let line = line.unwrap();
        let iter = line.split_whitespace();
        let mut bag = Bag::new();
        for elm in iter {
            let mut iter = elm.split(':');
            let index = iter.next().unwrap().parse::<usize>().unwrap();
            let value = iter.next().unwrap().parse::<usize>().unwrap();
            bag.insert(index - 1, value);    // We assume one-based indexing
            if vocab_size < index {
                vocab_size = index;
            }
        }
        bags.push(bag);
    }
    Ok((bags, vocab_size))
}

fn load_text_vocabulary<P: AsRef<Path>>(path: P) -> io::Result<Vec<String>> {
    let mut vocab = Vec::new();
    let file = try!(File::open(path));
    let file = BufReader::new(file);
    for line in file.lines() {
        let line = line.unwrap();
        vocab.push(line);
    }
    Ok(vocab)
}

#[derive(Clone)]
struct Categorical {
    prop: Vec<f64>,
}

impl Categorical {
    fn new(mut weights: Vec<f64>) -> Categorical {
        assert!(weights.iter().all(|&w| w >= 0.0), "Categorical::new() called with weights containing negative value(s)");

        let sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }
        Categorical {
            prop: weights,
        }
    }
}

impl Sample<usize> for Categorical {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> usize {
        self.ind_sample(rng)
    }
}

impl IndependentSample<usize> for Categorical {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> usize {
        let closed01 = RandSample::<Closed01<f64>>::new();
        let Closed01(x) = closed01.ind_sample(rng);

        let mut sum = 0.0;
        let mut ret = self.prop.len() - 1;
        for (k, &p) in self.prop.iter().enumerate() {
            sum += p;
            if x <= sum {
                ret = k;
                break;
            }
        }
        ret
    }
}

#[derive(Clone)]
struct Dirichlet {
    alpha: Vec<f64>,
}

impl Dirichlet {
    fn new(alpha: Vec<f64>) -> Dirichlet {
        assert!(alpha.iter().all(|&a| a > 0.0), "Dirichlet::new() called with alpha containing non-positive value(s)");

        Dirichlet {
            alpha: alpha,
        }
    }
}

impl Sample<Vec<f64>> for Dirichlet {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> Vec<f64> {
        self.ind_sample(rng)
    }
}

impl IndependentSample<Vec<f64>> for Dirichlet {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> Vec<f64> {
        let mut sum = 0.0;
        let mut xs = Vec::with_capacity(self.alpha.len());
        for &a in &self.alpha {
            let gamma = Gamma::new(a, 1.0);
            let y = gamma.ind_sample(rng);
            xs.push(y);
            sum += y;
        }
        for x in xs.iter_mut() {
            *x /= sum;
        }
        xs
    }
}

trait Samples<T> {
    type Mean;
    fn mean(self) -> Self::Mean;
}

impl<I, A, D> Samples<Array<A, D>> for I
    where I: IntoIterator<Item=Array<A, D>>, A: LinalgScalar, D: Dimension
{
    type Mean = Array<A, D>;

    fn mean(self) -> Self::Mean {
        let mut iter = self.into_iter();
        let mut sum = match iter.next() {
            Some(a) => {
                a
            },
            None => {
                panic!("No elements to sum")
            },
        };
        let mut count = A::one();
        for a in iter {
            sum = sum + a;
            count = count + A::one();
        }
        sum / &aview0(&count)
    }
}

impl<'a, I, A, D: 'a> Samples<&'a Array<A, D>> for I
    where I: IntoIterator<Item=&'a Array<A, D>>, A: LinalgScalar, D: Dimension
{
    type Mean = Array<A, D>;

    fn mean(self) -> Self::Mean {
        let mut iter = self.into_iter();
        let mut sum = match iter.next() {
            Some(a) => {
                a.to_owned()
            },
            None => {
                panic!("No elements to sum")
            },
        };
        let mut count = A::one();
        for a in iter {
            sum = sum + a;
            count = count + A::one();
        }
        sum / &aview0(&count)
    }
}

mod cmath {
    use libc::c_double;

    #[link(name = "m")]
    extern {
        fn lgamma(x: c_double) -> c_double;
    }

    pub fn lngamma(x: f64) -> f64 {
        unsafe { lgamma(x as f64) }
    }
}

// The implementation is based on "[Algorithm AS 103] Psi (Digamma) Function",
// José-Miguel Bernardo, Applied Statistics, Volume 25, pp. 315--317, 1976.
// http://www.uv.es/~bernardo/1976AppStatist.pdf
//
// Precision was improved based on the implementations from the following materials:
// https://boxtown.io/docs/statrs/0.3.0/src/statrs/src/function/gamma.rs.html#269-308
// https://github.com/bos/math-functions/blob/master/Numeric/SpecFunctions/Internal.hs
// https://github.com/lawrennd/gca/blob/master/matlab/digamma.m
// http://d.hatena.ne.jp/echizen_tm/20100627/1277646468
fn digamma(x: f64) -> f64 {
    use f64::consts::PI;
    const S:  f64 = 1e-6;
    const C:  f64 = 12.0;
    const S3: f64 = 1.0 / 12.0;
    const S4: f64 = 1.0 / 120.0;
    const S5: f64 = 1.0 / 252.0;
    const S6: f64 = 1.0 / 240.0;
    const S7: f64 = 1.0 / 132.0;
    const DIGAMMA1: f64 = -0.5772156649015328606065120;
    const TRIGAMMA1: f64 = PI * PI / 6.0;

    if x == f64::NEG_INFINITY || f64::is_nan(x) {
        f64::NAN
    }
    else if x <= 0.0 && f64::floor(x) == x {
        // x is zero or a negative integer
        f64::NAN
    }
    else if x < 0.0 {
        // x is negative non-integer
        // Use a reflection formula: psi(1 - x) - psi(x) = pi * cot(pi * x)
        digamma(1.0 - x) + PI / f64::tan(-PI * x)
    }
    else {
        // x is a positive real number
        if x <= S {
            DIGAMMA1 - 1.0 / x + TRIGAMMA1 * x
        }
        else {
            // Reduce to digamma(x + n), where y = x + n >= C
            let mut result = 0.0;
            let mut y = x;
            while y < C {
                // psi(x + 1) = psi(x) + 1 / x
                result -= 1.0 / y;
                y += 1.0;
            }
            // Compute digamma(y)
            let mut r = 1.0 / y;
            result += f64::ln(y) - 0.5 * r;
            r = r * r;
            result - r * (S3 - r * (S4 - r * (S5 - r * (S6 - r * S7))))
        }
    }
}

trait LDAModel: Serialize + Deserialize {
    fn alpha(&self) -> Array1<f64>;
    fn beta(&self) -> Array1<f64>;
    fn phi(&self) -> Array2<f64>;
    fn theta(&self) -> Array2<f64>;
    /// Approximate a marginal likelihood by a harmonic mean of likelihood samples
    fn marginal_likelihood(&self) -> f64;
}

#[allow(dead_code)]
// phi^-1: VxK matrix
fn print_term_topics<M: LDAModel>(model: &M) {
    print_term_topics_by(model, |&v| v);
}

#[allow(dead_code)]
// phi^-1: VxK matrix
fn print_term_topics_with_vocab<M: LDAModel>(model: &M, vocab: &[String]) {
    print_term_topics_by(model, |&v| &vocab[v]);
}

#[allow(dead_code)]
// phi^-1: VxK matrix
fn print_term_topics_by<M: LDAModel, T, F>(model: &M, mut f: F)
    where T: fmt::Display, F: FnMut(&usize) -> T
{
    let phi = model.phi();
    let num_topics = model.alpha().len();
    let vocab_size = model.beta().len();
    for v in 0..vocab_size {
        print!("{}:", f(&v));
        for k in 0..num_topics {
            print!(" {}*{}", phi[[k, v]], k);
        }
        println!("");
    }
}

#[allow(dead_code)]
// theta: MxK matrix
fn print_doc_topics<M: LDAModel>(model: &M) {
    let theta = model.theta();
    let num_docs = theta.rows();
    for d in 0..num_docs {
        print!("Document {}:", d);
        let mut doctopic_vec: Vec<_> = theta.row(d).iter().cloned().enumerate().collect();
        doctopic_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (k, prob) in doctopic_vec {
            print!(" {}*{}", prob, k);
        }
        println!("");
    }
}

#[allow(dead_code)]
// phi: KxV matrix
fn print_topics<M: LDAModel>(model: &M) {
    print_topics_by(model, |&v| v);
}

#[allow(dead_code)]
// phi: KxV matrix
fn print_topics_with_vocab<M: LDAModel>(model: &M, vocab: &[String]) {
    print_topics_by(model, |&v| &vocab[v]);
}

#[allow(dead_code)]
// phi: KxV matrix
fn print_topics_by<M: LDAModel, T, F>(model: &M, mut f: F)
    where T: fmt::Display, F: FnMut(&usize) -> T
{
    let phi = model.phi();
    let num_topics = model.alpha().len();
    for k in 0..num_topics {
        print!("Topic {}:", k);
        let mut topicword_vec: Vec<_> = phi.row(k).iter().cloned().enumerate().collect();
        topicword_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (v, prob) in topicword_vec {
            print!(" {}*{}", prob, f(&v));
        }
        println!("");
    }
}

#[derive(Serialize, Deserialize, Debug)]
enum DirichletPrior {
    SymmetricConstant(usize, f64),
    SymmetricVariable(usize, f64),
    AsymmetricConstant(Vec<f64>),
    AsymmetricVariable(Vec<f64>),
}

impl DirichletPrior {
    fn len(&self) -> usize {
        use DirichletPrior::*;
        match *self {
            SymmetricConstant(size, _) => size,
            SymmetricVariable(size, _) => size,
            AsymmetricConstant(ref params) => params.len(),
            AsymmetricVariable(ref params) => params.len(),
        }
    }
}

trait SamplingSolver {
    fn new(dataset: &[Bag], alpha_init: DirichletPrior, beta_init: DirichletPrior) -> Self;
    fn sample(&mut self, sample_index: Option<usize>);
}

#[derive(Serialize, Deserialize, Debug)]
struct GibbsSampler {
    alpha_init: DirichletPrior,
    beta_init: DirichletPrior,
    w:     Vec<Vec<usize>>,
    z:     Vec<Vec<usize>>,
    alpha: Array1<f64>,
    beta:  Array1<f64>,
    theta: Array2<f64>,
    phi:   Array2<f64>,
    ndk:   Array2<usize>,
    nkv:   Array2<usize>,
    nd:    Array1<usize>,
    nk:    Array1<usize>,
    // Samples
    theta_samples:          Vec<Array2<f64>>,
    phi_samples:            Vec<Array2<f64>>,
    alpha_samples:          Vec<Array1<f64>>,
    beta_samples:           Vec<Array1<f64>>,
    ndk_samples:            Vec<Array2<usize>>,
    nkv_samples:            Vec<Array2<usize>>,
    nd_samples:             Vec<Array1<usize>>,
    nk_samples:             Vec<Array1<usize>>,
    log_likelihood_samples: Vec<f64>,
}

impl SamplingSolver for GibbsSampler {
    fn new(dataset: &[Bag], alpha_init: DirichletPrior, beta_init: DirichletPrior) -> GibbsSampler {
        let num_docs: usize = dataset.len();
        let num_topics: usize = alpha_init.len();
        let vocab_size: usize = beta_init.len();
        let mut rng = rand::thread_rng();
        let alpha: Array1<f64> = {
            use DirichletPrior::*;
            match alpha_init {
                SymmetricConstant(size, param) => Array1::from_vec(vec![param; size]),
                SymmetricVariable(size, param) => Array1::from_vec(vec![param; size]),
                AsymmetricConstant(ref params) => Array1::from_vec(params.clone()),
                AsymmetricVariable(ref params) => Array1::from_vec(params.clone()),
            }
        };
        let beta: Array1<f64> = {
            use DirichletPrior::*;
            match beta_init {
                SymmetricConstant(size, param) => Array1::from_vec(vec![param; size]),
                SymmetricVariable(size, param) => Array1::from_vec(vec![param; size]),
                AsymmetricConstant(ref params) => Array1::from_vec(params.clone()),
                AsymmetricVariable(ref params) => Array1::from_vec(params.clone()),
            }
        };

        // Construct zero-filled nested arrays
        let mut w:     Vec<Vec<usize>> = Vec::with_capacity(num_docs);
        let mut z:     Vec<Vec<usize>> = Vec::with_capacity(num_docs);
        let mut theta: Array2<f64>     = Array2::zeros((num_docs, num_topics));
        let mut phi:   Array2<f64>     = Array2::zeros((num_topics, vocab_size));
        let mut ndk:   Array2<usize>   = Array2::zeros((num_docs, num_topics));
        let mut nkv:   Array2<usize>   = Array2::zeros((num_topics, vocab_size));
        let mut nd:    Array1<usize>   = Array1::zeros(num_docs);
        let mut nk:    Array1<usize>   = Array1::zeros(num_topics);
        // The same for samples
        let theta_samples:          Vec<Array2<f64>>   = Vec::new();
        let phi_samples:            Vec<Array2<f64>>   = Vec::new();
        let alpha_samples:          Vec<Array1<f64>>   = Vec::new();
        let beta_samples:           Vec<Array1<f64>>   = Vec::new();
        let ndk_samples:            Vec<Array2<usize>> = Vec::new();
        let nkv_samples:            Vec<Array2<usize>> = Vec::new();
        let nd_samples:             Vec<Array1<usize>> = Vec::new();
        let nk_samples:             Vec<Array1<usize>> = Vec::new();
        let log_likelihood_samples: Vec<f64>           = Vec::new();

        for bag in dataset {
            // n_d
            let mut n_d = 0;
            for &count in bag.values() {
                n_d += count;
            }
            w.push(vec![0; n_d]);
            z.push(vec![0; n_d]);
        }

        // Initialize w, z, ndk and nkv
        let among_topics = Range::new(0, num_topics);
        for (d, bag) in dataset.iter().enumerate() {
            let mut i = 0;
            for (&v, &count) in bag {
                for _ in 0..count {
                    w[d][i] = v;
                    let k = among_topics.ind_sample(&mut rng);
                    z[d][i] = k;
                    ndk[[d, k]] += 1;
                    nkv[[k, v]] += 1;
                    nd[d] += 1;
                    nk[k] += 1;
                    i += 1;
                }
            }
        }
        // Initialize theta, phi
        {
            let dir_a = Dirichlet::new(alpha.iter().cloned().collect());
            for d in 0..num_docs {
                theta.row_mut(d).assign(&Array1::from_vec(dir_a.ind_sample(&mut rng)));
            }
        }
        {
            let dir_b = Dirichlet::new(beta.iter().cloned().collect());
            for k in 0..num_topics {
                phi.row_mut(k).assign(&Array1::from_vec(dir_b.ind_sample(&mut rng)));
            }
        }

        GibbsSampler {
            alpha_init: alpha_init,
            beta_init: beta_init,
            w:      w,
            z:      z,
            alpha: alpha,
            beta:  beta,
            theta:  theta,
            phi:    phi,
            ndk:    ndk,
            nkv:    nkv,
            nd:     nd,
            nk:     nk,
            theta_samples: theta_samples,
            phi_samples:   phi_samples,
            alpha_samples: alpha_samples,
            beta_samples:  beta_samples,
            ndk_samples:   ndk_samples,
            nkv_samples:   nkv_samples,
            nd_samples:    nd_samples,
            nk_samples:    nk_samples,
            log_likelihood_samples:  log_likelihood_samples,
        }
    }

    fn sample(&mut self, sample_index: Option<usize>) {
        let num_docs: usize = self.w.len();
        let num_topics: usize = self.alpha_init.len();
        let vocab_size: usize = self.beta_init.len();
        let mut rng = rand::thread_rng();
        let (symmetric_alpha, constant_alpha) = {
            use DirichletPrior::*;
            match self.alpha_init {
                SymmetricConstant(_, _) => (true,  true),
                SymmetricVariable(_, _) => (true,  false),
                AsymmetricConstant(_)   => (false, true),
                AsymmetricVariable(_)   => (false, false),
            }
        };
        let (symmetric_beta, constant_beta) = {
            use DirichletPrior::*;
            match self.beta_init {
                SymmetricConstant(_, _) => (true,  true),
                SymmetricVariable(_, _) => (true,  false),
                AsymmetricConstant(_)   => (false, true),
                AsymmetricVariable(_)   => (false, false),
            }
        };

        let alpha_sum: f64 = self.alpha.scalar_sum();
        let beta_sum: f64 = self.beta.scalar_sum();
        for (d, w_d) in self.w.iter().enumerate() {
            for (i, &w_di) in w_d.iter().enumerate() {
                let v = w_di;
                // Sample z_di
                let mut weights: Vec<f64> = Vec::with_capacity(num_topics);
                for k in 0..num_topics {
                    weights.push(self.theta[[d, k]] * self.phi[[k, v]]);
                }
                let cat = Categorical::new(weights);
                let old_z_di = self.z[d][i];
                let new_z_di = cat.ind_sample(&mut rng);
                self.z[d][i] = new_z_di;
                // Update ndk and nkv
                self.ndk[[d, old_z_di]] -= 1;
                self.ndk[[d, new_z_di]] += 1;
                self.nkv[[old_z_di, v]] -= 1;
                self.nkv[[new_z_di, v]] += 1;
                self.nk[old_z_di] -= 1;
                self.nk[new_z_di] += 1;
            }
            // Sample theta_d
            let mut alpha_d: Vec<f64> = Vec::with_capacity(num_topics);
            for k in 0..num_topics {
                alpha_d.push(self.ndk[[d, k]] as f64 + self.alpha[k]);
            }
            let dir = Dirichlet::new(alpha_d);
            self.theta.row_mut(d).assign(&Array1::from_vec(dir.ind_sample(&mut rng)));
        }
        for k in 0..num_topics {
            // Sample phi_k
            let mut beta_k: Vec<f64> = Vec::with_capacity(vocab_size);
            for v in 0..vocab_size {
                beta_k.push(self.nkv[[k, v]] as f64 + self.beta[v]);
            }
            let dir = Dirichlet::new(beta_k);
            self.phi.row_mut(k).assign(&Array1::from_vec(dir.ind_sample(&mut rng)));
        }
        if let Some(i_sample) = sample_index {
            // Store samples
            {
                self.phi_samples.push(self.phi.clone());
                self.theta_samples.push(self.theta.clone());
                self.alpha_samples.push(self.alpha.clone());
                self.beta_samples.push(self.beta.clone());
                self.ndk_samples.push(self.ndk.clone());
                self.nkv_samples.push(self.nkv.clone());
                self.nd_samples.push(self.nd.clone());
                self.nk_samples.push(self.nk.clone());
            }
            // Evaluate the log-likelihood value for the current parameters
            let mut log_likelihood = 0.0;
            log_likelihood += num_topics as f64 *
                (cmath::lngamma(self.beta.scalar_sum()) - self.beta.map(|&b| cmath::lngamma(b)).scalar_sum());
            for k in 0..num_topics {
                log_likelihood += (self.nkv.row(k).map(|&x| x as f64) + &self.beta).map(|&x| cmath::lngamma(x)).scalar_sum()
                    - cmath::lngamma(self.nk[k] as f64 + self.beta.scalar_sum());
            }
            self.log_likelihood_samples.push(log_likelihood);
            // Update alpha and beta
            if !constant_alpha {
                let ndk = self.ndk_samples.iter().map(|a| a.map(|x| *x as f64)).mean();
                let nd = self.nd_samples.iter().map(|a| a.map(|x| *x as f64)).mean();
                for k in 0..num_topics {
                    let mut x = -(num_docs as f64) * digamma(self.alpha[k]);
                    let mut y = -(num_docs as f64) * digamma(alpha_sum);
                    for d in 0..num_docs {
                        x += digamma(ndk[[d, k]] as f64 + self.alpha[k]);
                        y += digamma(nd[d] as f64 + alpha_sum);
                    }
                    self.alpha[k] = self.alpha[k] * x / y;
                }
                if symmetric_alpha {
                    let a_sym = self.alpha.scalar_sum() / self.alpha.len() as f64;
                    for a in &mut self.alpha {
                        *a = a_sym;
                    }
                }
            }
            if !constant_beta {
                let nkv = self.nkv_samples.iter().map(|a| a.map(|x| *x as f64)).mean();
                let nk = self.nk_samples.iter().map(|a| a.map(|x| *x as f64)).mean();
                for v in 0..vocab_size {
                    let mut x = -(num_topics as f64) * digamma(self.beta[v]);
                    let mut y = -(num_topics as f64) * digamma(beta_sum);
                    for k in 0..num_topics {
                        x += digamma(nkv[[k, v]] as f64 + self.beta[v]);
                        y += digamma(nk[k] as f64 + beta_sum);
                    }
                    self.beta[v] = self.beta[v] * x / y;
                }
                if symmetric_beta {
                    let b_sym = self.beta.scalar_sum() / self.beta.len() as f64;
                    for b in &mut self.beta {
                        *b = b_sym;
                    }
                }
            }
        }
    }
}

impl LDAModel for GibbsSampler {
    fn alpha(&self) -> Array1<f64> {
        self.alpha.clone()
    }

    fn beta(&self) -> Array1<f64> {
        self.beta.clone()
    }

    fn phi(&self) -> Array2<f64> {
        self.phi.clone()
    }

    fn theta(&self) -> Array2<f64> {
        self.theta.clone()
    }

    /// Approximate a marginal likelihood by a harmonic mean of likelihood samples
    fn marginal_likelihood(&self) -> f64 {
        let samples = aview1(&self.log_likelihood_samples);
        samples.len() as f64 / samples.map(|x| 1.0 / x).scalar_sum()
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct CollapsedGibbsSampler {
    alpha_init: DirichletPrior,
    beta_init: DirichletPrior,
    w:     Vec<Vec<usize>>,
    z:     Vec<Vec<usize>>,
    alpha: Array1<f64>,
    beta:  Array1<f64>,
    ndk:   Array2<usize>,
    nkv:   Array2<usize>,
    nd:    Array1<usize>,
    nk:    Array1<usize>,
    // Samples
    alpha_samples:          Vec<Array1<f64>>,
    beta_samples:           Vec<Array1<f64>>,
    ndk_samples:            Vec<Array2<usize>>,
    nkv_samples:            Vec<Array2<usize>>,
    nd_samples:             Vec<Array1<usize>>,
    nk_samples:             Vec<Array1<usize>>,
    log_likelihood_samples: Vec<f64>,
}

impl SamplingSolver for CollapsedGibbsSampler {
    fn new(dataset: &[Bag], alpha_init: DirichletPrior, beta_init: DirichletPrior) -> CollapsedGibbsSampler {
        let num_docs: usize = dataset.len();
        let num_topics: usize = alpha_init.len();
        let vocab_size: usize = beta_init.len();
        let mut rng = rand::thread_rng();
        let alpha: Array1<f64> = {
            use DirichletPrior::*;
            match alpha_init {
                SymmetricConstant(size, param) => Array1::from_vec(vec![param; size]),
                SymmetricVariable(size, param) => Array1::from_vec(vec![param; size]),
                AsymmetricConstant(ref params) => Array1::from_vec(params.clone()),
                AsymmetricVariable(ref params) => Array1::from_vec(params.clone()),
            }
        };
        let beta: Array1<f64> = {
            use DirichletPrior::*;
            match beta_init {
                SymmetricConstant(size, param) => Array1::from_vec(vec![param; size]),
                SymmetricVariable(size, param) => Array1::from_vec(vec![param; size]),
                AsymmetricConstant(ref params) => Array1::from_vec(params.clone()),
                AsymmetricVariable(ref params) => Array1::from_vec(params.clone()),
            }
        };

        // Construct zero-filled nested arrays
        let mut w:     Vec<Vec<usize>> = Vec::with_capacity(num_docs);
        let mut z:     Vec<Vec<usize>> = Vec::with_capacity(num_docs);
        let mut ndk:   Array2<usize>   = Array2::zeros((num_docs, num_topics));
        let mut nkv:   Array2<usize>   = Array2::zeros((num_topics, vocab_size));
        let mut nd:    Array1<usize>   = Array1::zeros(num_docs);
        let mut nk:    Array1<usize>   = Array1::zeros(num_topics);
        // The same for samples
        let alpha_samples:          Vec<Array1<f64>>   = Vec::new();
        let beta_samples:           Vec<Array1<f64>>   = Vec::new();
        let ndk_samples:            Vec<Array2<usize>> = Vec::new();
        let nkv_samples:            Vec<Array2<usize>> = Vec::new();
        let nd_samples:             Vec<Array1<usize>> = Vec::new();
        let nk_samples:             Vec<Array1<usize>> = Vec::new();
        let log_likelihood_samples: Vec<f64>           = Vec::new();

        for bag in dataset {
            // n_d
            let mut n_d = 0;
            for &count in bag.values() {
                n_d += count;
            }
            w.push(vec![0; n_d]);
            z.push(vec![0; n_d]);
        }

        // Initialize w, z, ndk and nkv
        let among_topics = Range::new(0, num_topics);
        for (d, bag) in dataset.iter().enumerate() {
            let mut i = 0;
            for (&v, &count) in bag {
                for _ in 0..count {
                    w[d][i] = v;
                    let k = among_topics.ind_sample(&mut rng);
                    z[d][i] = k;
                    ndk[[d, k]] += 1;
                    nkv[[k, v]] += 1;
                    nd[d] += 1;
                    nk[k] += 1;
                    i += 1;
                }
            }
        }

        CollapsedGibbsSampler {
            alpha_init: alpha_init,
            beta_init: beta_init,
            w:      w,
            z:      z,
            alpha: alpha,
            beta:  beta,
            ndk:    ndk,
            nkv:    nkv,
            nd:     nd,
            nk:     nk,
            alpha_samples: alpha_samples,
            beta_samples:  beta_samples,
            ndk_samples:   ndk_samples,
            nkv_samples:   nkv_samples,
            nd_samples:    nd_samples,
            nk_samples:    nk_samples,
            log_likelihood_samples:  log_likelihood_samples,
        }
    }

    fn sample(&mut self, sample_index: Option<usize>) {
        let num_docs: usize = self.w.len();
        let num_topics: usize = self.alpha_init.len();
        let vocab_size: usize = self.beta_init.len();
        let mut rng = rand::thread_rng();
        let (symmetric_alpha, constant_alpha) = {
            use DirichletPrior::*;
            match self.alpha_init {
                SymmetricConstant(_, _) => (true,  true),
                SymmetricVariable(_, _) => (true,  false),
                AsymmetricConstant(_)   => (false, true),
                AsymmetricVariable(_)   => (false, false),
            }
        };
        let (symmetric_beta, constant_beta) = {
            use DirichletPrior::*;
            match self.beta_init {
                SymmetricConstant(_, _) => (true,  true),
                SymmetricVariable(_, _) => (true,  false),
                AsymmetricConstant(_)   => (false, true),
                AsymmetricVariable(_)   => (false, false),
            }
        };

        let alpha_sum: f64 = self.alpha.scalar_sum();
        let beta_sum: f64 = self.beta.scalar_sum();
        for (d, w_d) in self.w.iter().enumerate() {
            for (i, &w_di) in w_d.iter().enumerate() {
                let v = w_di;
                //
                let old_z_di = self.z[d][i];
                self.ndk[[d, old_z_di]] -= 1;
                self.nkv[[old_z_di, v]] -= 1;
                self.nd[d] -= 1;
                self.nk[old_z_di] -= 1;
                // Sample z_di
                let mut weights: Vec<f64> = Vec::with_capacity(num_topics);
                for k in 0..num_topics {
                    let e_theta_dk = (self.ndk[[d, k]] as f64 + self.alpha[k]) / (self.nd[d] as f64 + alpha_sum);
                    let e_phi_kv = (self.nkv[[k, v]] as f64 + self.beta[v]) / (self.nk[k] as f64 + beta_sum);
                    weights.push(e_theta_dk * e_phi_kv);
                }
                // println!("{:?}", weights);
                let cat = Categorical::new(weights);
                let new_z_di = cat.ind_sample(&mut rng);
                self.z[d][i] = new_z_di;
                self.ndk[[d, new_z_di]] += 1;
                self.nkv[[new_z_di, v]] += 1;
                self.nd[d] += 1;
                self.nk[new_z_di] += 1;
            }
        }
        if let Some(i_sample) = sample_index {
            // Store samples
            {
                self.alpha_samples.push(self.alpha.clone());
                self.beta_samples.push(self.beta.clone());
                self.ndk_samples.push(self.ndk.clone());
                self.nkv_samples.push(self.nkv.clone());
                self.nd_samples.push(self.nd.clone());
                self.nk_samples.push(self.nk.clone());
            }
            // Evaluate the log-likelihood value for the current parameters
            let mut log_likelihood = 0.0;
            log_likelihood += num_topics as f64 *
                (cmath::lngamma(self.beta.scalar_sum()) - self.beta.map(|&b| cmath::lngamma(b)).scalar_sum());
            for k in 0..num_topics {
                log_likelihood += (self.nkv.row(k).map(|&x| x as f64) + &self.beta).map(|&x| cmath::lngamma(x)).scalar_sum()
                    - cmath::lngamma(self.nk[k] as f64 + self.beta.scalar_sum());
            }
            self.log_likelihood_samples.push(log_likelihood);
            // Update alpha and beta
            if !constant_alpha {
                let ndk = self.ndk_samples.iter().map(|a| a.map(|x| *x as f64)).mean();
                let nd = self.nd_samples.iter().map(|a| a.map(|x| *x as f64)).mean();
                for k in 0..num_topics {
                    let mut x = -(num_docs as f64) * digamma(self.alpha[k]);
                    let mut y = -(num_docs as f64) * digamma(alpha_sum);
                    for d in 0..num_docs {
                        x += digamma(ndk[[d, k]] as f64 + self.alpha[k]);
                        y += digamma(nd[d] as f64 + alpha_sum);
                    }
                    self.alpha[k] = self.alpha[k] * x / y;
                }
                if symmetric_alpha {
                    let a_sym = self.alpha.scalar_sum() / self.alpha.len() as f64;
                    for a in &mut self.alpha {
                        *a = a_sym;
                    }
                }
            }
            if !constant_beta {
                let nkv = self.nkv_samples.iter().map(|a| a.map(|x| *x as f64)).mean();
                let nk = self.nk_samples.iter().map(|a| a.map(|x| *x as f64)).mean();
                for v in 0..vocab_size {
                    let mut x = -(num_topics as f64) * digamma(self.beta[v]);
                    let mut y = -(num_topics as f64) * digamma(beta_sum);
                    for k in 0..num_topics {
                        x += digamma(nkv[[k, v]] as f64 + self.beta[v]);
                        y += digamma(nk[k] as f64 + beta_sum);
                    }
                    self.beta[v] = self.beta[v] * x / y;
                }
                if symmetric_beta {
                    let b_sym = self.beta.scalar_sum() / self.beta.len() as f64;
                    for b in &mut self.beta {
                        *b = b_sym;
                    }
                }
            }
        }
    }
}

impl LDAModel for CollapsedGibbsSampler {
    fn alpha(&self) -> Array1<f64> {
        self.alpha.clone()
    }

    fn beta(&self) -> Array1<f64> {
        self.beta.clone()
    }

    fn phi(&self) -> Array2<f64> {
        let num_topics = self.alpha.len();
        let vocab_size = self.beta.len();
        let beta_sum: f64 = self.beta.scalar_sum();
        // Infer phi
        let mut phi: Array2<f64> = Array2::zeros((num_topics, vocab_size));
        for k in 0..num_topics {
            for v in 0..vocab_size {
                phi[[k, v]] = (self.nkv[[k, v]] as f64 + self.beta[v]) / (self.nk[k] as f64 + beta_sum);
            }
        }
        phi
    }

    fn theta(&self) -> Array2<f64> {
        let num_docs = self.w.len();
        let num_topics = self.alpha.len();
        let alpha_sum: f64 = self.alpha.scalar_sum();
        // Infer theta
        let mut theta: Array2<f64> = Array2::zeros((num_docs, num_topics));
        for d in 0..num_docs {
            for k in 0..num_topics {
                theta[[d, k]] = (self.ndk[[d, k]] as f64 + self.alpha[k]) / (self.nd[d] as f64 + alpha_sum);
            }
        }
        theta
    }

    /// Approximate a marginal likelihood by a harmonic mean of likelihood samples
    fn marginal_likelihood(&self) -> f64 {
        let samples = aview1(&self.log_likelihood_samples);
        samples.len() as f64 / samples.map(|x| 1.0 / x).scalar_sum()
    }
}

fn run_solver<S: SamplingSolver>(dataset: &[Bag], alpha_init: DirichletPrior, beta_init: DirichletPrior, burn_in: usize, num_samples: usize, lag: usize) -> S {
    write!(&mut std::io::stderr(), "Initializing...").unwrap();
    let mut sampler = S::new(dataset, alpha_init, beta_init);
    writeln!(&mut std::io::stderr(), "\rInitialized.").unwrap();

    write!(&mut std::io::stderr(), "Sampling...").unwrap();
    for s in 0..(burn_in + num_samples * lag) {
        if s >= burn_in && (s - burn_in + 1) % lag == 0 {
            let i_sample = (s - burn_in + 1) / lag - 1;
            sampler.sample(Some(i_sample));
        }
        else {
            sampler.sample(None);
        }
        if s < burn_in {
            write!(&mut std::io::stderr(), "\rBurn-in... {}/{}", s + 1, burn_in).unwrap();
        }
        else {
            if s == burn_in {
                writeln!(&mut std::io::stderr(), "").unwrap();
            }
            write!(&mut std::io::stderr(), "\rSampling... {}/{}", s - burn_in + 1, num_samples * lag).unwrap();
        }
    }
    writeln!(&mut std::io::stderr(), "\rSampled.").unwrap();

    sampler
}

fn make_dataset(num_docs: usize, mean_nd: f64, std_dev_nd: f64, alpha: &[f64], beta: &[f64]) -> Vec<Bag> {
    let num_topics: usize = alpha.len();
    let mut rng = rand::thread_rng();
    // phi
    let dir_beta = Dirichlet::new(beta.to_vec());
    let mut phi: Vec<Vec<f64>> = Vec::with_capacity(num_topics);
    for _k in 0..num_topics {
        // Sample phi_k
        phi.push(dir_beta.ind_sample(&mut rng));
    }
    // theta, nd, z, w
    let dir_alpha = Dirichlet::new(alpha.to_vec());
    let lognorm = LogNormal::new(mean_nd, std_dev_nd);
    let mut theta: Vec<Vec<f64>> = Vec::with_capacity(num_docs);
    let mut nd: Vec<usize> = Vec::with_capacity(num_docs);
    let mut z: Vec<Vec<usize>> = Vec::with_capacity(num_docs);
    let mut w: Vec<Vec<usize>> = Vec::with_capacity(num_docs);
    for d in 0..num_docs {
        theta.push(dir_alpha.ind_sample(&mut rng));
        nd.push(f64::ceil(lognorm.ind_sample(&mut rng)) as usize);
        let cat_theta = Categorical::new(theta[d].clone());
        let mut z_d = Vec::with_capacity(nd[d]);
        let mut w_d = Vec::with_capacity(nd[d]);
        for i in 0..nd[d] {
            z_d.push(cat_theta.ind_sample(&mut rng));
            let cat_phi = Categorical::new(phi[z_d[i]].clone());
            w_d.push(cat_phi.ind_sample(&mut rng));
        }
        z.push(z_d);
        w.push(w_d);
    }
    // Make bags
    let mut bags: Vec<Bag> = Vec::with_capacity(w.len());
    for w_d in w {
        let mut bag: Bag = Bag::new();
        for v in w_d {
            let counter = bag.entry(v).or_insert(0);
            *counter += 1;
        }
        bags.push(bag);
    }
    bags
}

fn make_visual_dataset(size: usize, num_docs: usize) -> Vec<Bag> {
    let num_topics: usize = size + size;
    let vocab_size: usize = size * size;
    let mut rng = rand::thread_rng();
    // phi
    let mut phi: Vec<Vec<f64>> = vec![vec![0.0; vocab_size]; num_topics];
    for k in 0..size {
        let j = k;
        for i in 0..size {
            let v = j * size + i;
            phi[k][v] = 1.0 / size as f64;
        }
    }
    for k in size..num_topics {
        let i = k - size;
        for j in 0..size {
            let v = j * size + i;
            phi[k][v] = 1.0 / size as f64;
        }
    }
    // theta
    let mut theta: Vec<Vec<f64>> = Vec::with_capacity(num_docs);
    let dir_alpha = Dirichlet::new(vec![1.0; num_topics]);
    for _d in 0..num_docs {
        theta.push(dir_alpha.ind_sample(&mut rng));
    }
    // nd
    let nd: Vec<usize> = vec![127; num_docs];
    // z, w
    let mut z: Vec<Vec<usize>> = Vec::with_capacity(num_docs);
    let mut w: Vec<Vec<usize>> = Vec::with_capacity(num_docs);
    for d in 0..num_docs {
        let cat_theta = Categorical::new(theta[d].clone());
        let mut z_d = Vec::with_capacity(nd[d]);
        let mut w_d = Vec::with_capacity(nd[d]);
        for i in 0..nd[d] {
            z_d.push(cat_theta.ind_sample(&mut rng));
            let cat_phi = Categorical::new(phi[z_d[i]].clone());
            w_d.push(cat_phi.ind_sample(&mut rng));
        }
        z.push(z_d);
        w.push(w_d);
    }
    // Make bags
    let mut bags: Vec<Bag> = Vec::with_capacity(w.len());
    for w_d in w {
        let mut bag: Bag = Bag::new();
        for v in w_d {
            let counter = bag.entry(v).or_insert(0);
            *counter += 1;
        }
        bags.push(bag);
    }
    bags
}

/// This function guarantees that the order relation of any pair of word IDs in
/// a given dataset are preserved also in an output dataset.
fn compact_words(mut bags: Vec<Bag>) -> (Vec<Bag>, usize, HashMap<usize, usize>) {
    // Find all used word IDs
    let mut ids: Vec<usize> = bags.iter().flat_map(|bag| bag.keys()).cloned().collect();
    // Make word IDs in the list sorted and unique
    ids.sort();
    ids.dedup();
    // Construct ID maps
    let id_map: HashMap<usize, usize> = ids.iter().cloned().zip(0..).collect();
    let inv_id_map: HashMap<usize, usize> = ids.into_iter().enumerate().collect();
    // Map IDs
    for bag in &mut bags {
        *bag = bag.into_iter().map(|(id, &mut c)| (id_map[id], c)).collect();
    }
    (bags, inv_id_map.len(), inv_id_map)
}

#[allow(dead_code)]
fn decompact_words(mut bags: Vec<Bag>, id_map: HashMap<usize, usize>) -> (Vec<Bag>, usize) {
    let mut vocab_size = 0;
    for bag in &mut bags {
        *bag = bag.into_iter().map(|(&id, &mut c)| {
            let new_id: usize = id_map[&id];
            if vocab_size < new_id + 1 {
                vocab_size = new_id + 1;
            }
            (new_id, c)
        }).collect();
    }
    (bags, vocab_size)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Method {
    Gibbs,
    CollapsedGibbs,
}

impl FromStr for Method {
    type Err = ParseMethodError;

    fn from_str(s: &str) -> Result<Method, ParseMethodError> {
        match s {
            "gibbs"           => Ok(Method::Gibbs),
            "collapsed-gibbs" => Ok(Method::CollapsedGibbs),
            _                 => Err(ParseMethodError)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParseMethodError;

impl fmt::Display for ParseMethodError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "provided string was not `gibbs` or `collapsed-gibbs`".fmt(f)
    }
}

fn main() {
    let matches = App::new("TiLDA")
        .version("0.2")
        .author("Yuta Taniguchi <yuta.taniguchi.y.t@gmail.com>")
        .about("Latent Dirichlet allocation implemented in Rust")
        .arg(Arg::with_name("topics")
             .long("topics")
             .takes_value(true)
             .value_name("NUMBER")
             .help("Set the number of topics"))
        .arg(Arg::with_name("method")
             .long("method")
             .takes_value(true)
             .value_name("METHOD")
             .possible_values(&["gibbs", "collapsed-gibbs"])
             .default_value("collapsed-gibbs")
             .help("Specify a method to use"))
        .arg(Arg::with_name("burn-in")
             .long("burn-in")
             .takes_value(true)
             .value_name("NUMBER")
             .default_value("1000")
             .help("Set the number of samples for burn-in"))
        .arg(Arg::with_name("samples")
             .long("samples")
             .takes_value(true)
             .value_name("NUMBER")
             .default_value("1000")
             .help("Set the number of samples"))
        .arg(Arg::with_name("lag")
             .long("lag")
             .takes_value(true)
             .value_name("NUMBER")
             .default_value("100")
             .help("Set lag for sampling"))
        .arg(Arg::with_name("fix-alpha")
             .long("fix-alpha")
             .help("Don't update alpha"))
        .arg(Arg::with_name("fix-beta")
             .long("fix-beta")
             .help("Don't update beta"))
        .arg(Arg::with_name("symmetric-alpha")
             .long("symmetric-alpha")
             .help("Make alpha symmetric"))
        .arg(Arg::with_name("symmetric-beta")
             .long("symmetric-beta")
             .help("Make beta symmetric"))
        .arg(Arg::with_name("test-dataset")
             .long("test-dataset")
             .help("Run with automatically generated dataset"))
        .arg(Arg::with_name("test-visual-dataset")
             .long("test-visual-dataset")
             .help("Run with automatically generated visual dataset"))
        .arg(Arg::with_name("model")
             .long("model")
             .value_name("MODEL-FILE")
             .help("Specify a model file")
             .takes_value(true))
        .arg(Arg::with_name("vocab")
             .long("vocab")
             .takes_value(true)
             .value_name("FILE")
             .help("Sets a vocabulary file to use"))
        .arg(Arg::with_name("INPUT")
             .help("Sets the input file to use")
             .required(false)
             .index(1)
             .requires("topics"))
        .setting(AppSettings::ArgRequiredElseHelp)
        .get_matches();

    let method = value_t_or_exit!(matches, "method", Method);
    let burn_in = value_t_or_exit!(matches, "burn-in", usize);
    let samples = value_t_or_exit!(matches, "samples", usize);
    let lag = value_t_or_exit!(matches, "lag", usize);

    if matches.is_present("test-dataset") {
        let num_topics = 10;
        let vocab_size = 10000;
        let alpha: Vec<f64> = vec![0.1; num_topics];
        let beta: Vec<f64> = vec![0.1; vocab_size];
        write!(&mut std::io::stderr(), "Generating a dataset...").unwrap();
        let dataset = make_dataset(1000, f64::ln(400f64), 0.3, &alpha, &beta);
        writeln!(&mut std::io::stderr(), " done.").unwrap();
        write!(&mut std::io::stderr(), "Compacting the dataset...").unwrap();
        let (dataset, vocab_size, inv_id_map) = compact_words(dataset);
        writeln!(&mut std::io::stderr(), " done.").unwrap();
        writeln!(&mut std::io::stderr(), "Vocab: {}", vocab_size).unwrap();

        let alpha_init = {
            if matches.is_present("symmetric-alpha") {
                if matches.is_present("fix-alpha") {
                    DirichletPrior::SymmetricConstant(num_topics, 1.0)
                }
                else {
                    DirichletPrior::SymmetricVariable(num_topics, 1.0)
                }
            }
            else {
                if matches.is_present("fix-alpha") {
                    DirichletPrior::AsymmetricConstant(vec![1.0; num_topics])
                }
                else {
                    DirichletPrior::AsymmetricVariable(vec![1.0; num_topics])
                }
            }
        };
        let beta_init = {
            if matches.is_present("symmetric-beta") {
                if matches.is_present("fix-beta") {
                    DirichletPrior::SymmetricConstant(vocab_size, 1.0)
                }
                else {
                    DirichletPrior::SymmetricVariable(vocab_size, 1.0)
                }
            }
            else {
                if matches.is_present("fix-beta") {
                    DirichletPrior::AsymmetricConstant(vec![1.0; vocab_size])
                }
                else {
                    DirichletPrior::AsymmetricVariable(vec![1.0; vocab_size])
                }
            }
        };

        match method {
            Method::Gibbs => {
                let model = run_solver::<GibbsSampler>(&dataset, alpha_init, beta_init, burn_in, samples, lag);

                print_doc_topics(&model);
                print_term_topics_by(&model, |id| inv_id_map[id]);
                print_topics_by(&model, |id| inv_id_map[id]);

                if let Some(fp) = matches.value_of("model") {
                    let file = File::create(&fp).unwrap();
                    let mut file = BufWriter::new(file);
                    serde_json::to_writer_pretty(&mut file, &model).unwrap();
                }
            },
            Method::CollapsedGibbs => {
                let model = run_solver::<CollapsedGibbsSampler>(&dataset, alpha_init, beta_init, burn_in, samples, lag);

                print_doc_topics(&model);
                print_term_topics_by(&model, |id| inv_id_map[id]);
                print_topics_by(&model, |id| inv_id_map[id]);

                if let Some(fp) = matches.value_of("model") {
                    let file = File::create(&fp).unwrap();
                    let mut file = BufWriter::new(file);
                    serde_json::to_writer_pretty(&mut file, &model).unwrap();
                }
            },
        }
    }
    else if matches.is_present("test-visual-dataset") {
        let size = 5;
        let num_topics: usize = size + size;
        write!(&mut std::io::stderr(), "Generating a dataset...").unwrap();
        let dataset = make_visual_dataset(size, 1000);
        writeln!(&mut std::io::stderr(), " done.").unwrap();
        write!(&mut std::io::stderr(), "Compacting the dataset...").unwrap();
        let (dataset, vocab_size, inv_id_map) = compact_words(dataset);
        writeln!(&mut std::io::stderr(), " done.").unwrap();
        writeln!(&mut std::io::stderr(), "Vocab: {}", vocab_size).unwrap();

        let alpha_init = {
            if matches.is_present("symmetric-alpha") {
                if matches.is_present("fix-alpha") {
                    DirichletPrior::SymmetricConstant(num_topics, 1.0)
                }
                else {
                    DirichletPrior::SymmetricVariable(num_topics, 1.0)
                }
            }
            else {
                if matches.is_present("fix-alpha") {
                    DirichletPrior::AsymmetricConstant(vec![1.0; num_topics])
                }
                else {
                    DirichletPrior::AsymmetricVariable(vec![1.0; num_topics])
                }
            }
        };
        let beta_init = {
            if matches.is_present("symmetric-beta") {
                if matches.is_present("fix-beta") {
                    DirichletPrior::SymmetricConstant(vocab_size, 1.0)
                }
                else {
                    DirichletPrior::SymmetricVariable(vocab_size, 1.0)
                }
            }
            else {
                if matches.is_present("fix-beta") {
                    DirichletPrior::AsymmetricConstant(vec![1.0; vocab_size])
                }
                else {
                    DirichletPrior::AsymmetricVariable(vec![1.0; vocab_size])
                }
            }
        };

        match method {
            Method::Gibbs => {
                let model = run_solver::<GibbsSampler>(&dataset, alpha_init, beta_init, burn_in, samples, lag);

                print_doc_topics(&model);
                print_term_topics_by(&model, |id| inv_id_map[id]);
                print_topics_by(&model, |id| inv_id_map[id]);
                println!("alpha = {:?}", model.alpha);
                println!("beta = {:?}", model.beta);

                if let Some(fp) = matches.value_of("model") {
                    let file = File::create(&fp).unwrap();
                    let mut file = BufWriter::new(file);
                    serde_json::to_writer_pretty(&mut file, &model).unwrap();
                }
            },
            Method::CollapsedGibbs => {
                let model = run_solver::<CollapsedGibbsSampler>(&dataset, alpha_init, beta_init, burn_in, samples, lag);

                print_doc_topics(&model);
                print_term_topics_by(&model, |id| inv_id_map[id]);
                print_topics_by(&model, |id| inv_id_map[id]);
                println!("alpha = {:?}", model.alpha);
                println!("beta = {:?}", model.beta);

                if let Some(fp) = matches.value_of("model") {
                    let file = File::create(&fp).unwrap();
                    let mut file = BufWriter::new(file);
                    serde_json::to_writer_pretty(&mut file, &model).unwrap();
                }
            },
        }
    }
    else if let Some(input_fp) = matches.value_of("INPUT") {
        let (dataset, _) = load_bags(input_fp).unwrap();
        let vocab: Option<Vec<String>> = if let Some(vocab_fp) = matches.value_of("vocab") {
            Some(load_text_vocabulary(vocab_fp).unwrap())
        }
        else {
            None
        };
        write!(&mut std::io::stderr(), "Compacting the dataset...").unwrap();
        let (dataset, vocab_size, inv_id_map) = compact_words(dataset);
        writeln!(&mut std::io::stderr(), " done.").unwrap();
        let num_topics = value_t_or_exit!(matches, "topics", usize);

        let alpha_init = {
            if matches.is_present("symmetric-alpha") {
                if matches.is_present("fix-alpha") {
                    DirichletPrior::SymmetricConstant(num_topics, 1.0)
                }
                else {
                    DirichletPrior::SymmetricVariable(num_topics, 1.0)
                }
            }
            else {
                if matches.is_present("fix-alpha") {
                    DirichletPrior::AsymmetricConstant(vec![1.0; num_topics])
                }
                else {
                    DirichletPrior::AsymmetricVariable(vec![1.0; num_topics])
                }
            }
        };
        let beta_init = {
            if matches.is_present("symmetric-beta") {
                if matches.is_present("fix-beta") {
                    DirichletPrior::SymmetricConstant(vocab_size, 1.0)
                }
                else {
                    DirichletPrior::SymmetricVariable(vocab_size, 1.0)
                }
            }
            else {
                if matches.is_present("fix-beta") {
                    DirichletPrior::AsymmetricConstant(vec![1.0; vocab_size])
                }
                else {
                    DirichletPrior::AsymmetricVariable(vec![1.0; vocab_size])
                }
            }
        };

        match method {
            Method::Gibbs => {
                let model = run_solver::<GibbsSampler>(&dataset, alpha_init, beta_init, burn_in, samples, lag);

                print_doc_topics(&model);
                match vocab {
                    Some(vocab) => {
                        print_term_topics_by(&model, |id| &vocab[inv_id_map[id]]);
                        print_topics_by(&model, |id| &vocab[inv_id_map[id]]);
                    },
                    None => {
                        print_term_topics_by(&model, |id| inv_id_map[id]);
                        print_topics_by(&model, |id| inv_id_map[id]);
                    }
                }

                if let Some(fp) = matches.value_of("model") {
                    let file = File::create(&fp).unwrap();
                    let mut file = BufWriter::new(file);
                    serde_json::to_writer_pretty(&mut file, &model).unwrap();
                }
            },
            Method::CollapsedGibbs => {
                let model = run_solver::<CollapsedGibbsSampler>(&dataset, alpha_init, beta_init, burn_in, samples, lag);

                print_doc_topics(&model);
                match vocab {
                    Some(vocab) => {
                        print_term_topics_by(&model, |id| &vocab[inv_id_map[id]]);
                        print_topics_by(&model, |id| &vocab[inv_id_map[id]]);
                    },
                    None => {
                        print_term_topics_by(&model, |id| inv_id_map[id]);
                        print_topics_by(&model, |id| inv_id_map[id]);
                    }
                }

                if let Some(fp) = matches.value_of("model") {
                    let file = File::create(&fp).unwrap();
                    let mut file = BufWriter::new(file);
                    serde_json::to_writer_pretty(&mut file, &model).unwrap();
                }
            },
        }
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn compact_preserves_order_of_ids() {
        let num_topics = 4;
        let vocab_size = 10000;
        let alpha: Vec<f64> = vec![0.1; num_topics];
        let beta: Vec<f64> = vec![0.1; vocab_size];
        let dataset = ::make_dataset(10, f64::ln(10f64), 0.01, &alpha, &beta);
        let (_, _, inv_id_map) = ::compact_words(dataset);
        let mut map_pairs: Vec<(usize, usize)> = inv_id_map.into_iter().collect();
        map_pairs.sort();
        assert!(map_pairs.windows(2).all(|w| {
            let (new_id1, old_id1) = w[0];
            let (new_id2, old_id2) = w[1];
            new_id1 < new_id2 && old_id1 < old_id2
        }));
    }

    #[test]
    fn compact_makes_dense_word_ids() {
        let num_topics = 4;
        let vocab_size = 10000;
        let alpha: Vec<f64> = vec![0.1; num_topics];
        let beta: Vec<f64> = vec![0.1; vocab_size];
        let dataset = ::make_dataset(10, f64::ln(10f64), 0.01, &alpha, &beta);
        let (_, compact_vocab_size, _) = ::compact_words(dataset);
        assert!(compact_vocab_size < vocab_size);
    }

    #[test]
    fn decompact_inverses_compact() {
        let num_topics = 4;
        let vocab_size = 10000;
        let alpha: Vec<f64> = vec![0.1; num_topics];
        let beta: Vec<f64> = vec![0.1; vocab_size];
        let dataset = ::make_dataset(10, f64::ln(10f64), 0.01, &alpha, &beta);
        let (compacted_dataset, _, inv_id_map) = ::compact_words(dataset.clone());
        let (decompacted_dataset, _) = ::decompact_words(compacted_dataset, inv_id_map);
        assert_eq!(decompacted_dataset, dataset);
    }
}
