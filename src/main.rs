#![feature(proc_macro)]
#[macro_use]
extern crate clap;
extern crate rand;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;

use std::string::String;
use std::collections::HashMap;
use std::vec::Vec;
use std::fs::{File};
use std::path::{Path};
use std::io::{self, BufReader, BufRead, Write};
use std::f64;
use std::fmt;
use std::str::FromStr;

use clap::{Arg, App, AppSettings};
use rand::{Closed01, Rng};
use rand::distributions::{IndependentSample, Sample, Gamma, LogNormal, RandSample, Range};


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
        let mut xs = Vec::new();
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
    const S:  f64 = 1e-6;
    const C:  f64 = 12.0;
    const S3: f64 = 1.0 / 12.0;
    const S4: f64 = 1.0 / 120.0;
    const S5: f64 = 1.0 / 252.0;
    const S6: f64 = 1.0 / 240.0;
    const S7: f64 = 1.0 / 132.0;
    const DIGAMMA1: f64 = -0.5772156649015328606065120;
    const TRIGAMMA1: f64 = f64::consts::PI * f64::consts::PI / 6.0;

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
        digamma(1.0 - x) + f64::consts::PI / f64::tan(-f64::consts::PI * x)
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

#[derive(Serialize, Deserialize, Debug)]
struct Model {
    alpha_init: Vec<f64>,
    beta_init: Vec<f64>,
    burn_in: usize,
    num_samples: usize,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    theta: Vec<Vec<f64>>,
    phi:   Vec<Vec<f64>>,
    z_samples: Vec<Vec<Vec<usize>>>,
}

impl Model {
    // phi^-1: VxK matrix
    fn print_term_topics(&self) {
        self.print_term_topics_by(|&v| v);
    }

    // phi^-1: VxK matrix
    fn print_term_topics_with_vocab(&self, vocab: &[String]) {
        self.print_term_topics_by(|&v| &vocab[v]);
    }

    // phi^-1: VxK matrix
    fn print_term_topics_by<T, F>(&self, mut f: F)
        where T: fmt::Display, F: FnMut(&usize) -> T
    {
        let num_topics = self.alpha.len();
        let vocab_size = self.beta_init.len();
        for v in 0..vocab_size {
            print!("{}:", f(&v));
            for k in 0..num_topics {
                if self.phi[k][v] > 1e-9 {
                    print!(" {}*{}", self.phi[k][v], k);
                }
            }
            println!("");
        }
    }

    // theta: MxK matrix
    fn print_doc_topics(&self) {
        unimplemented!();
    }

    // phi: KxV matrix
    fn print_topics(&self) {
        self.print_topics_by(|&v| v);
    }

    // phi: KxV matrix
    fn print_topics_with_vocab(&self, vocab: &[String]) {
        self.print_topics_by(|&v| &vocab[v]);
    }

    // phi: KxV matrix
    fn print_topics_by<T, F>(&self, mut f: F)
        where T: fmt::Display, F: FnMut(&usize) -> T
    {
        let num_topics = self.alpha.len();
        for k in 0..num_topics {
            print!("Topic {}:", k);
            let mut topic_vec: Vec<_> = self.phi[k].iter().enumerate().collect();
            topic_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            for (v, &prob) in topic_vec {
                if prob > 1e-6 {
                    print!(" {}*{}", prob, f(&v));
                }
            }
            println!("");
        }
    }
}

fn gibbs(dataset: &[Bag], alpha_init: &[f64], beta_init: &[f64], burn_in: usize, num_samples: usize) -> Model {
    let num_docs: usize = dataset.len();
    let num_topics: usize = alpha_init.len();
    let vocab_size: usize = beta_init.len();
    let mut rng = rand::thread_rng();
    let mut alpha = alpha_init.to_vec();
    let mut beta = beta_init.to_vec();
    println!("K = {}", num_topics);
    println!("M = {}", num_docs);
    println!("V = {}", vocab_size);

    write!(&mut std::io::stderr(), "Initializing...").unwrap();

    // Construct zero-filled nested arrays
    let mut w:     Vec<Vec<usize>> = Vec::with_capacity(num_docs);
    let mut z:     Vec<Vec<usize>> = Vec::with_capacity(num_docs);
    let mut theta: Vec<Vec<f64>>   = Vec::with_capacity(num_docs);
    let mut phi:   Vec<Vec<f64>>   = Vec::with_capacity(num_topics);
    let mut ndk:   Vec<Vec<usize>> = Vec::with_capacity(num_docs);
    let mut nkv:   Vec<Vec<usize>> = Vec::with_capacity(num_topics);
    let mut nd:    Vec<usize>      = Vec::with_capacity(num_docs);
    let mut nk:    Vec<usize>      = Vec::with_capacity(num_topics);
    let mut z_samples: Vec<Vec<Vec<usize>>> = Vec::with_capacity(num_docs);
    for bag in dataset {
        // n_d
        let mut n_d = 0;
        for &count in bag.values() {
            n_d += count;
        }
        w.push(vec![0; n_d]);
        z.push(vec![0; n_d]);
        theta.push(vec![0.0; num_topics]);
        ndk.push(vec![0; num_topics]);
        nd.push(0);
        z_samples.push(vec![vec![0; num_topics]; n_d]);
    }
    for _k in 0..num_topics {
        phi.push(vec![0.0; vocab_size]);
        nkv.push(vec![0; vocab_size]);
        nk.push(0);
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
                ndk[d][k] += 1;
                nkv[k][v] += 1;
                nd[d] += 1;
                nk[k] += 1;
                i += 1;
            }
        }
    }
    // Initialize theta, phi
    {
        let dir_a = Dirichlet::new(alpha.clone());
        for d in 0..num_docs {
            theta[d] = dir_a.ind_sample(&mut rng);
        }
    }
    {
        let dir_b = Dirichlet::new(beta.clone());
        for k in 0..num_topics {
            phi[k] = dir_b.ind_sample(&mut rng);
        }
    }

    println!("z = {:?}", z);
    println!("theta = {:?}", theta);
    println!("ndk = {:?}", ndk);
    println!("phi = {:?}", phi);
    println!("nkv = {:?}", nkv);

    writeln!(&mut std::io::stderr(), "\rInitialized.").unwrap();

    // Sampling
    write!(&mut std::io::stderr(), "Sampling...").unwrap();
    for s in 0..(burn_in + num_samples) {
        let alpha_sum: f64 = alpha.iter().sum();
        let beta_sum: f64 = beta.iter().sum();
        for (d, w_d) in w.iter().enumerate() {
            for (i, &w_di) in w_d.iter().enumerate() {
                let v = w_di;
                // Sample z_di
                let mut weights: Vec<f64> = Vec::with_capacity(num_topics);
                for k in 0..num_topics {
                    weights.push(theta[d][k] * phi[k][v]);
                }
                let cat = Categorical::new(weights);
                let old_z_di = z[d][i];
                let new_z_di = cat.ind_sample(&mut rng);
                z[d][i] = new_z_di;
                if s >= burn_in {
                    z_samples[d][i][new_z_di] += 1;
                }
                // Update ndk and nkv
                ndk[d][old_z_di] -= 1;
                ndk[d][new_z_di] += 1;
                nkv[old_z_di][v] -= 1;
                nkv[new_z_di][v] += 1;
                nk[old_z_di] -= 1;
                nk[new_z_di] += 1;
            }
            // Sample theta_d
            let mut alpha_d: Vec<f64> = Vec::with_capacity(num_topics);
            for k in 0..num_topics {
                alpha_d.push(ndk[d][k] as f64 + alpha[k]);
            }
            let dir = Dirichlet::new(alpha_d);
            theta[d] = dir.ind_sample(&mut rng);
        }
        for k in 0..num_topics {
            // Sample phi_k
            let mut beta_k: Vec<f64> = Vec::with_capacity(vocab_size);
            for v in 0..vocab_size {
                beta_k.push(nkv[k][v] as f64 + beta[v]);
            }
            let dir = Dirichlet::new(beta_k);
            phi[k] = dir.ind_sample(&mut rng);
        }
        // Update alpha and beta
        for k in 0..num_topics {
            let mut x = -(num_docs as f64) * digamma(alpha[k]);
            let mut y = -(num_docs as f64) * digamma(alpha_sum);
            for d in 0..num_docs {
                x += digamma(ndk[d][k] as f64 + alpha[k]);
                y += digamma(nd[d] as f64 + alpha_sum);
            }
            alpha[k] = alpha[k] * x / y;
        }
        for v in 0..vocab_size {
            let mut x = -(num_topics as f64) * digamma(beta[v]);
            let mut y = -(num_topics as f64) * digamma(beta_sum);
            for k in 0..num_topics {
                x += digamma(nkv[k][v] as f64 + beta[v]);
                y += digamma(nk[k] as f64 + beta_sum);
            }
            beta[v] = beta[v] * x / y;
        }
        // Evaluate the log-likelihood value for the current parameters
        let mut log_likelihood = 0.0;
        for k in 0..num_topics {
            for v in 0..vocab_size {
                log_likelihood += nkv[k][v] as f64 * f64::ln(phi[k][v]);
            }
        }
        println!("log_likelihood = {}", log_likelihood);
        if s < burn_in {
            write!(&mut std::io::stderr(), "\rBurn-in... {}/{}", s + 1, burn_in).unwrap();
        }
        else {
            if s == burn_in {
                writeln!(&mut std::io::stderr(), "").unwrap();
            }
            write!(&mut std::io::stderr(), "\rSampling... {}/{}", s - burn_in + 1, num_samples).unwrap();
        }
    }
    writeln!(&mut std::io::stderr(), "\rSampled.").unwrap();

    // Distribution of z
    for (d, z_samples_d) in z_samples.iter().enumerate() {
        for (i, samples) in z_samples_d.iter().enumerate() {
            let mut sum = 0.0;
            for k in 0..num_topics {
                sum += samples[k] as f64;
            }
            print!("z[{:3}][{:3}] = ", d, i);
            for k in 0..num_topics {
                print!("{:3}:{:.2} ", k, samples[k] as f64 / sum);
            }
            println!("");
        }
    }

    Model {
        alpha_init:  alpha_init.to_vec(),
        beta_init:   beta_init.to_vec(),
        burn_in:     burn_in,
        num_samples: num_samples,
        alpha:       alpha,
        beta:        beta,
        theta:       theta,
        phi:         phi,
        z_samples:   z_samples,
    }
}

fn collapsed_gibbs(dataset: &[Bag], alpha_init: &[f64], beta_init: &[f64], burn_in: usize, num_samples: usize) -> Model {
    let num_docs: usize = dataset.len();
    let num_topics: usize = alpha_init.len();
    let vocab_size: usize = beta_init.len();
    let mut rng = rand::thread_rng();
    let mut alpha = alpha_init.to_vec();
    let mut beta = beta_init.to_vec();
    // println!("K = {}", num_topics);
    // println!("M = {}", num_docs);
    // println!("V = {}", vocab_size);

    write!(&mut std::io::stderr(), "Initializing...").unwrap();

    // Construct zero-filled nested arrays
    let mut w:     Vec<Vec<usize>> = Vec::with_capacity(num_docs);
    let mut z:     Vec<Vec<usize>> = Vec::with_capacity(num_docs);
    let mut theta: Vec<Vec<f64>>   = Vec::with_capacity(num_docs);
    let mut phi:   Vec<Vec<f64>>   = Vec::with_capacity(num_topics);
    let mut ndk:   Vec<Vec<usize>> = Vec::with_capacity(num_docs);
    let mut nkv:   Vec<Vec<usize>> = Vec::with_capacity(num_topics);
    let mut nd:    Vec<usize>      = Vec::with_capacity(num_docs);
    let mut nk:    Vec<usize>      = Vec::with_capacity(num_topics);
    let mut z_samples: Vec<Vec<Vec<usize>>> = Vec::with_capacity(num_docs);
    for bag in dataset {
        // n_d
        let mut n_d = 0;
        for &count in bag.values() {
            n_d += count;
        }
        w.push(vec![0; n_d]);
        z.push(vec![0; n_d]);
        theta.push(vec![0.0; num_topics]);
        ndk.push(vec![0; num_topics]);
        nd.push(0);
        z_samples.push(vec![vec![0; num_topics]; n_d]);
    }
    for _k in 0..num_topics {
        phi.push(vec![0.0; vocab_size]);
        nkv.push(vec![0; vocab_size]);
        nk.push(0);
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
                ndk[d][k] += 1;
                nkv[k][v] += 1;
                nd[d] += 1;
                nk[k] += 1;
                i += 1;
            }
        }
    }

    // println!("z = {:?}", z);
    // println!("ndk = {:?}", ndk);
    // println!("nkv = {:?}", nkv);

    writeln!(&mut std::io::stderr(), "\rInitialized.").unwrap();

    // Sampling
    write!(&mut std::io::stderr(), "Sampling...").unwrap();
    for s in 0..(burn_in + num_samples) {
        // println!("s = {}", s);
        let alpha_sum: f64 = alpha.iter().sum();
        let beta_sum: f64 = beta.iter().sum();
        for (d, w_d) in w.iter().enumerate() {
            for (i, &w_di) in w_d.iter().enumerate() {
                let v = w_di;
                //
                let old_z_di = z[d][i];
                ndk[d][old_z_di] -= 1;
                nkv[old_z_di][v] -= 1;
                nd[d] -= 1;
                nk[old_z_di] -= 1;
                // Sample z_di
                let mut weights: Vec<f64> = Vec::with_capacity(num_topics);
                for k in 0..num_topics {
                    let e_theta_dk = (ndk[d][k] as f64 + alpha[k]) / (nd[d] as f64 + alpha_sum);
                    let e_phi_kv = (nkv[k][v] as f64 + beta[v]) / (nk[k] as f64 + beta_sum);
                    weights.push(e_theta_dk * e_phi_kv);
                }
                // println!("{:?}", weights);
                let cat = Categorical::new(weights);
                let new_z_di = cat.ind_sample(&mut rng);
                z[d][i] = new_z_di;
                ndk[d][new_z_di] += 1;
                nkv[new_z_di][v] += 1;
                nd[d] += 1;
                nk[new_z_di] += 1;
                // Save the sample
                if s >= burn_in {
                    z_samples[d][i][new_z_di] += 1;
                }
            }
            // println!("{:?}", z);
        }
        // println!("{:?}", z);
        // Infer phi and theta
        for d in 0..num_docs {
            for k in 0..num_topics {
                theta[d][k] = (ndk[d][k] as f64 + alpha[k]) / (nd[d] as f64 + alpha_sum);
            }
        }
        for k in 0..num_topics {
            for v in 0..vocab_size {
                phi[k][v] = (nkv[k][v] as f64 + beta[v]) / (nk[k] as f64 + beta_sum);
            }
        }
        // Update alpha and beta
        for k in 0..num_topics {
            let mut x = -(num_docs as f64) * digamma(alpha[k]);
            let mut y = -(num_docs as f64) * digamma(alpha_sum);
            for d in 0..num_docs {
                x += digamma(ndk[d][k] as f64 + alpha[k]);
                y += digamma(nd[d] as f64 + alpha_sum);
            }
            alpha[k] = alpha[k] * x / y;
        }
        for v in 0..vocab_size {
            let mut x = -(num_topics as f64) * digamma(beta[v]);
            let mut y = -(num_topics as f64) * digamma(beta_sum);
            for k in 0..num_topics {
                x += digamma(nkv[k][v] as f64 + beta[v]);
                y += digamma(nk[k] as f64 + beta_sum);
            }
            beta[v] = beta[v] * x / y;
        }
        // Evaluate the log-likelihood value for the current parameters
        let mut log_likelihood = 0.0;
        for k in 0..num_topics {
            for v in 0..vocab_size {
                log_likelihood += nkv[k][v] as f64 * f64::ln(phi[k][v]);
            }
        }
        // println!("log_likelihood = {}", log_likelihood);
        if s < burn_in {
            write!(&mut std::io::stderr(), "\rBurn-in... {}/{}", s + 1, burn_in).unwrap();
        }
        else {
            if s == burn_in {
                writeln!(&mut std::io::stderr(), "").unwrap();
            }
            write!(&mut std::io::stderr(), "\rSampling... {}/{}", s - burn_in + 1, num_samples).unwrap();
        }
        println!("{} {}", s, log_likelihood);
    }
    writeln!(&mut std::io::stderr(), "\rSampled.").unwrap();

    // Distribution of z
    for (d, z_samples_d) in z_samples.iter().enumerate() {
        for (i, samples) in z_samples_d.iter().enumerate() {
            let mut sum = 0.0;
            for k in 0..num_topics {
                sum += samples[k] as f64;
            }
            // print!("z[{:3}][{:3}] = ", d, i);
            // for k in 0..num_topics {
            //     print!("{:3}:{:.2} ", k, samples[k] as f64 / sum);
            // }
            // println!("");
        }
    }

    Model {
        alpha_init:  alpha_init.to_vec(),
        beta_init:   beta_init.to_vec(),
        burn_in:     burn_in,
        num_samples: num_samples,
        alpha:       alpha,
        beta:        beta,
        theta:       theta,
        phi:         phi,
        z_samples:   z_samples,
    }
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
    let mut bags: Vec<Bag> = Vec::new();
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
    let mut bags: Vec<Bag> = Vec::new();
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
    let rev_id_map: HashMap<usize, usize> = ids.into_iter().enumerate().collect();
    // Map IDs
    for bag in &mut bags {
        *bag = bag.into_iter().map(|(id, &mut c)| (id_map[id], c)).collect();
    }
    (bags, rev_id_map.len(), rev_id_map)
}

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
        .version("0.1")
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
        .arg(Arg::with_name("INPUT")
             .help("Sets the input file to use")
             .required(false)
             .index(1)
             .requires("topics"))
        .arg(Arg::with_name("VOCAB")
             .help("Sets the input file to use")
             .required(false)
             .index(2))
        .setting(AppSettings::ArgRequiredElseHelp)
        .get_matches();

    let learn = {
        let method = value_t_or_exit!(matches, "method", Method);
        match method {
            Method::Gibbs          => gibbs,
            Method::CollapsedGibbs => collapsed_gibbs,
        }
    };

    let burn_in = value_t_or_exit!(matches, "burn-in", usize);
    let samples = value_t_or_exit!(matches, "samples", usize);

    if matches.is_present("test-dataset") {
        let num_topics = 10;
        let vocab_size = 10000;
        let alpha: Vec<f64> = vec![0.1; num_topics];
        let beta: Vec<f64> = vec![0.1; vocab_size];
        write!(&mut std::io::stderr(), "Generating a dataset...").unwrap();
        let dataset = make_dataset(1000, f64::ln(400f64), 0.3, &alpha, &beta);
        writeln!(&mut std::io::stderr(), " done.").unwrap();
        write!(&mut std::io::stderr(), "Compacting the dataset...").unwrap();
        let (dataset, vocab_size, rev_id_map) = compact_words(dataset);
        writeln!(&mut std::io::stderr(), " done.").unwrap();
        writeln!(&mut std::io::stderr(), "Vocab: {}", vocab_size).unwrap();
        let beta: Vec<f64> = vec![0.1; vocab_size];

        let model = learn(&dataset, &alpha, &beta, burn_in, samples);

        model.print_term_topics_by(|id| rev_id_map[id]);
        model.print_topics_by(|id| rev_id_map[id]);

        if let Some(fp) = matches.value_of("model") {
            let mut file = File::create(&fp).unwrap();
            serde_json::to_writer_pretty(&mut file, &model).unwrap();
        }
    }
    else if matches.is_present("test-visual-dataset") {
        let size = 5;
        let num_topics: usize = size + size;
        write!(&mut std::io::stderr(), "Generating a dataset...").unwrap();
        let dataset = make_visual_dataset(size, 1000);
        writeln!(&mut std::io::stderr(), " done.").unwrap();
        write!(&mut std::io::stderr(), "Compacting the dataset...").unwrap();
        let (dataset, vocab_size, rev_id_map) = compact_words(dataset);
        writeln!(&mut std::io::stderr(), " done.").unwrap();
        writeln!(&mut std::io::stderr(), "Vocab: {}", vocab_size).unwrap();

        let alpha: Vec<f64> = vec![1.0; num_topics];
        let beta: Vec<f64> = vec![1.0; vocab_size];

        let model = learn(&dataset, &alpha, &beta, burn_in, samples);
        model.print_term_topics_by(|id| rev_id_map[id]);
        model.print_topics_by(|id| rev_id_map[id]);
        println!("alpha = {:?}", model.alpha);
        println!("beta = {:?}", model.beta);

        if let Some(fp) = matches.value_of("model") {
            let mut file = File::create(&fp).unwrap();
            serde_json::to_writer_pretty(&mut file, &model).unwrap();
        }
    }
    else if let Some(input_fp) = matches.value_of("INPUT") {
        let (dataset, _) = load_bags(input_fp).unwrap();
        let vocab: Option<Vec<String>> = if let Some(vocab_fp) = matches.value_of("VOCAB") {
            Some(load_text_vocabulary(vocab_fp).unwrap())
        }
        else {
            None
        };
        write!(&mut std::io::stderr(), "Compacting the dataset...").unwrap();
        let (dataset, vocab_size, rev_id_map) = compact_words(dataset);
        writeln!(&mut std::io::stderr(), " done.").unwrap();
        let num_topics = value_t_or_exit!(matches, "topics", usize);
        let alpha: Vec<f64> = vec![0.1; num_topics];
        let beta: Vec<f64> = vec![0.1; vocab_size];

        let model = learn(&dataset, &alpha, &beta, burn_in, samples);

        match vocab {
            Some(vocab) => {
                model.print_term_topics_by(|id| &vocab[rev_id_map[id]]);
                model.print_topics_by(|id| &vocab[rev_id_map[id]]);
            },
            None => {
                model.print_term_topics_by(|id| rev_id_map[id]);
                model.print_topics_by(|id| rev_id_map[id]);
            }
        }

        if let Some(fp) = matches.value_of("model") {
            let mut file = File::create(&fp).unwrap();
            serde_json::to_writer_pretty(&mut file, &model).unwrap();
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
        let (_, _, rev_id_map) = ::compact_words(dataset);
        let mut map_pairs: Vec<(usize, usize)> = rev_id_map.into_iter().collect();
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
        let (compacted_dataset, _, rev_id_map) = ::compact_words(dataset.clone());
        let (decompacted_dataset, _) = ::decompact_words(compacted_dataset, rev_id_map);
        assert_eq!(decompacted_dataset, dataset);
    }
}
