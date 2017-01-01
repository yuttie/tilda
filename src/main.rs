extern crate rand;

use std::string::String;
use std::collections::HashMap;
use std::vec::Vec;
use std::fs::{File};
use std::env;
use std::path::{Path};
use std::io::{self, BufReader, BufRead};
use std::iter;

use rand::{Closed01, Rng};
use rand::distributions::{IndependentSample, Sample, Gamma, LogNormal, RandSample};


type Bag = HashMap<usize, usize>;

fn load_bags<P: AsRef<Path>>(path: P) -> io::Result<(Vec<usize>, Vec<Bag>, usize)> {
    let mut bags = Vec::new();
    let mut labels = Vec::new();
    let mut vocab_size = 0;
    let file = try!(File::open(path));
    let file = BufReader::new(file);
    for line in file.lines() {
        let line = line.unwrap();
        let mut iter = line.split_whitespace();
        let label = iter.next().unwrap().parse::<usize>().unwrap();
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
        labels.push(label);
        bags.push(bag);
    }
    Ok((labels, bags, vocab_size))
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
    fn new(weights: Vec<f64>) -> Categorical {
        let sum: f64 = weights.iter().sum();
        let prop: Vec<f64> = weights.iter().map(|w| w / sum).collect();
        Categorical {
            prop: prop,
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

fn lda(dataset: Vec<Bag>, alpha: Vec<f64>, beta: Vec<f64>, burn_in: usize, num_samples: usize) {
    // Initialization
    let num_topics: usize = alpha.len();
    let vocab_size: usize = beta.len();
    println!("K = {}", num_topics);
    println!("M = {}", &dataset.len());
    println!("V = {}", vocab_size);

    let mut w: Vec<Vec<usize>> = Vec::new();
    let mut z: Vec<Vec<usize>> = Vec::new();
    let mut z_samples: Vec<Vec<Vec<usize>>> = Vec::new();
    let mut theta: Vec<Vec<f64>> = Vec::new();
    let mut ndk: Vec<Vec<usize>> = Vec::new();
    for bag in &dataset {
        // w
        let mut w_d = Vec::new();
        for (&index, &count) in bag {
            for _ in 0..count {
                w_d.push(index);
            }
        }
        w.push(w_d);
        // z
        let mut z_d = Vec::new();
        let mut z_samples_d = Vec::new();
        for &count in bag.values() {
            for _ in 0..count {
                z_d.push(0);
                let zeros: Vec<usize> = iter::repeat(0).take(num_topics).collect();
                z_samples_d.push(zeros);
            }
        }
        let n_d = z_d.len();
        z.push(z_d);
        z_samples.push(z_samples_d);
        // theta
        let mut theta_d = Vec::with_capacity(num_topics);
        for _k in 0..num_topics {
            theta_d.push(1.0 / num_topics as f64);
        }
        theta.push(theta_d);
        // ndk
        let mut ndk_d = Vec::with_capacity(num_topics);
        ndk_d.push(n_d);
        for _k in 1..num_topics {
            ndk_d.push(0);
        }
        ndk.push(ndk_d);
    }
    println!("z = {:?}", z);
    println!("theta = {:?}", theta);
    println!("ndk = {:?}", ndk);

    let mut phi: Vec<Vec<f64>> = Vec::new();
    let mut nkv: Vec<Vec<usize>> = Vec::new();
    // phi
    for _k in 0..num_topics {
        let mut phi_k = Vec::new();
        for _v in 0..vocab_size {
            phi_k.push(1.0 / vocab_size as f64);
        }
        phi.push(phi_k);
    }
    // nkv
    let mut nkv_0 = Vec::new();
    for _v in 0..vocab_size {
        nkv_0.push(0);
    }
    for bag in &dataset {
        for (&index, &count) in bag {
            nkv_0[index] += count;
        }
    }
    nkv.push(nkv_0);
    for _k in 1..num_topics {
        let mut nkv_k = Vec::new();
        for _v in 0..vocab_size {
            nkv_k.push(0);
        }
        nkv.push(nkv_k);
    }
    println!("phi = {:?}", phi);
    println!("nkv = {:?}", nkv);

    // Sampling
    let mut rng = rand::thread_rng();
    for s in 0..(burn_in + num_samples) {
        for (d, w_d) in w.iter().enumerate() {
            for (i, &w_di) in w_d.iter().enumerate() {
                let v = w_di;
                // Sample z_di
                let mut weights: Vec<f64> = Vec::new();
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
            }
            // Sample theta_d
            let mut alpha_d: Vec<f64> = Vec::new();
            for k in 0..num_topics {
                alpha_d.push(ndk[d][k] as f64 + alpha[k]);
            }
            let dir = Dirichlet::new(alpha_d);
            theta[d] = dir.ind_sample(&mut rng);
        }
        for k in 0..num_topics {
            // Sample phi_k
            let mut beta_k: Vec<f64> = Vec::new();
            for v in 0..vocab_size {
                beta_k.push(nkv[k][v] as f64 + beta[v]);
            }
            let dir = Dirichlet::new(beta_k);
            phi[k] = dir.ind_sample(&mut rng);
        }
        // Evaluate the log-likelihood value for the current parameters
        let mut log_likelihood = 0.0;
        for k in 0..num_topics {
            for v in 0..vocab_size {
                log_likelihood += nkv[k][v] as f64 * f64::ln(phi[k][v]);
            }
        }
        println!("log_likelihood = {}", log_likelihood);
    }
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
}

fn make_dataset(num_docs: usize, alpha: Vec<f64>, beta: Vec<f64>) -> Vec<Bag> {
    let num_topics: usize = alpha.len();
    let vocab_size: usize = beta.len();
    let mut rng = rand::thread_rng();
    // phi
    let dir_beta = Dirichlet::new(beta);
    let mut phi: Vec<Vec<f64>> = Vec::with_capacity(num_topics);
    for k in 0..num_topics {
        // Sample phi_k
        phi.push(dir_beta.ind_sample(&mut rng));
    }
    // theta, nd, z, w
    let dir_alpha = Dirichlet::new(alpha);
    let lognorm = LogNormal::new(100.0, 30.0);
    let mut theta: Vec<Vec<f64>> = Vec::with_capacity(num_docs);
    let mut nd: Vec<usize> = Vec::with_capacity(num_docs);
    let mut z: Vec<Vec<usize>> = Vec::with_capacity(num_docs);
    let mut w: Vec<Vec<usize>> = Vec::with_capacity(num_docs);
    for d in 0..num_docs {
        theta.push(dir_alpha.ind_sample(&mut rng));
        nd.push(f64::floor(lognorm.ind_sample(&mut rng)) as usize);
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

fn main() {
    let args: Vec<String> = env::args().collect();
    let (labels, dataset, vocab_size) = load_bags(&args[1]).unwrap();
    let vocab: Option<Vec<String>> = if args.len() < 3 {
        None
    }
    else {
        Some(load_text_vocabulary(&args[2]).unwrap())
    };
    println!("{:?}", labels);
    println!("{:?}", dataset);
    println!("{:?}", vocab);

    let num_topics = 3;
    let alpha: Vec<f64> = iter::repeat(1.0).take(num_topics).collect();
    let beta: Vec<f64> = iter::repeat(1.0).take(vocab_size).collect();

    lda(dataset, alpha, beta, 1000, 10000);
}
