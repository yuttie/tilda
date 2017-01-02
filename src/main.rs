extern crate rand;

use std::string::String;
use std::collections::HashMap;
use std::vec::Vec;
use std::fs::{File};
use std::env;
use std::path::{Path};
use std::io::{self, BufReader, BufRead, Write};
use std::iter;

use rand::{Closed01, Rng};
use rand::distributions::{IndependentSample, Sample, Gamma, LogNormal, RandSample, Range};


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
    let num_docs: usize = dataset.len();
    let num_topics: usize = alpha.len();
    let vocab_size: usize = beta.len();
    let mut rng = rand::thread_rng();
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
    let mut z_samples: Vec<Vec<Vec<usize>>> = Vec::with_capacity(num_docs);
    for bag in &dataset {
        // n_d
        let mut n_d = 0;
        for &count in bag.values() {
            n_d += count;
        }
        w.push(vec![0; n_d]);
        z.push(vec![0; n_d]);
        theta.push(vec![0.0; num_topics]);
        ndk.push(vec![0; num_topics]);
        z_samples.push(vec![vec![0; num_topics]; n_d]);
    }
    for k in 0..num_topics {
        phi.push(vec![0.0; vocab_size]);
        nkv.push(vec![0; vocab_size]);
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
                i += 1;
            }
        }
    }
    // Initialize theta, phi
    for d in 0..num_docs {
        let dir = Dirichlet::new(alpha.clone());
        theta[d] = dir.ind_sample(&mut rng);
    }
    for k in 0..num_topics {
        let dir = Dirichlet::new(beta.clone());
        phi[k] = dir.ind_sample(&mut rng);
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
}

fn lda_collapsed(dataset: Vec<Bag>, alpha: Vec<f64>, beta: Vec<f64>, burn_in: usize, num_samples: usize) {
    let num_docs: usize = dataset.len();
    let num_topics: usize = alpha.len();
    let vocab_size: usize = beta.len();
    let mut rng = rand::thread_rng();
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
    let mut z_samples: Vec<Vec<Vec<usize>>> = Vec::with_capacity(num_docs);
    for bag in &dataset {
        // n_d
        let mut n_d = 0;
        for &count in bag.values() {
            n_d += count;
        }
        w.push(vec![0; n_d]);
        z.push(vec![0; n_d]);
        theta.push(vec![0.0; num_topics]);
        ndk.push(vec![0; num_topics]);
        z_samples.push(vec![vec![0; num_topics]; n_d]);
    }
    for k in 0..num_topics {
        phi.push(vec![0.0; vocab_size]);
        nkv.push(vec![0; vocab_size]);
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
        for (d, w_d) in w.iter().enumerate() {
            for (i, &w_di) in w_d.iter().enumerate() {
                let v = w_di;
                //
                let old_z_di = z[d][i];
                ndk[d][old_z_di] -= 1;
                nkv[old_z_di][v] -= 1;
                // Sample z_di
                let mut weights: Vec<f64> = Vec::with_capacity(num_topics);
                for k in 0..num_topics {
                    let E_theta_dk = {
                        let mut sum = 0.0;
                        for k in 0..num_topics {
                            sum += ndk[d][k] as f64 + alpha[k];
                        }
                        (ndk[d][k] as f64 + alpha[k]) / sum
                    };

                    let E_phi_kv = {
                        let mut sum = 0.0;
                        for v in 0..vocab_size {
                            sum += nkv[k][v] as f64 + beta[v];
                        }
                        (nkv[k][v] as f64 + beta[v]) / sum
                    };

                    weights.push(E_theta_dk * E_phi_kv);
                }
                // println!("{:?}", weights);
                let cat = Categorical::new(weights);
                let new_z_di = cat.ind_sample(&mut rng);
                z[d][i] = new_z_di;
                ndk[d][new_z_di] += 1;
                nkv[new_z_di][v] += 1;
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
                theta[d][k] = {
                    let mut sum = 0.0;
                    for k in 0..num_topics {
                        sum += ndk[d][k] as f64 + alpha[k];
                    }
                    (ndk[d][k] as f64 + alpha[k]) / sum
                };
            }
        }
        for k in 0..num_topics {
            for v in 0..vocab_size {
                phi[k][v] = {
                    let mut sum = 0.0;
                    for v in 0..vocab_size {
                        sum += nkv[k][v] as f64 + beta[v];
                    }
                    (nkv[k][v] as f64 + beta[v]) / sum
                };
            }
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
}

fn make_dataset(num_docs: usize, mean_nd: f64, std_dev_nd: f64, alpha: Vec<f64>, beta: Vec<f64>) -> Vec<Bag> {
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
    let lognorm = LogNormal::new(mean_nd, std_dev_nd);
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
