extern crate rand;

use std::string::String;
use std::collections::HashMap;
use std::vec::Vec;
use std::fs::{File};
use std::env;
use std::path::{Path};
use std::io::{self, BufReader, BufRead};

use rand::distributions::{IndependentSample, Range};


type Bag = HashMap<usize, usize>;

fn load_bags<P: AsRef<Path>>(path: P) -> io::Result<(Vec<usize>, Vec<Bag>)> {
    let mut bags = Vec::new();
    let mut labels = Vec::new();
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
            bag.insert(index - 1, value);
        }
        labels.push(label);
        bags.push(bag);
    }
    Ok((labels, bags))
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

fn lda(num_topics: usize, dataset: Vec<Bag>, alpha: Vec<f64>, beta: Vec<f64>) {
    // Initialization
    let mut rng = rand::thread_rng();
    let topics = Range::new(0, num_topics);

    let vocab_size: usize = {
        let mut have_some = false;
        let mut max_index = 0;
        for bag in &dataset {
            for &index in bag.keys() {
                have_some = true;
                if index > max_index {
                    max_index = index;
                }
            }
        }
        if have_some {
            max_index + 1
        }
        else {
            0
        }
    };
    println!("K = {}", num_topics);
    println!("M = {}", &dataset.len());
    println!("V = {}", vocab_size);

    let mut z: Vec<Vec<usize>> = Vec::new();
    let mut theta: Vec<Vec<f64>> = Vec::new();
    let mut ndk: Vec<Vec<usize>> = Vec::new();
    for bag in &dataset {
        // z
        let mut z_d = Vec::new();
        for &count in bag.values() {
            for _ in 0..count {
                z_d.push(0);
            }
        }
        let n_d = z_d.len();
        z.push(z_d);
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
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let (labels, dataset) = load_bags(&args[1]).unwrap();
    let vocab: Option<Vec<String>> = if args.len() < 3 {
        None
    }
    else {
        Some(load_text_vocabulary(&args[2]).unwrap())
    };
    println!("{:?}", labels);
    println!("{:?}", dataset);
    println!("{:?}", vocab);

    lda(3, dataset, vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]);
}
