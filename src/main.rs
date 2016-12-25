use std::string::String;
use std::collections::HashMap;
use std::vec::Vec;
use std::fs::{File};
use std::env;
use std::path::{Path};
use std::io::{self, BufReader, BufRead};


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
            bag.insert(index, value);
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
}
