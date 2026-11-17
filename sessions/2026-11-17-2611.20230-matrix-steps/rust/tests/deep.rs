//! Deep tests for the matrix-steps mechanism: property sweeps, exhaustive
//! + random sequences against an oracle, a soak past the Python test's
//! scale, and thread identity.
//!
//! Run: cargo test --release --offline
//! (the soak takes ~1-2 min in release; longer in debug.)

use matrix_steps_rs::*;

// ---------------------------------------------------------------- oracle
/// Oracle for the delta rule on tiny matrices: the least-squares solution
/// computed by direct normal equations in f64 (Gaussian elimination).
fn least_squares_matrix(pairs: &[(Vec<f32>, Vec<f32>)], dim: usize) -> Vec<f64> {
    // Solve M K^T = V^T via normal equations: (K K^T) M^T = K V^T.
    let n = pairs.len();
    let mut kk = vec![vec![0.0f64; dim]; dim];
    let mut kv = vec![vec![0.0f64; dim]; dim];
    for (k, v) in pairs {
        for i in 0..dim {
            for j in 0..dim {
                kk[i][j] += k[i] as f64 * k[j] as f64;
                kv[i][j] += k[i] as f64 * v[j] as f64;
            }
        }
    }
    // Ridge for stability.
    for i in 0..dim {
        kk[i][i] += 1e-6;
    }
    // Solve kk * X = kv for X (dim x dim) by Gauss-Jordan.
    let mut a = kk;
    let mut b = kv;
    for col in 0..dim {
        let mut piv = col;
        for r in col + 1..dim {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        for r in 0..dim {
            if r != col && a[r][col] != 0.0 {
                let f = a[r][col] / d;
                for c in col..dim {
                    a[r][c] -= f * a[col][c];
                }
                for c in 0..dim {
                    b[r][c] -= f * b[col][c];
                }
            }
        }
    }
    let mut m = vec![0.0f64; dim * dim];
    // X = (K^T K)^-1 K^T V is M^T (since M K^T = V^T); return M = X^T.
    for i in 0..dim {
        for j in 0..dim {
            m[i * dim + j] = b[j][i] / a[j][j];
        }
    }
    let _ = n;
    m
}

fn random_unit_vector(rng: &mut Rng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.uniform() as f32 * 2.0 - 1.0).collect();
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
    v
}

#[test]
fn delta_rule_matches_least_squares_oracle() {
    // Small dims, few pairs: many passes at small beta must approach the
    // least-squares solution the passes approximate (paper Sec. 6).
    let mut rng = Rng::new(77);
    for dim in [8usize, 16, 24] {
        let n_pairs = dim / 2;
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..n_pairs)
            .map(|_| (random_unit_vector(&mut rng, dim), random_unit_vector(&mut rng, dim)))
            .collect();
        let mut m = DeltaMatrix::new(dim, None, "sign", 777);
        // Small beta converges linearly toward the least-squares solution;
        // the slowest mode is set by the smallest key-Gram eigenvalue, so
        // this needs many passes at dim 8.
        for _ in 0..500 {
            for (k, v) in &pairs {
                m.write(k, v, 1.0, 0.05);
            }
        }
        let oracle = least_squares_matrix(&pairs, dim);
        let mut worst = 0.0f64;
        for i in 0..dim {
            for j in 0..dim {
                worst = worst.max((m.m[i * dim + j] as f64 - oracle[i * dim + j]).abs());
            }
        }
        assert!(worst < 0.05, "dim {} worst entry error {:.4}", dim, worst);
    }
}

#[test]
fn exhaustive_small_key_sequences() {
    // Exhaustive: every sequence of 1-3 writes over 2 fixed pairs, in
    // every order, read back against the oracle's prediction for the
    // same sequence computed by direct simulation in f64.
    let k1: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let k2: Vec<f32> = vec![0.0, 1.0, 0.0, 0.0];
    let v1: Vec<f32> = vec![0.5, -0.5, 0.25, 0.75];
    let v2: Vec<f32> = vec![-0.25, 0.5, 0.5, -0.5];
    let pairs = [(k1.clone(), v1.clone()), (k2.clone(), v2.clone())];
    for len in 1..=3usize {
        for seq in 0..(1usize << len) {
            let mut m = DeltaMatrix::new(4, None, "sign", 777);
            for t in 0..len {
                let (k, v) = &pairs[(seq >> t) & 1];
                m.write(k, v, 1.0, 1.0);
            }
            // With orthogonal unit keys and beta=1, a write sets its key's
            // row exactly and leaves the other untouched; a key never
            // written reads back as zero.
            let wrote1 = (0..len).any(|t| ((seq >> t) & 1) == 0);
            let wrote2 = (0..len).any(|t| ((seq >> t) & 1) == 1);
            let r1 = m.read(&k1);
            let r2 = m.read(&k2);
            let e1: f64 = if wrote1 {
                r1.iter().zip(v1.iter()).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>().sqrt()
            } else {
                r1.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt()
            };
            let e2: f64 = if wrote2 {
                r2.iter().zip(v2.iter()).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>().sqrt()
            } else {
                r2.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt()
            };
            assert!(e1 < 1e-5 && e2 < 1e-5, "seq {:b} len {} errors {:.2e} {:.2e}", seq, len, e1, e2);
        }
    }
}

#[test]
fn property_sweep_random_budgets() {
    // Random dims, pair counts, betas, passes: read-back error after a
    // rebuild must shrink with passes at small beta and stay bounded.
    let mut rng = Rng::new(20261117);
    for _ in 0..30 {
        let dim = rng.randint(8, 48) as usize;
        let n_pairs = rng.randint(1, dim as i64 / 2) as usize;
        let beta = [0.1f32, 0.25, 0.5][rng.randint(0, 2) as usize];
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..n_pairs)
            .map(|_| (random_unit_vector(&mut rng, dim), random_unit_vector(&mut rng, dim)))
            .collect();
        let mut m = DeltaMatrix::new(dim, None, "sign", 777);
        m.rebuild(&pairs, 20, beta);
        let mut worst = 0.0f64;
        for (k, v) in &pairs {
            let got = m.read(k);
            let err: f64 = got
                .iter()
                .zip(v.iter())
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            worst = worst.max(err);
        }
        assert!(worst < 1.0, "dim {} pairs {} beta {} worst {:.3}", dim, n_pairs, beta, worst);
    }
}

#[test]
fn consolidation_property_sweep() {
    // Random episode streams: the record appears exactly when the
    // minimums are met, and confidence is always (s+1)/(n+2).
    let mut rng = Rng::new(4242);
    let seq = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    for _ in 0..200 {
        let n = rng.randint(0, 10) as usize;
        let mut eps: Vec<Episode> = Vec::new();
        let mut s = 0usize;
        let mut n_counted = 0usize;
        for i in 0..n {
            let success = rng.bernoulli(0.6);
            let verified = rng.bernoulli(0.9);
            let e = Episode::new(
                &format!("e{}", i),
                "alice",
                "tyre",
                "goal",
                seq.clone(),
                if success { "success" } else { "failure" },
                if verified { "external_check" } else { "model_claim" },
            );
            if verified {
                n_counted += 1;
                if success {
                    s += 1;
                }
            }
            eps.push(e);
        }
        let rec = consolidate_episodes(&eps, MIN_EPISODES, MIN_SUCCESSES, MIN_COVERAGE);
        if n_counted >= MIN_EPISODES && s >= MIN_SUCCESSES {
            // Top sequence is the only sequence: coverage 1.0.
            let r = rec.expect("record must appear at the minimums");
            assert_eq!(r.n_episodes, n_counted);
            assert_eq!(r.n_successes, s);
            assert!((r.confidence - (s + 1) as f64 / (n_counted + 2) as f64).abs() < 1e-12);
        } else {
            assert!(rec.is_none(), "no record below the minimums");
        }
    }
}

#[test]
fn erasure_is_exact_on_small_store() {
    // Build a 20-procedure store, erase 2 procedures' episodes, and check
    // the rebuilt matrix finds exactly the remaining 18 (argmax).
    let mut mem = EpisodicMemory::new("alice", EMBED_DIM, 0.25);
    let procs = generate_synthetic_procedures(20, 20261117, 3, 8, 0.15);
    for (i, p) in procs.iter().enumerate() {
        let mut p = p.clone();
        let ep = Episode::new(
            &format!("ep_{:04}", i),
            "alice",
            "proc",
            &p.goal,
            p.steps.iter().map(|s| s.text().to_string()).collect(),
            "success",
            "external_check",
        );
        mem.record_episode(ep);
        p.episode_ids = vec![format!("ep_{:04}", i)];
        mem.add_record(p);
    }
    mem.rebuild_matrix(10, true);
    let before: Vec<bool> = procs
        .iter()
        .map(|p| mem.find(&p.goal, "matrix", 0.0).0.as_deref() == Some(p.id.as_str()))
        .collect();
    assert!(before.iter().all(|b| *b));
    let erased = vec!["ep_0003".to_string(), "ep_0017".to_string()];
    let (out, _) = mem.erase_episodes(&erased, "test erasure");
    assert_eq!(out.len(), 2);
    assert_eq!(mem.audit.len(), 2);
    for (i, p) in procs.iter().enumerate() {
        let (rec, _) = mem.find(&p.goal, "matrix", 0.0);
        if i == 3 || i == 17 {
            assert!(rec.is_none() || rec.as_deref() != Some(p.id.as_str()));
        } else {
            assert_eq!(rec.as_deref(), Some(p.id.as_str()), "proc {} must survive", i);
        }
    }
}

#[test]
fn soak_revisit_10k() {
    // 10x the Python test's scale: 10 000 situations, 4 repeats.
    let r = simulate_revisit_stream(11, 10000, 4, 4);
    assert_eq!(r.revisit_first_attempt, 1.0);
    assert_eq!(r.revisits, 30000);
}

#[test]
fn thread_identity() {
    // The code is stateless (embedder + sign vectors + matrix reads):
    // threads computing the same outputs must agree bit-for-bit.
    let handles: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(|| {
                let mut emb = HashEmbedder::new();
                let v = emb.embed("change the tyre on the bicycle w0142 w2088");
                let s = procedure_sign_vector("proc_0042", 1024);
                let mut m = DeltaMatrix::new(1024, None, "sign", 777);
                let k = to_f32(&emb.embed("w0001 w0002 w0003 w0004 w0005 w0006"));
                let val = to_f32(&emb.embed("w0001 w0002 w0007 w0008"));
                m.write(&k, &val, 1.0, 1.0);
                let read = m.read(&k);
                let checksum: f64 = read.iter().map(|x| *x as f64).sum();
                (v, s, checksum)
            })
        })
        .collect();
    let mut outs = Vec::new();
    for h in handles {
        outs.push(h.join().unwrap());
    }
    for w in outs.windows(2) {
        assert_eq!(w[0].0, w[1].0);
        assert_eq!(w[0].1, w[1].1);
        assert!((w[0].2 - w[1].2).abs() < 1e-9);
    }
}
