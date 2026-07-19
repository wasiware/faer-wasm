use faer_wasm_blas::l1::*;

#[test]
fn rotg_identities() {
	let cases = [
		(3.0, 4.0),
		(4.0, 3.0),
		(-3.0, 4.0),
		(3.0, -4.0),
		(-3.0, -4.0),
		(5.0, 0.0),
		(0.0, 5.0),
		(0.0, -5.0),
		(1e300, 1e300),   // would overflow unguarded
		(1e-308, 1e-308), // r² would underflow unguarded (normal-range inputs)
		(1.0, 1e-200),
	];
	for (a, b) in cases {
		let g = drotg(a, b);
		let hyp = (a / g.r).hypot(b / g.r); // c² + s² via stable hypot
		assert!((hyp - 1.0).abs() < 1e-12, "({a},{b}): c²+s² = {hyp}");
		// the rotation maps (a,b) to (r,0)
		let r1 = g.c * a + g.s * b;
		let z = g.c * b - g.s * a;
		assert!(
			(r1 - g.r).abs() <= 1e-12 * g.r.abs().max(f64::MIN_POSITIVE),
			"({a},{b}): c·a+s·b = {r1}, r = {}",
			g.r
		);
		assert!(
			z.abs() <= 1e-12 * g.r.abs().max(f64::MIN_POSITIVE),
			"({a},{b}): residual {z}"
		);
		// r carries the sign of the larger-magnitude input
		let roe = if a.abs() > b.abs() { a } else { b };
		assert_eq!(g.r < 0.0, roe < 0.0, "({a},{b}): sign of r");
	}
	// subnormal inputs: reference drotg legitimately loses precision
	// (subnormals carry ~13 bits) — require only a sane, finite result
	let g = drotg(1e-320, 1e-320);
	assert!(g.r.is_finite() && g.r > 0.0);
	assert!((g.c * g.c + g.s * g.s - 1.0).abs() < 1e-3, "subnormal: c²+s² far off");

	// the zero case: identity rotation
	let g = drotg(0.0, 0.0);
	assert_eq!((g.c, g.s, g.r), (1.0, 0.0, 0.0));
	// classic 3-4-5 exactness
	let g = drotg(3.0, 4.0);
	assert!((g.r - 5.0).abs() < 1e-15 && (g.c - 0.6).abs() < 1e-15 && (g.s - 0.8).abs() < 1e-15);
}
