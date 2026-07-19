use faer_wasm_blas::l1::*;

#[test]
fn rotg_identities() {
	let cases = [
		(3.0f32, 4.0f32),
		(4.0, 3.0),
		(-3.0, 4.0),
		(3.0, -4.0),
		(-3.0, -4.0),
		(5.0, 0.0),
		(0.0, 5.0),
		(0.0, -5.0),
		(1e30, 1e30),   // would overflow unguarded (squares > f32::MAX)
		(1e-30, 1e-30), // r² would underflow unguarded (normal-range inputs)
		(1.0, 1e-20),
	];
	for (a, b) in cases {
		let g = srotg(a, b);
		let hyp = (a / g.r).hypot(b / g.r);
		assert!((hyp - 1.0).abs() < 5e-6, "({a},{b}): c²+s² = {hyp}");
		let r1 = g.c * a + g.s * b;
		let z = g.c * b - g.s * a;
		assert!(
			(r1 - g.r).abs() <= 5e-6 * g.r.abs().max(f32::MIN_POSITIVE),
			"({a},{b}): c·a+s·b = {r1}, r = {}",
			g.r
		);
		assert!(
			z.abs() <= 5e-6 * g.r.abs().max(f32::MIN_POSITIVE),
			"({a},{b}): residual {z}"
		);
		let roe = if a.abs() > b.abs() { a } else { b };
		assert_eq!(g.r < 0.0, roe < 0.0, "({a},{b}): sign of r");
	}
	// subnormal inputs (f32 subnormals below ~1.2e-38): reference srotg
	// legitimately loses precision — require only a sane, finite result
	let g = srotg(1e-42, 1e-42);
	assert!(g.r.is_finite() && g.r > 0.0);
	assert!((g.c * g.c + g.s * g.s - 1.0).abs() < 1e-2, "subnormal: c²+s² far off");

	let g = srotg(0.0, 0.0);
	assert_eq!((g.c, g.s, g.r), (1.0, 0.0, 0.0));
	let g = srotg(3.0, 4.0);
	assert!((g.r - 5.0).abs() < 1e-6 && (g.c - 0.6).abs() < 1e-6 && (g.s - 0.8).abs() < 1e-6);
}
