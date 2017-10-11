use distribution;
use source::Source;

/// An erlang distribution.
// We use the fact that the Erlang distribution is a special form of the
// gamma distribution.
#[derive(Clone, Copy)]
pub struct Erlang(distribution::Gamma);

impl Erlang {
    /// Create an erlang distribution with shape parametr `k` and rate parameter
    /// `l`.
    ///
    /// It should hold that `k > 0` and `l > 0`.
    #[inline]
    pub fn new(k: u64, l: f64) -> Self {
        should!(k > 0 && l > 0.0);
        Erlang(distribution::Gamma::new((k as f64), 1.0 / l))
    }

    /// Return the degrees of freedom parameter `k`.
    #[inline(always)]
    pub fn k(&self) -> u64 { self.0.k() as u64 }

    /// Return the rate parameter `l`.
    #[inline(always)]
    pub fn l(&self) -> f64 { 1.0 / self.0.theta() }
}

impl distribution::Continuous for Erlang {
    fn density(&self, x: f64) -> f64 {
        self.0.density(x)
    }
}

impl distribution::Distribution for Erlang {
    type Value = <distribution::Gamma as distribution::Distribution>::Value;

    fn distribution(&self, x: f64) -> f64 {
        self.0.distribution(x)
    }
}

impl distribution::Entropy for Erlang {
    fn entropy(&self) -> f64 {
        self.0.entropy()
    }
}

impl distribution::Kurtosis for Erlang {
    #[inline]
    fn kurtosis(&self) -> f64 {
        self.0.kurtosis()
    }
}

impl distribution::Mean for Erlang {
    #[inline]
    fn mean(&self) -> f64 {
        self.0.mean()
    }
}

impl distribution::Modes for Erlang {
    fn modes(&self) -> Vec<f64> {
        self.0.modes()
    }
}

impl distribution::Sample for Erlang {
    #[inline]
    fn sample<S>(&self, source: &mut S) -> f64 where S: Source {
        self.0.sample(source)
    }
}

impl distribution::Skewness for Erlang {
    #[inline]
    fn skewness(&self) -> f64 {
        self.0.skewness()
    }
}

impl distribution::Variance for Erlang {
    #[inline]
    fn variance(&self) -> f64 {
        self.0.variance()
    }
}

#[cfg(test)]
mod tests {
    use assert;
    use prelude::*;

    macro_rules! new(
        ($k:expr, $l:expr) => (Erlang::new($k, $l));
    );

    #[test]
    fn density() {
        let d = new!(2, 0.5);
        let x = vec![
            -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
             4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0,
        ];
        let p = vec![
            0.0000000000000000e+00, 0.0000000000000000e+00, 0.9735009788392561e-01,
            0.1516326649281584e+00, 0.1771374572778805e+00, 0.1839397205857212e+00,
            0.1790654980376188e+00, 0.1673476201113224e+00, 0.1520522005191395e+00,
            0.1353352832366127e+00, 0.1185741276320974e+00, 0.1026062482798735e+00,
            0.8790080915922291e-01, 0.7468060255179591e-01, 0.6300808772654827e-01,
            0.5284542098905738e-01, 0.4409577348001708e-01, 0.3663127777746836e-01,
            0.3031149705662342e-01, 0.2499524221104519e-01,
        ];

        assert::close(&x.iter().map(|&x| d.density(x)).collect::<Vec<_>>(), &p, 1e-14);
    }

    #[test]
    fn distribution() {
        let d = new!(2, 0.5);
        let x = vec![
            -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0
        ];
        let p = vec![
            0.0000000000000000e+00, 0.0000000000000000e+00, 0.9020401043104986e-01,
            0.2642411176571154e+00, 0.4421745996289254e+00, 0.5939941502901619e+00,
            0.7127025048163542e+00, 0.8008517265285442e+00, 0.8641117745995667e+00,
            0.9084218055563291e+00, 0.9389005190396673e+00, 0.9595723180054872e+00,
            0.9734359856499836e+00, 0.9826487347633355e+00, 0.9887242060526682e+00,
            0.9927049442755639e+00, 0.9952987828537434e+00, 0.9969808363488774e+00,
            0.9980670504943989e+00, 0.9987659019591332e+00, 0.9992140557861791e+00,
        ];

        assert::close(&x.iter().map(|&x| d.distribution(x)).collect::<Vec<_>>(), &p, 1e-14);
    }

    #[test]
    fn entropy() {
        assert_eq!(new!(3, 0.5).entropy(), 0.2540725690922956e+01);
    }

    #[test]
    fn kurtosis() {
        assert_eq!(new!(3, 0.5).kurtosis(), 2.0);
    }

    #[test]
    fn mean() {
        assert_eq!(new!(5, 0.5).mean(), 10.0);
    }

    #[test]
    fn modes() {
        assert_eq!(new!(5, 0.5).modes(), vec![8.0]);
    }

    #[test]
    fn skewness() {
        assert_eq!(new!(5, 0.5).skewness(), 0.8944271909999159e+00);
    }

    #[test]
    fn variance() {
        assert_eq!(new!(5, 0.5).variance(), 20.0);
    }
}
